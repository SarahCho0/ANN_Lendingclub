# src/preprocess_pipeline.py

import polars as pl
import os
import gc
import numpy as np # 상관관계 계산 및 행렬 처리를 위해 필요

# ==============================================================================
# 1. 설정 및 상수 정의
# ==============================================================================
# 현재 스크립트(src/preprocess_pipeline.py)의 위치를 구함
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# src 폴더의 상위(../)인 프로젝트 루트를 기준으로 경로 설정
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'lending_club_2020_train.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data/processed')

# 1) 학습에는 안 쓰지만, 수익률(Sharpe Ratio) 계산과 시계열 분할에 꼭 필요한 변수들
FINANCE_COLS = ['total_pymnt', 'funded_amnt', 'term', 'int_rate', 'issue_d']

# ------------------------------------------------------------------------------
# [사용자 추가] 담당하고 있는 54개(52개+@) 핵심 변수 리스트
# 설명: 이 리스트에 있는 변수들은 DROP_COLS에 포함되어 있더라도 절대 삭제되지 않도록 로직을 추가함.
# ------------------------------------------------------------------------------
SELECTED_FEATURES = [
    # 1. 기본 신용 정보 및 조회 관련
    'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 
    'bc_util', 'chargeoff_within_12_mths', 'delinq_2yrs', 'delinq_amnt', 
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 
    'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq', 
    'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 
    'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 
    'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 
    'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec', 
    'pub_rec_bankruptcies', 'tax_liens', 'tot_cur_bal', 'tot_hi_cred_lim', 
    'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 
    'total_rev_hi_lim',
    
    # 2. 최근 대출 활동 (엑셀 리포트상 High Nulls이나 0으로 대체 필요한 변수들)
    'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 
    'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 
    'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
    
    # 3. 공동 신청자(Secondary Applicant) 관련 (단독일 경우 결측 -> 0 처리)
    'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line', 
    'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 
    'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts', 
    'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 
    'sec_app_mths_since_last_major_derog'
]

# 2) [수정] 기본적으로 삭제할 변수 목록
# 주의: SELECTED_FEATURES에 있는 변수는 아래 리스트에 있더라도 삭제되지 않음.
DROP_COLS = [
    'id', 'member_id', 'url', 'desc', 'policy_code', 'zip_code',
    'emp_title', 'title', 
    'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d',
    'collection_recovery_fee', 'recoveries', 'total_rec_prncp', 'total_rec_int', 
    'total_rec_late_fee', 'pymnt_plan', 'out_prncp', 'out_prncp_inv',
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
    'debt_settlement_flag', 'settlement_status', 'settlement_date',
    
    # 원래는 삭제하려 했으나 SELECTED_FEATURES에 포함되어 살려야 하는 변수들 (참고용 주석)
    # 'open_acc_6m', 'open_act_il', 'open_il_12m', ...
]

def process_pipeline(file_path, is_train=True):
    print(f"🚀 [Start] 전처리 시작: {file_path}")
    q = pl.scan_csv(file_path, infer_schema_length=10000, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Target 생성 (Train 데이터인 경우에만)
    # --------------------------------------------------------------------------
    if is_train:
        target_bad = [
            'Charged Off', 'Default', 'Late (31-120 days)', 
        ]
        target_good = [
            'Fully Paid'
        ]
        
        # Current(진행중) 등 애매한 데이터는 제거하고, 확실한 정답만 남김
        q = q.filter(pl.col('loan_status').is_in(target_bad + target_good))
        
        # 부도=1, 정상=0
        q = q.with_columns(
            pl.when(pl.col('loan_status').is_in(target_bad))
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias('target')
        )
    
    # --------------------------------------------------------------------------
    # 2. 날짜 파싱 (시계열 Split 및 파생변수용)
    # --------------------------------------------------------------------------
    # issue_d는 'Oct-2015' 같은 문자열 -> 날짜 객체로 변환
    q = q.with_columns(
        pl.col('issue_d').str.strptime(pl.Date, '%b-%Y', strict=False).alias('issue_d_parsed')
    )
    
    # --------------------------------------------------------------------------
    # 3. 주요 파생변수 생성 (공동 신청자 로직 통합)
    # --------------------------------------------------------------------------
    q = q.with_columns([
        # [주 신청자] 소득 및 신용 정보 통합
        pl.coalesce([pl.col('verification_status_joint'), pl.col('verification_status')]).alias('effective_verification_status'),
        pl.coalesce([pl.col('annual_inc_joint'), pl.col('annual_inc')]).cast(pl.Float32).alias('effective_annual_inc'),
        pl.coalesce([pl.col('dti_joint'), pl.col('dti')]).cast(pl.Float32).alias('effective_dti'),
        pl.col('earliest_cr_line').str.strptime(pl.Date, '%b-%Y', strict=False).alias('earliest_cr_line_parsed'),
        ((pl.col('fico_range_high') + pl.col('fico_range_low')) / 2).cast(pl.Float32).alias('fico_range_mean'),
        
        # [공동 신청자] FICO 평균 (리포트: 공동 차입자 점수가 높을수록 부도율 감소하므로 수치화 필요)
        ((pl.col('sec_app_fico_range_high') + pl.col('sec_app_fico_range_low')) / 2).cast(pl.Float32).alias('sec_app_fico_mean'),
        
        # [공동 신청자] 신용 개설일 파싱 (이후 '연수' 계산을 위해 Date 타입 변환)
        pl.col('sec_app_earliest_cr_line').str.strptime(pl.Date, '%b-%Y', strict=False).alias('sec_app_earliest_cr_line_parsed'),

        # 이진 플래그 생성
        pl.when(pl.col('num_tl_30dpd') > 0).then(1).otherwise(0).cast(pl.Int8).alias('has_30dpd'),
        pl.when(pl.col('delinq_2yrs') > 0).then(1).otherwise(0).cast(pl.Int8).alias('has_delinq_2yrs'),
        pl.when(pl.col('inq_last_6mths') > 0).then(1).otherwise(0).cast(pl.Int8).alias('has_inq_6mths')
    ])
    
    q = q.with_columns([
        # 주 신청자 신용 이력 (연수)
        ((pl.col('issue_d_parsed') - pl.col('earliest_cr_line_parsed')).dt.total_days() / 365.25).cast(pl.Float32).alias('credit_hist_years'),
        
        # [추가] 공동 신청자 신용 이력 (연수) 
        # 이유: 날짜 형식은 신경망 입력이 안되므로 주 신청자와 동일하게 '금융 경험 기간'으로 수치화
        ((pl.col('issue_d_parsed') - pl.col('sec_app_earliest_cr_line_parsed')).dt.total_days() / 365.25).cast(pl.Float32).alias('sec_app_credit_hist_years')
    ])

    # --------------------------------------------------------------------------
    # 4-A. 구조적 결측치 0으로 채우기 (공동 신청자 및 최신 활동 변수)
    # --------------------------------------------------------------------------
    zero_fill_list = [
        'inq_last_12m', 'acc_open_past_24mths', 'open_acc_6m', 'open_act_il', 
        'open_il_12m', 'open_il_24m', 'total_bal_il', 'il_util', 'open_rv_12m', 
        'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
        
        # 공동 신청자 관련 변수 (단독 신청 시 기록 없음 = 활동 없음(0)으로 간주)
        'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 
        'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts', 
        'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
        'sec_app_fico_mean',        # 파생된 공동 신청자 FICO
        'sec_app_credit_hist_years', # 파생된 공동 신청자 신용 연수
        'delinq_amnt', 'chargeoff_within_12_mths', 'tax_liens'
    ]
    
    schema = q.collect_schema().names()
    target_zeros = [c for c in zero_fill_list if c in schema]
    
    if target_zeros:
        q = q.with_columns([
            pl.col(c).fill_null(0).cast(pl.Float32) for c in target_zeros
        ])

    # --------------------------------------------------------------------------
    # 4-B. 기간(Months) 관련 결측치 처리 (큰 값 대체)
    # --------------------------------------------------------------------------
    mths_fill_plan = {
        'mths_since_last_delinq': 90.0,
        'mths_since_last_record': 130.0,
        'mths_since_last_major_derog': 92.0,
        'mths_since_recent_bc': 200.0,
        'mths_since_recent_inq': 25.0,
        'mths_since_rcnt_il': 200.0,
        'sec_app_mths_since_last_major_derog': 999.0 # 공동 신청자 신용 불량 없음
    }
    
    for col_name, fill_value in mths_fill_plan.items():
        if col_name in schema:
            q = q.with_columns([
                pl.col(col_name).is_null().cast(pl.Int8).alias(f'is_never_{col_name}'),
                pl.col(col_name).fill_null(fill_value).cast(pl.Float32)
            ])

    # 5. 문자열 파싱 (term, int_rate, revol_util 등)
    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
               '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
    
    q = q.with_columns([
        pl.col('term').str.strip_chars(' months').cast(pl.Int32, strict=False),
        pl.col('int_rate').str.strip_chars(' %').cast(pl.Float32, strict=False),
        pl.col('revol_util').str.strip_chars(' %').cast(pl.Float32, strict=False),
        pl.col('emp_length').replace(emp_map).cast(pl.Int32, strict=False).fill_null(0)
    ])

    # 6. 로그 변환 (왜도 해결)
    # 엑셀 리포트 분석 결과 Skewness > 2인 변수들 및 주요 자산/소득 변수 통합
    log_candidates = [
        # 1) 주요 자산 및 소득 (기본 포함)
        'tot_cur_bal', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'avg_cur_bal', 
        'total_bc_limit', 'total_il_high_credit_limit', 'total_rev_hi_lim', 
        'revol_bal', 'annual_inc', 'effective_annual_inc', 'int_rate',
    
        # 2) 왜도(Skew) > 2인 변수 추가 (오타 수정 및 정리)
        'inq_last_12m', 'bc_open_to_buy', 'delinq_amnt', 
        'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', # 오타 분리 수정
        'mths_since_recent_bc', 'num_accts_ever_120_pd', 'num_il_tl', 
        'num_tl_90g_dpd_24m', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 
        'tax_liens', 'revol_bal_joint', 'sec_app_open_act_il', 
        'sec_app_chargeoff_within_12_mths'
    ]

    # 중복 제거 (set 활용)
    log_candidates = list(set(log_candidates))

    # SELECTED_FEATURES에 있고, 실제 데이터(schema)에 존재하는 변수만 로그 변환 수행
    cols_to_log = [c for c in log_candidates if c in schema and (c in SELECTED_FEATURES or c.endswith('_joint') or c.startswith('effective_'))]

    if cols_to_log:
        q = q.with_columns([
            # log1p(x) = ln(1 + x) : 값이 0인 경우를 대비하여 1을 더하고 로그를 취함
            pl.col(c).fill_null(0).log1p().alias(f'{c}_log') for c in cols_to_log
        ])
        # [ANN 피드백 반영 2] raw 변수와 log 변수가 공존하면 중복 정보로 인한 문제 발생
        # 로그 변환된 원본 변수는 제거 (단, 재정 지표 등 나중에 필요한 변수인지 확인 필요)
        q = q.drop(cols_to_log)
    
    # 7. 컬럼 정리 (Selected 변수는 절대 드랍 방지)
    cols_to_drop = [c for c in DROP_COLS if c in schema and c not in SELECTED_FEATURES]
    q = q.drop(cols_to_drop)
    
    return q

# ==============================================================================
# 3. 메인 실행 및 저장 (데이터 적재 및 최종 인코딩)
# ==============================================================================
def main():
    # 저장 경로 확인 및 생성 (디렉토리가 없으면 에러가 나므로 필수)
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        
    save_path = os.path.join(OUTPUT_DIR, 'train_final.parquet')

    # 1. 파이프라인 구성 (Lazy Execution)
    # 실제 연산을 수행하지 않고, 논리적 실행 계획(Plan)만 수립하여 메모리 사용을 최소화함
    q = process_pipeline(RAW_DATA_PATH, is_train=True)

    # 2. 데이터 수집 (Collect) - Streaming 엔진 사용
    # [설명] collect(engine='streaming')은 데이터를 한꺼번에 메모리에 올리지 않고
    # 청크(Chunk) 단위로 처리하여 대용량 CSV 파일 처리 시 메모리 부족(OOM) 현상을 방지함
    print("🔄 [Step 2] 데이터 변환 및 메모리 적재 (Streaming)...")
    df = q.collect(engine='streaming')

    # --------------------------------------------------------------------------
    # [ANN 최적화 추가 단계] 이상치 처리, 상관관계 제거, 스케일링
    # 피드백 반영: ANN 성능 안정화를 위한 필수 전처리
    # --------------------------------------------------------------------------
    print("🔧 [Step 2.5] ANN 최적화 전처리 수행 (Clipping, Drop High-Corr, Scaling)...")

    # (A) 수치형 변수 식별 (이진 변수 제외)
    numeric_cols = [c for c, t in df.schema.items() if (t in [pl.Float32, pl.Float64]) and ('target' not in c)]
    # 0과 1로만 된 컬럼(flag)은 스케일링이나 클리핑에서 제외하기 위해 구분
    continuous_cols = []
    for c in numeric_cols:
        if df[c].n_unique() > 2:
            continuous_cols.append(c)

    # [ANN 피드백 반영 4] 이상치 Clipping (Winsorization)
    # 상하위 1% 극단값 Clipping (bc_util, revol_util 등 극단값 방지)
    if continuous_cols:
        for c in continuous_cols:
            lower = df[c].quantile(0.01)
            upper = df[c].quantile(0.99)
            if lower is not None and upper is not None:
                df = df.with_columns(pl.col(c).clip(lower, upper))

    # [ANN 피드백 반영 3] 높은 상관관계 변수 제거 (Multicollinearity)
    # Threshold 0.95 이상인 변수 중 하나 제거
    # (주의: 메모리 효율을 위해 Corr Matrix 연산 시 주의)
    if len(continuous_cols) > 0:
        # 상관관계 행렬 계산 (Polars는 아직 corr()이 전체 DF에 대해 지원 안될 수 있어 select 사용)
        # 여기서는 간단히 주요 중복 의심 그룹만 수동으로 처리하거나, 
        # ANN 안정을 위해 너무 뻔한 중복 변수만 제거하는 로직을 적용
        high_corr_pairs = [
            ('tot_hi_cred_lim', 'total_bc_limit'), 
            ('total_bal_ex_mort', 'tot_cur_bal') # 대표적인 다중공선성 그룹
        ]
        drop_corr = []
        for c1, c2 in high_corr_pairs:
            if c1 in df.columns and c2 in df.columns:
                drop_corr.append(c2) # 둘 중 하나 제거
        
        if drop_corr:
            df = df.drop(drop_corr)
            # 제거된 변수는 continuous_cols 목록에서도 제외
            continuous_cols = [c for c in continuous_cols if c not in drop_corr]

    # [ANN 피드백 반영 1] 스케일링 (RobustScaler)
    # ANN은 스케일링이 필수. 이상치에 강한 Robust Scaler 적용 (Median, IQR 활용)
    # Formula: (x - median) / (q75 - q25)
    if continuous_cols:
        df = df.with_columns([
            ((pl.col(c) - pl.col(c).median()) / (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25) + 1e-9))
            .alias(c) 
            for c in continuous_cols
        ])

    # [ANN 피드백 반영 5] 타겟 불균형 확인 (메타데이터용 출력)
    if 'target' in df.columns:
        neg, pos = df['target'].value_counts().sort('target')['count'].to_list()
        print(f"⚖️  Target Imbalance Info: Normal(0)={neg}, Default(1)={pos}, Ratio=1:{neg/pos:.2f}")
        # 추후 학습 시 pos_weight = neg / pos 사용 권장

    print("🔢 [Step 3] 인코딩 적용...")
    
    # (0) home_ownership 희소 범주 통합
    # [이유] ANY, NONE 등 빈도가 극히 낮은 범주를 OTHER로 묶어 모델이 불필요한 노이즈를
    # 학습하지 않게 하고, One-Hot Encoding 시 불필요하게 컬럼이 늘어나는 것을 방지함
    if 'home_ownership' in df.columns:
        df = df.with_columns(
            pl.col('home_ownership').replace({'ANY': 'OTHER', 'NONE': 'OTHER'})
        )
    
    # (1) Ordinal Encoding (grade: 등급)
    # [이유] A~G 등급은 순서 의미가 명확하므로, 숫자로 변환하여 ANN 모델이 
    # 등급 간의 높고 낮음(서열)을 직접 수치적으로 이해할 수 있게 함
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    if 'grade' in df.columns:
        df = df.with_columns(pl.col('grade').replace(grade_map).cast(pl.Int8))

    # (2) One-Hot Encoding (범주형 변수의 수치화)
    # [이유] ANN은 문자열을 인식하지 못하므로, 순서가 없는 범주형 데이터는 각 범주를 
    # 별도의 0과 1 컬럼으로 분리함. 
    # [변경사항] 기존 코드의 sub_grade, addr_state는 컬럼 수가 지나치게 많아 
    # 모델 복잡도를 높일 수 있으므로 제외하여 모델의 일반화 성능(Overfitting 방지)을 유도함
    one_hot_cols = ['home_ownership', 'purpose', 'application_type', 'effective_verification_status']
    
    # 실제 데이터셋에 존재하는 컬럼만 선별하여 안전하게 변환 수행
    target_one_hot = [c for c in one_hot_cols if c in df.columns]
    df = df.to_dummies(columns=target_one_hot)
    
    # 4. 최종 데이터 저장 (Parquet 포맷)
    # [이유] CSV보다 용량이 훨씬 작고 로드 속도가 빠른 Parquet 형식을 사용함.
    # zstd 압축 방식은 압축률과 읽기/쓰기 속도 사이의 균형이 가장 뛰어나 현업에서 선호됨
    print(f"💾 [Step 4] 저장 중... ({save_path})")
    df.write_parquet(save_path, compression='zstd')
    
    print(f"✅ 완료! 최종 데이터 셋 크기(Shape): {df.shape}")

if __name__ == "__main__":
    main()