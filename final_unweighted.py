# preprocessing is not fully included

import polars as pl
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# STEP 1: 데이터 로딩 및 전처리
# ============================================================================
# 📌 목표: CSV 파일에서 대출 데이터 로딩 → 결측치 처리 → 변수 생성 → 정제

df = pd.read_csv(
    './lending_club_2020_train.csv',
    na_values=['Loans that do not meet the credit policy', 'N', 'w'],
    low_memory=False,
    encoding='utf-8-sig',  # BOM 처리
    engine='python'  # C engine 대신 Python engine 사용 (더 안정적)
)
print(f"📊 CSV 로드 완료: shape={df.shape}, columns={len(df.columns)}")
print(f"   첫 5개 컬럼: {list(df.columns[:5])}")
df = pl.from_pandas(df)

# 건들지 말기!!! 금융계산에 사용됨!!, drop_cols 넣지 말기!!!!
FINANCE_COLS = [
    'id', 'member_id',
    'loan_status', 'issue_d', 'term',
    'int_rate', 'installment', 'grade', 'sub_grade',
    'funded_amnt', 'funded_amnt_inv',
    'total_pymnt', 'total_rec_prncp', 'total_rec_int',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d'
]

# 사용자가 제공한 핵심 변수 목록 (삭제 금지용)
USER_RETAIN_VARS = [
    'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
    'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt',
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
    'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd',
    'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl',
    'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
    'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
    'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort',
    'total_bc_limit', 'total_il_high_credit_limit', 'revol_bal_joint',
    'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
    'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
    'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts',
    'sec_app_chargeoff_within_12_mths', 'acc_now_delinq', 'fico_range_high'
]

# 기존 SELECTED_FEATURES (보존 목록)
SELECTED_FEATURES = USER_RETAIN_VARS + [
    # 필요시 추가 보존 변수 넣기
]

# 기본 삭제 후보 (원본에서 불필요한 미래 정보 등)
def get_cols_to_drop():
    drop_cols = [
        'url', 'desc', 'emp_title', 'title', 'zip_code', 'addr_state', 'policy_code',
        'pymnt_plan', 'issue_d_parsed', 'earliest_cr_line_parsed',
        'roi_pct', 'last_fico_range_high', 'last_fico_range_low',
        'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'last_pymnt_d', 'next_pymnt_d',
        'last_credit_pull_d', 'out_prncp', 'out_prncp_inv',
        'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
        'payment_plan_start_date', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
        'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
        'hardship_last_payment_amount', 'settlement_status', 'settlement_date'
    ]
    # 절대 삭제 금지: FINANCE_COLS 및 SELECTED_FEATURES/USER_RETAIN_VARS
    safe_drop = [c for c in drop_cols if c not in FINANCE_COLS and c not in SELECTED_FEATURES]
    return safe_drop

# ------------------------------------------------------------------------------
# 전처리 함수
# ------------------------------------------------------------------------------
def process_pipeline(df_input: pl.DataFrame, is_train: bool = True) -> pl.DataFrame:
    """
    입력: pl.DataFrame (읽어온 원본)
    출력: 전처리된 pl.DataFrame (즉시 계산된 상태)
    """
    print(f"🚀 [Start] 전처리 시작: 입력 shape {df_input.shape}")
    print(f"   사용 가능한 컬럼: {list(df_input.columns[:10])}...")
    
    q = df_input
    
    # -------------------------
    # 0.5) loan_status 컬럼 존재 확인
    # -------------------------
    if 'loan_status' not in q.columns:
        print(f"❌ 에러: loan_status 컬럼이 없습니다!")
        print(f"   전체 컬럼 리스트: {list(q.columns)}")
        raise ValueError("loan_status column not found")

    # -------------------------
    # 1) target 정의 (Train)
    # -------------------------
    if is_train:
        target_bad = ['Charged Off', 'Default', 'Late (31-120 days)']
        target_good = ['Fully Paid']
        q = q.filter(pl.col('loan_status').is_in(target_bad + target_good))
        q = q.with_columns(
            pl.when(pl.col('loan_status').is_in(target_bad)).then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.Int8).alias('target')
        )

    # -------------------------
    # 2) 날짜 파싱, 기본 파생
    # -------------------------
    q = q.with_columns([
        pl.col('issue_d').str.strptime(pl.Date, '%b-%Y', strict=False).alias('issue_d_parsed'),
        pl.col('earliest_cr_line').str.strptime(pl.Date, '%b-%Y', strict=False).alias('earliest_cr_line_parsed'),
        pl.col('sec_app_earliest_cr_line').str.strptime(pl.Date, '%b-%Y', strict=False).alias('sec_app_earliest_cr_line_parsed'),
        pl.col('term').str.strip_chars(' months').cast(pl.Int32, strict=False),
        pl.col('int_rate').str.strip_chars(' %').cast(pl.Float32, strict=False),
        pl.col('revol_util').str.strip_chars(' %').cast(pl.Float32, strict=False),
        # emp_length 정제
        pl.col('emp_length').str.replace('< 1 year', '0').str.replace('10\\+ years', '10')
            .str.extract(r'(\d+)', 1).cast(pl.Int32, strict=False).fill_null(0).alias('emp_length_int')
    ])

    # -------------------------
    # 3) Joint applicant 통합 변수
    # -------------------------
    q = q.with_columns([
        pl.coalesce([pl.col('annual_inc_joint'), pl.col('annual_inc')]).cast(pl.Float32).alias('effective_annual_inc'),
        pl.coalesce([pl.col('dti_joint'), pl.col('dti')]).cast(pl.Float32).alias('effective_dti'),
        pl.coalesce([pl.col('verification_status_joint'), pl.col('verification_status')]).alias('effective_verification_status'),
        (pl.col('annual_inc') / 12).alias('monthly_inc'),
        ((pl.col('sec_app_fico_range_high') + pl.col('sec_app_fico_range_low')) / 2).cast(pl.Float32).alias('sec_app_fico_mean'),
    ])

    # -------------------------
    # 4) credit history 연수 파생
    # -------------------------
    q = q.with_columns([
        ((pl.col('issue_d_parsed') - pl.col('earliest_cr_line_parsed')).dt.total_days() / 365.25).cast(pl.Float32).alias('credit_hist_years'),
        ((pl.col('issue_d_parsed') - pl.col('sec_app_earliest_cr_line_parsed')).dt.total_days() / 365.25).cast(pl.Float32).alias('sec_app_credit_hist_years'),
        ((pl.col('fico_range_low') + pl.col('fico_range_high')) / 2).alias('fico_range_average')
    ])

    # -------------------------
    # 5) 구조적 결측 0으로 채우기 (공동 신청자 등)
    # -------------------------
    zero_fill_list = [
        'inq_last_12m', 'acc_open_past_24mths', 'open_acc_6m', 'open_act_il',
        'open_il_12m', 'open_il_24m', 'total_bal_il', 'il_util', 'open_rv_12m',
        'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
        # 공동 신청자
        'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
        'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts',
        'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
        'sec_app_fico_mean', 'sec_app_credit_hist_years',
        # 기타
        'delinq_amnt', 'chargeoff_within_12_mths', 'tax_liens', 'revol_bal_joint',
        'acc_now_delinq'
    ]
    existing_zero_cols = [c for c in zero_fill_list if c in q.schema]
    if existing_zero_cols:
        q = q.with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in existing_zero_cols])

    # -------------------------
    # 6) 기간형(mths_since_...) 결측 처리: '결측==없음' -> 큰 값으로 대체 + is_never flag
    # -------------------------
    mths_fill_plan = {
        'mths_since_last_delinq': 90.0,
        'mths_since_last_record': 130.0,
        'mths_since_last_major_derog': 92.0,
        'mths_since_recent_bc': 200.0,
        'mths_since_recent_inq': 25.0,
        'mths_since_rcnt_il': 200.0,
        # 추가로 꼭 처리해야 할 연체 관련 mths 변수들
        'mths_since_recent_bc_dlq': 999.0,
        'mths_since_recent_revol_delinq': 999.0,
        'sec_app_mths_since_last_major_derog': 999.0
    }
    for col_name, fill_value in mths_fill_plan.items():
        if col_name in q.schema:
            q = q.with_columns([
                pl.col(col_name).is_null().cast(pl.Int8).alias(f'is_never_{col_name}'),
                pl.col(col_name).fill_null(fill_value).cast(pl.Float32)
            ])

    # 그 외 'mths_since' 계열은 기본적으로 상위 percentile 값으로 채움
    mths_cols = [c for c in q.schema.names() if 'mths_since' in c and c not in mths_fill_plan]
    for col_name in mths_cols:
        q = q.with_columns([
            pl.col(col_name).is_null().cast(pl.Int8).alias(f'is_never_{col_name}'),
            pl.col(col_name).fill_null(pl.col(col_name).quantile(0.95)).cast(pl.Float32)
        ])

    # -------------------------
    # 7) 범주형/플래그 처리
    # -------------------------
    # initial_list_status: w -> 1, f -> 0
    if 'initial_list_status' in q.schema:
        q = q.with_columns([
            pl.when(pl.col('initial_list_status') == 'w').then(pl.lit(1))
              .when(pl.col('initial_list_status') == 'f').then(pl.lit(0))
              .otherwise(pl.lit(0)).cast(pl.Int8).alias('initial_list_status')
        ])

    # home_ownership: 희소값 -> UNKNOWN, 이후 one-hot 처리는 main에서 수행
    if 'home_ownership' in q.schema:
        q = q.with_columns([
            pl.when(pl.col('home_ownership').str.to_uppercase().is_in(['ANY','OTHER','NONE']))
              .then(pl.lit('UNKNOWN'))
              .otherwise(pl.col('home_ownership').str.to_uppercase())
              .alias('home_ownership')
        ])

    # is_joint_app flag
    if 'application_type' in q.schema:
        q = q.with_columns([
            (pl.col('application_type') == 'Joint App').cast(pl.Int8).alias('is_joint_app')
        ])

    # acc_now_delinq: 0/1로 보장
    if 'acc_now_delinq' in q.schema:
        q = q.with_columns([
            pl.col('acc_now_delinq').fill_null(0).cast(pl.Int8)
        ])

    # pub_rec_bankruptcies: cap at 2 -> 0/1/2+
    if 'pub_rec_bankruptcies' in q.schema:
        q = q.with_columns([
            pl.when(pl.col('pub_rec_bankruptcies') > 2).then(2)
              .otherwise(pl.col('pub_rec_bankruptcies')).cast(pl.Int8).alias('pub_rec_bankruptcies_cat')
        ])

    # red flag 이진화: 연체/부도 관련(희소)
    flag_list = ['num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_120dpd_2m']
    for f in flag_list:
        if f in q.schema:
            q = q.with_columns([
                (pl.col(f) > 0).cast(pl.Int8).alias(f)
            ])

    # -------------------------
    # 8) 로그 변환 후보들 (원본 대체)
    # -------------------------
    log_candidates = [
        'tot_cur_bal', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'avg_cur_bal',
        'total_bc_limit', 'total_il_high_credit_limit', 'total_rev_hi_lim',
        'revol_bal', 'annual_inc', 'effective_annual_inc', 'int_rate',
        'inq_last_12m', 'bc_open_to_buy', 'delinq_amnt',
        'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc',
        'num_accts_ever_120_pd', 'num_il_tl', 'num_tl_90g_dpd_24m',
        'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 'tax_liens',
        'revol_bal_joint'
    ]
    cols_to_log = [c for c in log_candidates if c in q.schema]
    if cols_to_log:
        q = q.with_columns([
            pl.when(pl.col(c).is_null()).then(pl.lit(0.0))
              .otherwise(pl.col(c))
              .map_elements(lambda x: float(np.log1p(x)) if x is not None else 0.0)  # 안전성 확보
              .alias(f'log_{c}')
            for c in cols_to_log
        ])
        # 중복 방지: 로그로 대체하기로 결정했으면 원본 제거(원하면 주석 처리하고 원본 유지)
        q = q.drop(cols_to_log)

    # -------------------------
    # 9) pct_tl_nvr_dlq: scaled / squared 파생 (권장)
    # -------------------------
    if 'pct_tl_nvr_dlq' in q.schema:
        q = q.with_columns([
            (pl.col('pct_tl_nvr_dlq') / 100.0).alias('pct_tl_nvr_dlq_frac'),
            ( (pl.col('pct_tl_nvr_dlq') / 100.0) ** 2 ).alias('pct_tl_nvr_dlq_sq')
        ])

    # -------------------------
    # 10) 기타 파생 / flags
    # -------------------------
    if 'tax_liens' in q.schema:
        q = q.with_columns([(pl.col('tax_liens') > 0).cast(pl.Int8).alias('has_tax_liens')])

    if 'pub_rec' in q.schema:
        q = q.with_columns([(pl.col('pub_rec') > 0).cast(pl.Int8).alias('has_pub_rec')])

    # -------------------------
    # 11) 컬럼 드랍: 안전드랍만 (사용자 보존 목록 제외)
    # -------------------------
    cols_to_drop = get_cols_to_drop()
    current_cols = q.schema.names()
    final_drop_list = [c for c in cols_to_drop if c in current_cols]
    if final_drop_list:
        q = q.drop(final_drop_list)

    print(f"✅ [Complete] 전처리 완료: 최종 shape {q.shape}")
    return q



# ============================================================================
# STEP 2: 금융 계산 (Risk-free rate, IRR, Sample weights)
# ============================================================================
# 📌 목표: 국채 금리 매핑 → 실제 IRR 계산 → 샘플 가중치 생성
#

def load_treasury_rates():
    """국채 금리 데이터 로드 (GS3, GS5)"""
    try:
        gs3 = pd.read_csv("GS3.csv")
        gs5 = pd.read_csv("GS5.csv")

        gs3.columns = [c.strip().lower() for c in gs3.columns]
        gs5.columns = [c.strip().lower() for c in gs5.columns]

        if "observation_date" in gs3.columns:
            gs3.rename(columns={"observation_date": "DATE"}, inplace=True)
        if "observation_date" in gs5.columns:
            gs5.rename(columns={"observation_date": "DATE"}, inplace=True)

        gs3["DATE"] = pd.to_datetime(gs3["DATE"], errors="coerce")
        gs5["DATE"] = pd.to_datetime(gs5["DATE"], errors="coerce")

        if "gs3" in gs3.columns:
            gs3.rename(columns={"gs3": "GS3"}, inplace=True)
        elif len(gs3.columns) > 1:
            col_name = [c for c in gs3.columns if c != "DATE"][0]
            gs3.rename(columns={col_name: "GS3"}, inplace=True)

        if "gs5" in gs5.columns:
            gs5.rename(columns={"gs5": "GS5"}, inplace=True)
        elif len(gs5.columns) > 1:
            col_name = [c for c in gs5.columns if c != "DATE"][0]
            gs5.rename(columns={col_name: "GS5"}, inplace=True)

        gs3["GS3"] = pd.to_numeric(gs3["GS3"], errors="coerce")
        gs5["GS5"] = pd.to_numeric(gs5["GS5"], errors="coerce")

        gs3 = gs3.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE").resample("D").interpolate().reset_index()
        gs5 = gs5.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE").resample("D").interpolate().reset_index()

        print(f"✅ 국채 금리 로드 성공: GS3({len(gs3)}건), GS5({len(gs5)}건)")
        return gs3, gs5

    except Exception as e:
        print(f"⚠️ 국채 금리 로드 실패: {e}")
        return None, None


def map_risk_free_rate(df_pl):
    """대출 기간(term)에 맞춰 무위험 이자율(국채) 매핑"""
    gs3, gs5 = load_treasury_rates()
    df = df_pl.to_pandas()

    if gs3 is None or gs5 is None:
        print("⚠️ 국채 금리 파일을 읽지 못해 기본값(2%)을 사용합니다.")
        df["risk_free_rate"] = 0.02
        return pl.from_pandas(df)

    if "risk_free_rate" not in df.columns:
        df["risk_free_rate"] = np.nan

    df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["term_str"] = df["term"].astype(str).str.strip()

    mask_36 = df["term_str"].str.contains("36", na=False) & df["issue_d_parsed"].notna()
    mask_60 = df["term_str"].str.contains("60", na=False) & df["issue_d_parsed"].notna()

    if mask_36.any():
        base_36 = df.loc[mask_36, ["issue_d_parsed"]].copy()
        base_36["__idx__"] = base_36.index
        base_36 = base_36.sort_values("issue_d_parsed")
        merged_36 = pd.merge_asof(base_36, gs3.sort_values("DATE"), left_on="issue_d_parsed", right_on="DATE", direction="backward")
        df.loc[merged_36["__idx__"].to_numpy(), "risk_free_rate"] = merged_36["GS3"].to_numpy() / 100.0

    if mask_60.any():
        base_60 = df.loc[mask_60, ["issue_d_parsed"]].copy()
        base_60["__idx__"] = base_60.index
        base_60 = base_60.sort_values("issue_d_parsed")
        merged_60 = pd.merge_asof(base_60, gs5.sort_values("DATE"), left_on="issue_d_parsed", right_on="DATE", direction="backward")
        df.loc[merged_60["__idx__"].to_numpy(), "risk_free_rate"] = merged_60["GS5"].to_numpy() / 100.0

    df.drop(columns=["issue_d_parsed", "term_str"], inplace=True, errors="ignore")
    df["risk_free_rate"] = df["risk_free_rate"].fillna(0.02)
    return pl.from_pandas(df)


def _solve_monthly_rate_annuity(principal: float, pmt: float, n_months: int) -> float:
    if principal <= 0 or pmt <= 0 or n_months <= 0:
        return np.nan
    n = int(n_months)
    def npv(r):
        if r <= -0.9999:
            return np.nan
        d = 1.0 + r
        if abs(r) < 1e-12:
            pv = pmt * n
        else:
            pv = pmt * (1.0 - d ** (-n)) / r
        return -principal + pv
    lo, hi = -0.999, 5.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.isnan(f_lo) or np.isnan(f_hi):
        return np.nan
    k = 0
    while f_lo * f_hi > 0 and k < 20:
        hi *= 1.5
        f_hi = npv(hi)
        if np.isnan(f_hi):
            return np.nan
        k += 1
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if np.isnan(f_mid):
            return np.nan
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _solve_monthly_irr_optimized(row_tuple):
    principal, installment, n_months, total_inflow = row_tuple
    if principal <= 0 or total_inflow <= 0 or n_months <= 0:
        return np.nan
    n = int(n_months)
    reg_n = max(0, n - 1)
    balloon = total_inflow - (installment * reg_n)
    if balloon <= 0:
        pmt_new = total_inflow / n
        r_m = _solve_monthly_rate_annuity(principal, pmt_new, n)
        if np.isnan(r_m):
            return np.nan
        return (1.0 + r_m) ** 12 - 1.0
    def npv(r):
        if r <= -0.9999:
            return np.nan
        d = 1.0 + r
        if reg_n == 0:
            pv_reg = 0.0
        elif abs(r) < 1e-12:
            pv_reg = installment * reg_n
        else:
            pv_reg = installment * (1.0 - d ** (-reg_n)) / r
        pv_balloon = balloon / (d ** n)
        return -principal + pv_reg + pv_balloon
    lo, hi = -0.999, 5.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.isnan(f_lo) or np.isnan(f_hi):
        return np.nan
    k = 0
    while f_lo * f_hi > 0 and k < 20:
        hi *= 1.5
        f_hi = npv(hi)
        if np.isnan(f_hi):
            return np.nan
        k += 1
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if np.isnan(f_mid):
            return np.nan
        if abs(f_mid) < 1e-8:
            return (1.0 + mid) ** 12 - 1.0
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (1.0 + 0.5 * (lo + hi)) ** 12 - 1.0


def calculate_actual_irr(df_pl):
    print("⏳ actual_irr 계산 중...")
    df = df_pl.to_pandas()
    if "actual_irr" in df.columns:
        df.drop(columns=["actual_irr"], inplace=True)
    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    last_dt = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    df["n_months"] = ((last_dt - issue_dt).dt.days / 30.4375).round().clip(lower=1).fillna(0).astype(int)
    if "total_pymnt" not in df.columns:
        df["total_pymnt"] = (
            pd.to_numeric(df["total_rec_prncp"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["total_rec_int"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["recoveries"], errors="coerce").fillna(0.0)
        )
    df["funded_amnt"] = pd.to_numeric(df["funded_amnt"], errors="coerce")
    df["installment"] = pd.to_numeric(df["installment"], errors="coerce")
    df["total_pymnt"] = pd.to_numeric(df["total_pymnt"], errors="coerce")
    terminal = df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])
    df["_k_principal"] = df["funded_amnt"].round(2)
    df["_k_install"] = df["installment"].round(2)
    df["_k_total"] = df["total_pymnt"].round(2)
    key_cols = ["_k_principal", "_k_install", "n_months", "_k_total"]
    valid = (
        terminal
        & df["_k_principal"].notna() & (df["_k_principal"] > 0)
        & df["_k_install"].notna() & (df["_k_install"] >= 0)
        & df["_k_total"].notna() & (df["_k_total"] > 0)
        & (df["n_months"] > 0)
    )
    target_df = df.loc[valid, key_cols].copy()
    unique_cases = target_df.drop_duplicates()
    print(f"   - 계산 대상: {len(target_df):,}건 / 유니크: {len(unique_cases):,}건")
    records = unique_cases.to_records(index=False)
    results = []
    for row in tqdm(records, desc="IRR 계산"):
        results.append(_solve_monthly_irr_optimized(row))
    unique_cases["actual_irr"] = np.array(results, dtype=float)
    df = df.join(unique_cases.set_index(key_cols)["actual_irr"], on=key_cols)
    
    # 🔥 IRR 범위 제한 (loan_status 및 int_rate 기반)
    # int_rate를 비율로 변환 (백분율인 경우)
    if "int_rate" in df.columns:
        # int_rate가 백분율(>1)이면 비율로 변환
        int_rate_decimal = df["int_rate"].copy()
        if df["int_rate"].mean() > 1:
            int_rate_decimal = df["int_rate"] / 100.0
        
        # loan_status별 IRR 제한
        charged_off_mask = df["loan_status"] == "Charged Off"
        fully_paid_mask = df["loan_status"] == "Fully Paid"
        
        # 부도(Charged Off): -1 < IRR < int_rate
        if charged_off_mask.any():
            df.loc[charged_off_mask, "actual_irr"] = df.loc[charged_off_mask].apply(
                lambda row: np.clip(
                    row["actual_irr"], 
                    None, 
                    int_rate_decimal.loc[row.name] if pd.notna(int_rate_decimal.loc[row.name]) else 0.35
                ) if pd.notna(row["actual_irr"]) else row["actual_irr"],
                axis=1
            )
        
        # 완전 상환(Fully Paid): 0 < IRR < 0.6
        if fully_paid_mask.any():
            df.loc[fully_paid_mask, "actual_irr"] = np.clip(
                df.loc[fully_paid_mask, "actual_irr"],
                None,
                0.35
            )
    else:
        # int_rate가 없으면 기본 클리핑
        df["actual_irr"] = df["actual_irr"].clip(lower=None, upper=0.6)
    
    df.drop(columns=["_k_principal", "_k_install", "_k_total", "n_months"], inplace=True, errors="ignore")
    return pl.from_pandas(df)


def compute_sample_weights(df_pl, default_penalty=1.5):
    df = df_pl.to_pandas()
    weights = np.ones(len(df))
    mask_default = (df["target"] == 1)
    if "loan_amnt" in df.columns:
        amt = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0.0).clip(lower=0.0)
        median_amount = float(np.nanmedian(amt.values))
        if median_amount <= 0:
            median_amount = 1.0
        denom = np.log1p(median_amount)
        if denom <= 0:
            denom = 1.0
        amount_factor = np.log1p(amt) / denom
        amount_factor = amount_factor.clip(lower=0.5, upper=2.0)
        weights[mask_default] = default_penalty * amount_factor[mask_default]
    else:
        weights[mask_default] = default_penalty
    return weights

print("\n" + "="*80)
print("🔧 데이터 전처리 시작")
print("="*80)

# 기본 전처리 실행
df2 = process_pipeline(df, is_train=True)

# ============================================================================
# STEP 2: 금융 계산 실행
# ============================================================================

print("\n" + "="*80)
print("💰 금융 계산 시작")
print("="*80)

# STEP 2-1. Risk-free rate 매핑
print("\n[1/3] Risk-free rate 매핑...")
df2 = map_risk_free_rate(df2)

# STEP 2-2. Actual IRR 계산
print("\n[2/3] Actual IRR 계산...")
df3 = calculate_actual_irr(df2)

# STEP 2-3. Sample weight 계산
print("\n[3/3] Sample weight 계산...")
weights = compute_sample_weights(df3, default_penalty=1.5)
df3 = df3.with_columns(pl.Series("sample_weight", weights))
mean_weight = df3.select(pl.col("sample_weight").mean()).item()
df3 = df3.with_columns((pl.col("sample_weight") / mean_weight).alias("sample_weight"))

# STEP 2-4. int_rate_spread 계산
df3_pd = df3.to_pandas()
if "int_rate" in df3_pd.columns:
    if df3_pd["int_rate"].mean() > 1:
        df3_pd["int_rate_spread"] = (df3_pd["int_rate"] / 100.0) - df3_pd["risk_free_rate"]
    else:
        df3_pd["int_rate_spread"] = df3_pd["int_rate"] - df3_pd["risk_free_rate"]
df3 = pl.from_pandas(df3_pd)

print("\n" + "="*80)
print("📊 최종 결과")
print("="*80)
print(f"최종 shape: {df3.shape}")
print(f"\nTarget 분포:")
print(df3.select('target').to_pandas()['target'].value_counts())

s = pd.to_numeric(df3_pd["actual_irr"], errors="coerce")
valid = s[np.isfinite(s)]
print(f"\nactual_irr 통계:")
print(f"  count: {len(valid):,}")
print(f"  mean : {valid.mean():.4f}")
print(f"  std  : {valid.std():.4f}")
print(f"  min  : {valid.min():.4f}")
print(f"  max  : {valid.max():.4f}")

print("\n✅ 전처리 완료! df3 사용 가능")

# ============================================================================
# STEP 3: Feature 선택 및 중요도 계산
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("🔍 Feature 선택 및 중요도 계산")
print("="*80)

# STEP 3-1. 전체 Feature 목록 수집 (One-hot 인코딩 포함)
# ============================================================================
print("\n📋 전체 Feature 수집 중...")
# STEP 3-2. 기본 Feature 리스트 선정
# ============================================================================
# 📌 목표: 학습에 사용할 변수 선택 → Random Forest로 중요도 계산 → 상위 30개 선택

# ============================================================================
# ⚙️ [조절 포인트 1] Feature 변수 선택
# ============================================================================
# 📝 설명: 아래 리스트에 포함된 변수들만 모델 학습에 사용됨
# 🔧 조절 방법:
#    - 변수 추가: BASE_FEATURES 리스트에 변수명 추가
#    - 변수 제거: BASE_FEATURES 리스트에서 변수명 삭제
#    - 예시: 'log_annual_inc' 변수 추가하려면 리스트에 추가

BASE_FEATURES = [ # 이번 학습에 잠시 뺀 것 여기에 표시 ()
    # 대출 기본 정보
    'loan_amnt', 'term', 'installment', 
    # 신용 정보
    'fico_range_low', 'revol_bal', 'revol_util',
    'total_acc', 'open_acc', 'pub_rec_bankruptcies',
    # 신용 이력
    'emp_length_int', 'credit_hist_years',
    'inq_last_6mths', 'delinq_2yrs',
    # 파생 변수
    'effective_annual_inc', 'effective_dti',
    'initial_list_status', 'is_joint_app',
    'has_tax_liens', 'has_pub_rec',
    # 로그화
    'log_annual_inc','log_int_rate',
    # 플래그 변수들
    'is_never_mths_since_last_delinq', 'is_never_mths_since_last_record',
    'is_never_mths_since_last_major_derog', 'is_never_mths_since_recent_inq',
    # 추가
    'num_tl_30dpd', 'num_tl_120dpd_2m',
]

# One-hot 인코딩된 변수들 자동 추가
df3_pd_cols = df3.columns
for col in df3_pd_cols:
    if col.startswith(('home_ownership_', 'verification_status_', 'purpose_')):
        if col not in BASE_FEATURES:
            BASE_FEATURES.append(col)

# 실제 존재하는 변수만 선택
ALL_FEATURES = [f for f in BASE_FEATURES if f in df3_pd_cols]
print(f"✅ 전체 후보 Feature: {len(ALL_FEATURES)}개")

# STEP 3-3. Random Forest로 Feature 중요도 계산
# ============================================================================
print("\n" + "="*80)
print("🌲 Random Forest Feature Importance 계산")
print("="*80)

# Random Forest용 임시 데이터 준비 (변수명 충돌 방지)
X_rf = df3_pd[ALL_FEATURES].copy()

# 문자열 타입 컬럼을 숫자로 변환
for col in X_rf.columns:
    if X_rf[col].dtype == 'object':
        X_rf[col] = pd.to_numeric(X_rf[col], errors='coerce')

# 결측치를 평균으로 대체
X_rf = X_rf.fillna(X_rf.mean())
y_rf = df3_pd['target'].copy()

print(f"\n🔧 Random Forest 학습 중... (샘플: {len(X_rf):,}건)")
print(f"   ⏳ 잠시만 기다려주세요...")

# ============================================================================
# ⚙️ [조절 포인트 1-1] Random Forest 하이퍼파라미터
# ============================================================================
# 🔧 조절 방법:
#    - n_estimators: 트리 개수 (클수록 정확하지만 느림) → 기본값 100
#    - max_depth: 트리 깊이 (작을수록 간단함, 과적합 방지) → 기본값 10

# Random Forest 모델 학습
rf = RandomForestClassifier(
    n_estimators=100,      # 🔧 트리 개수: 50~500 범위에서 조절 가능
    max_depth=10,          # 🔧 트리 깊이: 5~20 범위에서 조절 가능
    random_state=42,
    class_weight='balanced',  # 불균형 데이터 처리
    n_jobs=-1              # 모든 CPU 코어 사용
)

rf.fit(X_rf, y_rf)

# Feature 중요도 정리
importances_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"✅ Random Forest 학습 완료!")

# STEP 3-4. Feature 중요도 시각화 및 선택
# ============================================================================
print("\n📊 Feature 중요도 Top 30:")
print("-" * 60)
print(f"{'순위':>4} | {'Feature':40s} | {'중요도':>10}")
print("-" * 60)
for idx, row in importances_df.head(30).iterrows():
    print(f"{importances_df.index.get_loc(idx)+1:>4} | {row['feature']:40s} | {row['importance']:>10.4f}")

# 누적 중요도 계산
importances_df['cumsum'] = importances_df['importance'].cumsum()

# 누적 90% 커버하는 Feature 개수
n_features_90 = (importances_df['cumsum'] <= 0.90).sum() + 1
print(f"\n💡 누적 중요도 90% 커버: {n_features_90}개 Feature")

# STEP 3-5. 여러 Feature 세트 준비 (비교용)
# ============================================================================
print("\n" + "="*80)
print("🎯 Feature 선택 옵션")
print("="*80)

feature_sets = {
    'top_20': importances_df.head(20)['feature'].tolist(),
    'top_30': importances_df.head(30)['feature'].tolist(),
    'top_40': importances_df.head(40)['feature'].tolist(),
    'cumsum_90': importances_df.head(n_features_90)['feature'].tolist(),
}

print("\n사용 가능한 Feature 세트:")
for name, features in feature_sets.items():
    print(f"  - {name:12s}: {len(features):2d}개")

# STEP 3-6. 최종 Feature 선택
# ============================================================================
print("\n" + "="*80)
print("✅ 최종 Feature 선택")
print("="*80)

# 🔽 여기를 수정해서 원하는 세트 선택!
SELECTED_FEATURES = feature_sets['top_30']  # 또는 'top_20', 'top_40', 'cumsum_90'

print(f"\n선택된 Feature 세트: top_30")
print(f"Feature 개수: {len(SELECTED_FEATURES)}개")
print("\n선택된 Feature 목록:")
for i, feat in enumerate(SELECTED_FEATURES, 1):
    imp = importances_df[importances_df['feature'] == feat]['importance'].values[0]
    print(f"  {i:2d}. {feat:40s} (중요도: {imp:.4f})")

# STEP 3-7. 데이터 준비 및 actual_irr 결측치 처리
# ============================================================================
print(f"\n📊 데이터 준비 중...")
print(f"  원본 데이터: {len(df3_pd):,}건")

# 메타 데이터
META_COLS = ['actual_irr', 'risk_free_rate', 'loan_amnt']
meta_available = [c for c in META_COLS if c in df3_pd.columns]

X = df3_pd[SELECTED_FEATURES].copy()
y = df3_pd['target'].copy()
weights = df3_pd['sample_weight'].copy() if 'sample_weight' in df3_pd.columns else np.ones(len(df3_pd))
meta = df3_pd[meta_available].copy()

# actual_irr 결측치를 risk_free_rate로 대체
if 'actual_irr' in meta.columns and 'risk_free_rate' in meta.columns:
    null_mask = ~np.isfinite(meta['actual_irr'])
    if null_mask.any():
        meta.loc[null_mask, 'actual_irr'] = meta.loc[null_mask, 'risk_free_rate']
        print(f"  ℹ️ actual_irr 결측 {null_mask.sum():,}건 → risk_free_rate로 대체")

print(f"✅ 데이터 준비 완료: X shape = {X.shape}, y shape = {y.shape}")

# STEP 3-8. 결측치 처리 (평균값으로 대체)
# ============================================================================
print(f"\n🔧 결측치 처리 중...")
null_counts_before = X.isnull().sum()
features_with_nulls = null_counts_before[null_counts_before > 0]

if len(features_with_nulls) > 0:
    print(f"⚠️ 결측치가 있는 Feature: {len(features_with_nulls)}개")
    for feat, count in features_with_nulls.items():
        mean_val = X[feat].mean()
        X[feat].fillna(mean_val, inplace=True)
    print(f"✅ 모든 결측치를 평균값으로 대체!")
else:
    print("✅ 결측치가 없습니다!")

print("\n" + "="*80)

# ============================================================================
# STEP 4: 데이터 분할
# ============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm

# STEP 4-1. 데이터 결측치 처리 및 무한대 값 정규화
# ============================================================================
print(f"\n🔧 Feature 전처리 중...")
X_processed = X.fillna(X.median())

# 무한대 값 처리 (있을 경우)
inf_count = 0
for col in tqdm(X_processed.columns, desc="무한대 값 검사", ncols=100):
    if X_processed[col].dtype in [np.float64, np.float32]:
        if np.isinf(X_processed[col]).any():
            finite_median = X_processed[col].replace([np.inf, -np.inf], np.nan).median()
            X_processed[col] = X_processed[col].replace([np.inf, -np.inf], finite_median)
            inf_count += 1

if inf_count > 0:
    print(f"  ℹ️ 총 {inf_count}개 컬럼에서 무한대 값을 중앙값으로 대체")

print(f"✅ Feature 전처리 완료: shape = {X_processed.shape}")

# STEP 4-2. 데이터 분할 (Train : Val : Test = 60% : 20% : 20%)
# ============================================================================
# 📌 목표: Test는 시계열 가장 뒤 20% 고정
#          Train/Val은 나머지 80%에서 랜덤 stratified 분할 (75% : 25% = 60% : 20%)

# ============================================================================
# ⚙️ [조절 포인트 2] 데이터 분할 비율
# ============================================================================
# 🔧 조절 방법:
#    - test_ratio=0.2     → Test 비율 변경 (시계열 뒤에서 고정 추출)
#    - val_of_train=0.25  → 나머지 중 Validation 비율 (0.25 = 전체의 20%)

print(f"\n📊 데이터 분할 중 (Test: 시계열 뒤 20% / Train·Val: 랜덤 Stratified)...")

# ── 1) issue_d 기준 시계열 정렬하여 Test 분리 ──────────────────────────────
issue_d_series = pd.to_datetime(df3_pd.loc[X_processed.index, 'issue_d'], format='%b-%Y', errors='coerce')
sorted_indices  = issue_d_series.sort_values().index   # 날짜 오름차순 인덱스

n_total    = len(X_processed)
test_ratio = 0.2
n_test     = int(n_total * test_ratio)
n_trainval = n_total - n_test

# 시계열 앞 80% → Train+Val 후보 / 뒤 20% → Test
trainval_indices = sorted_indices[:n_trainval]   # 앞 80%
test_indices_ts  = sorted_indices[n_trainval:]   # 뒤 20% (Test 고정)

X_trainval  = X_processed.loc[trainval_indices]
y_trainval  = y.loc[trainval_indices]
w_trainval  = (weights.loc[trainval_indices]
               if hasattr(weights, 'loc')
               else pd.Series(weights, index=X_processed.index).loc[trainval_indices])
meta_trainval = meta.loc[trainval_indices]

X_test    = X_processed.loc[test_indices_ts]
y_test    = y.loc[test_indices_ts]
w_test    = (weights.loc[test_indices_ts]
             if hasattr(weights, 'loc')
             else pd.Series(weights, index=X_processed.index).loc[test_indices_ts])
meta_test = meta.loc[test_indices_ts]

# ── 2) Train+Val 80% → Stratified 랜덤 분할 (Train 75% / Val 25%) ──────────
# 75% × 80% = 60% (전체), 25% × 80% = 20% (전체)
val_of_trainval = 0.25   # 🔧 조절 포인트

X_train, X_val, y_train, y_val, w_train, w_val, meta_train, meta_val = train_test_split(
    X_trainval, y_trainval, w_trainval, meta_trainval,
    test_size=val_of_trainval,
    random_state=42,
    stratify=y_trainval
)

# ── 3) 분할 결과 출력 ─────────────────────────────────────────────────────
train_dates     = issue_d_series.loc[X_train.index]
val_dates       = issue_d_series.loc[X_val.index]
test_dates_split = issue_d_series.loc[X_test.index]

print(f"\n📊 데이터 분할 완료:")
print(f"  Train: {len(X_train):,}건 ({len(X_train)/n_total*100:.1f}%) | {train_dates.min().strftime('%Y-%m')} ~ {train_dates.max().strftime('%Y-%m')} (랜덤)")
print(f"  Val:   {len(X_val):,}건 ({len(X_val)/n_total*100:.1f}%) | {val_dates.min().strftime('%Y-%m')} ~ {val_dates.max().strftime('%Y-%m')} (랜덤)")
print(f"  Test:  {len(X_test):,}건 ({len(X_test)/n_total*100:.1f}%) | {test_dates_split.min().strftime('%Y-%m')} ~ {test_dates_split.max().strftime('%Y-%m')} (시계열 고정)")
# STEP 5: 데이터 정규화 (StandardScaler)
# ============================================================================
# 📌 목표: StandardScaler로 특성 정규화 (평균=0, 표준편차=1)
# ============================================================================
print(f"\n🔧 데이터 정규화 중...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✅ 데이터 정규화 완료 (StandardScaler)")

# ============================================================================
# STEP 6: ANN 모델 구축 및 학습
# ============================================================================
# 📌 목표: 신경망 정의 → 컴파일 → 학습 (early stopping 포함)

# STEP 6-1. 모델 함수 정의
# ============================================================================
def build_ann_model(input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3):
    """ANN 모델 생성"""
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation='relu', name=f'hidden_{i+1}'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    return model

print(f"\n🏗️ ANN 모델 구축 중...")
input_dim = X_train_scaled.shape[1]

# STEP 6-2. 모델 구조 및 컴파일
# ============================================================================
# ⚙️ [조절 포인트 3] ANN 모델 구조 설정 (Hidden Layer)
# ============================================================================
# 🔧 조절 방법:
#    - hidden_layers: 은닉층의 뉴런 개수 리스트
#      예시: [128, 64, 32] → 3개 은닉층 (128→64→32 뉴런)
#      예시: [256, 128] → 2개 은닉층 (256→128 뉴런)
#      예시: [64, 64, 64] → 3개 동일 크기 은닉층
#    - dropout_rate: 과적합 방지 비율 (0.2~0.5 권장)
#      예시: 0.3 → 30% 뉴런 제거
#      예시: 0.5 → 50% 뉴런 제거

model = build_ann_model(
    input_dim=input_dim,
    hidden_layers=[128, 64, 32],    # 🔧 은닉층 구조: 뉴런 개수 리스트로 조절
    dropout_rate=0.3                 # 🔧 드롭아웃율: 0.1~0.5 범위에서 조절
)

# STEP 6-3. 컴파일 설정 (Loss Function, Optimizer)
# ============================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # 🔧 학습률: 0.0001~0.01 범위에서 조절
    loss='binary_crossentropy',      # 샘플 가중치 사용 (불균형 데이터 처리)
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("\n🏗️ ANN 모델 구조:")
model.summary()

# STEP 6-4. 모델 학습 (Early Stopping)
# ============================================================================
# ⚙️ [조절 포인트 4] ANN 모델 학습 설정
# ============================================================================
# 🔧 조절 방법:
#    - epochs: 반복 학습 횟수 (클수록 더 학습) → 기본값 50, 권장 범위: 30~100
#    - batch_size: 한 번에 처리할 샘플 수 (작을수록 정확하지만 느림) → 기본값 256
#    - patience (Early Stopping): 몇 epoch 동안 개선 없으면 중단 → 기본값 10
# 
# 💡 팁: Early Stopping이 작동하면 자동으로 epochs보다 먼저 중단됨

print(f"\n🚀 모델 학습 시작 (Early Stopping patience=10)...")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=1,                    # 🔧 개선 없는 epoch 수 (기본값 10, 범위: 5~20)
    restore_best_weights=True,
    verbose=1
)

# Keras 진행바는 자동으로 표시됨
history = model.fit(
    X_train_scaled, y_train,
    sample_weight=w_train.values,
    validation_data=(X_val_scaled, y_val, w_val.values),
    epochs=1,                      # 🔧 최대 반복 횟수 (기본값 50, 범위: 30~150)
    batch_size=256,                 # 🔧 배치 크기 (기본값 256, 범위: 32~512)
    callbacks=[early_stopping],
    verbose=1  # epoch별 진행상황 표시
)

print(f"✅ 모델 학습 완료! (총 {len(history.history['loss'])} epochs)")

# ============================================================================
# STEP 7: Top-K Portfolio 최적화 (Sharpe 극대화)
# ============================================================================
# 📌 목표: Validation 데이터로 K 탐색 → 최적 비율 결정 → 포트폴리오 구성

# STEP 7-1. 포트폴리오 성과 계산 함수
# ============================================================================
def calculate_sharpe_ratio_topk(y_pred_prob, meta_data, df_full, top_k):
    """
    Top-K개 포트폴리오의 Weighted Sharpe Ratio 계산
    (명백한 데이터 오류 제거)
    """
    # 부도 확률 낮은 순서대로 정렬
    sorted_indices = np.argsort(y_pred_prob)
    n_select = min(top_k, len(sorted_indices))
    
    if n_select < 10:
        return -999.0, 0.
    
    # 상위 K개 선택
    top_k_indices = sorted_indices[:n_select]
    selected_meta = meta_data.iloc[top_k_indices].copy()
    
    # 🔥 필터링: 명백한 데이터 오류 제거
    filter_mask = np.ones(len(selected_meta), dtype=bool)
    
    for i, idx in enumerate(selected_meta.index):
        if idx in df_full.index:
            row = df_full.loc[idx]
            
            # ❌ 필터 1: Recoveries > 100% 원금
            if 'recoveries' in row and 'funded_amnt' in row:
                if row['funded_amnt'] > 0:
                    rec_ratio = row['recoveries'] / row['funded_amnt']
                    if rec_ratio > 1.0:
                        filter_mask[i] = False
                        continue
            
            # ❌ 필터 2: Charged Off + total_pymnt > 110% 원금
            if row.get('loan_status') == 'Charged Off':
                if 'total_pymnt' in row and 'funded_amnt' in row:
                    if row['funded_amnt'] > 0:
                        recovery_rate = row['total_pymnt'] / row['funded_amnt']
                        if recovery_rate > 1.10:
                            filter_mask[i] = False
                            continue
    
    # 필터링 적용
    selected_meta = selected_meta[filter_mask]
    
    # 필수 컬럼 확인
    if 'actual_irr' not in selected_meta.columns or 'risk_free_rate' not in selected_meta.columns:
        return -999.0, 0.
    
    if 'loan_amnt' not in selected_meta.columns:
        return -999.0, 0.
    
    portfolio_returns = selected_meta['actual_irr'].values
    risk_free_rates = selected_meta['risk_free_rate'].values
    loan_amounts = selected_meta['loan_amnt'].values
    
    # 유한한 값만 필터링
    valid_mask = (
        np.isfinite(portfolio_returns) & 
        np.isfinite(risk_free_rates) &
        np.isfinite(loan_amounts) &
        (loan_amounts > 0)
    )
    
    if valid_mask.sum() < 10:
        return -999.0, 0.
    
    portfolio_returns = portfolio_returns[valid_mask]
    risk_free_rates = risk_free_rates[valid_mask]
    loan_amounts = loan_amounts[valid_mask]
    
    # 초과 수익률 계산
    excess_returns = portfolio_returns - risk_free_rates
    
    # 포트폴리오 가중치 계산 (금액 비중)
    #portfolio_weights = loan_amounts / loan_amounts.sum()
    
    # 가중 평균 초과 수익률
    mean_excess = np.mean(excess_returns)
    
    # 가중 평균 수익률
    mean_portfolio_return = np.mean(excess_returns + risk_free_rates)
    
    # 가중 분산 계산
    std_returns = np.std(excess_returns + risk_free_rates)
    
    # ⚠️ 극도로 낮은 변동성 체크
    MIN_STD = 1e-6  # 임계값을 더 크게 설정
    if std_returns < MIN_STD:
        # 변동성이 거의 없으면 Sharpe 제한 (최대 3.0 - 더 현실적)
        if mean_excess > 0:
            sharpe_ratio = 3.0  # 상한선 설정 (현실적 수준)
        else:
            sharpe_ratio = 0.0
        return sharpe_ratio, mean_portfolio_return
    
    # Weighted Sharpe Ratio (수정)
    sharpe_ratio = mean_excess / std_returns
    # 🔥 Sharpe ratio 상한선 (이상치 방지: -3 ~ 3)
    sharpe_ratio = np.clip(sharpe_ratio, -3.0, 3.0)
    
    return sharpe_ratio, mean_portfolio_return

# STEP 7-2. Validation 데이터 예측 및 최적 K 탐색
# ============================================================================
print(f"\n🔮 Validation 데이터 예측 중...")
y_val_pred_prob = model.predict(X_val_scaled, verbose=0).flatten()
print(f"✅ 예측 완료: {len(y_val_pred_prob):,}건")

# ============================================================================
# ⚙️ [조절 포인트 5] 포트폴리오 최적화 설정 (K 탐색 범위)
# ============================================================================
# 🔧 조절 방법:
#    - range(START, END): START%부터 END%까지 테스트
#    - 예시 1: range(1, 51) → 1%~50% 탐색
#    - 예시 2: range(5, 30) → 5%~30% 탐색 (범위를 좁히면 빠름)
#    - 간격: 현재는 1% 단위 (range의 끝값에 +1)

max_k = len(y_val_pred_prob)

# 비율로 K 후보 생성 (5%~50%, 1% 단위)
# 🔧 여기서 비율 범위 수정: range(START, END+1)
ratio_candidates = [i / 100 for i in range(5, 50)]  # [0.05, 0.06, ..., 0.49] 
k_candidates = [int(max_k * ratio) for ratio in ratio_candidates]
k_candidates = sorted(list(set(k_candidates)))  # 중복 제거 및 정렬

print(f"\n📈 Top-K Strategy: 최적 K 탐색 중 (Sharpe)...")
print(f"  데이터 크기: {len(y_val_pred_prob):,}건")
print(f"  K 후보 개수: {len(k_candidates)}개")
print(f"  비율 범위: {min(k_candidates)/len(y_val_pred_prob)*100:.1f}% ~ {max(k_candidates)/len(y_val_pred_prob)*100:.1f}%")
print(f"\n{'비율':>8} | {'K (개수)':>10} | {'Sharpe':>15}")
print("-" * 40)

sharpe_results = {}
return_results = {}
for k in tqdm(k_candidates, desc="최적 K 탐색", ncols=100):
    sharpe, port_return = calculate_sharpe_ratio_topk(y_val_pred_prob, meta_val, df3_pd, k)
    sharpe_results[k] = sharpe
    return_results[k] = port_return
    ratio = (k / len(y_val_pred_prob)) * 100
    
    # 매 5%마다 출력
    if abs(ratio - round(ratio / 5) * 5) < 0.1 or k == max(k_candidates):
        print(f"{ratio:>7.1f}% | {k:>10,} | {sharpe:>15.4f}")

# 최적 K 선택
best_k = max(sharpe_results, key=sharpe_results.get)
best_sharpe = sharpe_results[best_k]
best_port_return = return_results[best_k]

print("\n" + "="*80)
best_ratio = (best_k / len(y_val_pred_prob)) * 100
print(f"🏆 최적 비율: {best_ratio:.1f}% ({best_k:,}개) (Sharpe: {best_sharpe:.4f})")
print(f"   Portfolio Return: {best_port_return:.4f}")
print("="*80)

# STEP 7-3. 최고 성능 K 비율 Top 10 출력
# ============================================================================
print(f"\n📊 Top 10 최고 성능 비율:")
print(f"{'순위':>4} | {'비율':>8} | {'K (개수)':>10} | {'Sharpe':>15}")
print("-" * 50)

sorted_sharpe = sorted(sharpe_results.items(), key=lambda x: x[1], reverse=True)
for rank, (k, sharpe) in enumerate(sorted_sharpe[:10], 1):
    ratio = (k / len(y_val_pred_prob)) * 100
    marker = "🏆" if k == best_k else "  "
    print(f"{marker} {rank:>2} | {ratio:>7.1f}% | {k:>10,} | {sharpe:>15.4f}")

print("-" * 50)

# ============================================================================
# STEP 8: Test 데이터 평가 및 벤치마크 비교
# ============================================================================
# STEP 8-1. Test 데이터 예측
# ============================================================================
print(f"\n🔮 Test 데이터 예측 중...")
y_test_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
print(f"✅ 예측 완료: {len(y_test_pred_prob):,}건")

print(f"\n📊 Test 데이터 평가 중...")
test_sharpe, test_port_return = calculate_sharpe_ratio_topk(y_test_pred_prob, meta_test, df3_pd, best_k)

print(f"\n📊 Test 데이터 성능:")
print(f"  최적 K: {best_k:,}개")
print(f"  선택 비율: {(best_k/len(y_test_pred_prob)*100):.1f}%")
print(f"  Sharpe: {test_sharpe:.4f}")
print(f"  Portfolio Return: {test_port_return:.4f}")

# ============================================================================
# STEP 8-3. 벤치마크 전략 성과 계산 (위치 수정)
# ============================================================================
def get_benchmark_performance(meta_data, df_full):
    """
    벤치마크 전략: 모든 대출을 승인했을 때의 성과
    (명백한 데이터 오류 제거)
    """
    # 🔥 필터링: 명백한 데이터 오류 제거
    filter_mask = np.ones(len(meta_data), dtype=bool)
    
    for i, idx in enumerate(meta_data.index):
        if idx in df_full.index:
            row = df_full.loc[idx]
            
            # ❌ 필터 1: Recoveries > 100% 원금
            if 'recoveries' in row and 'funded_amnt' in row:
                if row['funded_amnt'] > 0:
                    rec_ratio = row['recoveries'] / row['funded_amnt']
                    if rec_ratio > 1.0:
                        filter_mask[i] = False
                        continue
            
            # ❌ 필터 2: Charged Off + total_pymnt > 110% 원금
            if row.get('loan_status') == 'Charged Off':
                if 'total_pymnt' in row and 'funded_amnt' in row:
                    if row['funded_amnt'] > 0:
                        recovery_rate = row['total_pymnt'] / row['funded_amnt']
                        if recovery_rate > 1.10:
                            filter_mask[i] = False
                            continue
    
    # 필터링 적용
    meta_data_clean = meta_data[filter_mask]
    
    # 필수 컬럼 확인
    if 'actual_irr' not in meta_data_clean.columns or 'risk_free_rate' not in meta_data_clean.columns:
        return 0.0, 0.0
    
    if 'loan_amnt' not in meta_data_clean.columns:
        return 0.0, 0.0
    
    actual_irrs = meta_data_clean['actual_irr'].values
    risk_free_rates = meta_data_clean['risk_free_rate'].values
    loan_amounts = meta_data_clean['loan_amnt'].values
    
    # 유한한 값만 사용
    valid_mask = (
        np.isfinite(actual_irrs) & 
        np.isfinite(risk_free_rates) &
        np.isfinite(loan_amounts) &
        (loan_amounts > 0)
    )
    
    actual_irrs = actual_irrs[valid_mask]
    risk_free_rates = risk_free_rates[valid_mask]
    loan_amounts = loan_amounts[valid_mask]
    
    if len(actual_irrs) < 10:
        return 0.0, 0.0
    
    # 초과 수익률 계산
    excess_returns = actual_irrs - risk_free_rates
    
    # 포트폴리오 가중치
    #portfolio_weights = loan_amounts / loan_amounts.sum()
    
    # 가중 평균 초과 수익률 (수정)
    mean_excess = np.mean(excess_returns)
    
    # 가중 평균 수익률 (수정)
    mean_portfolio_return = np.mean(excess_returns + risk_free_rates)
    
    # 가중 분산
    std_returns = np.std(excess_returns + risk_free_rates)
    
    if std_returns < 1e-9:
        return 0.0, 0.0
    
    sharpe_ratio = mean_excess / std_returns  # ← 이건 새로 추가 (수정)
    
    return sharpe_ratio, mean_portfolio_return  # ← 여기만 수정 (원래 return weighted_excess / weighted_std) (수정)

# STEP 8-2. 시기별 Test 평가 (Temporal Robustness)
# ============================================================================
print("\n" + "="*80)
print("📅 시기별 Test 평가 (Temporal Robustness)")
print("="*80)

# Test 데이터의 issue_d 파싱
test_indices = X_test.index
test_dates = pd.to_datetime(df3_pd.loc[test_indices, 'issue_d'], format='%b-%Y', errors='coerce')

# 시기별 마스크 생성
crisis_mask = (test_dates >= '2007-01-01') & (test_dates <= '2009-12-31')
stable_mask = (test_dates >= '2010-01-01') & (test_dates <= '2019-12-31')
covid_mask = (test_dates >= '2020-01-01') & (test_dates <= '2020-12-31')

# 시기별 결과 저장
temporal_results = {}

print(f"\n📊 Test 데이터 시기별 분포:")
print(f"  금융위기 (2007-2009): {crisis_mask.sum():,}건 ({crisis_mask.sum()/len(test_dates)*100:.1f}%)")
print(f"  안정기 (2010-2019):   {stable_mask.sum():,}건 ({stable_mask.sum()/len(test_dates)*100:.1f}%)")
print(f"  코로나 (2020):        {covid_mask.sum():,}건 ({covid_mask.sum()/len(test_dates)*100:.1f}%)")

print(f"\n{'시기':15} | {'샘플 수':>10} | {'Top-K Sharpe':>15} | {'Benchmark':>15} | {'개선도':>10}")
print("-" * 80)

for period_name, mask in [('금융위기 (07-09)', crisis_mask), 
                           ('안정기 (10-19)', stable_mask), 
                           ('코로나 (2020)', covid_mask)]:
    if mask.sum() < 10:  # 샘플이 너무 적으면 스킵
        print(f"{period_name:15} | {'N/A':>10} | {'N/A':>15} | {'N/A':>15} | {'N/A':>10}")
        temporal_results[period_name] = {
            'n_samples': 0,
            'top_k_sharpe': None,
            'benchmark_sharpe': None,
            'improvement_pct': None
        }
        continue
    
    # 해당 시기 데이터 추출
    period_pred = y_test_pred_prob[mask.values]
    period_meta = meta_test[mask.values].reset_index(drop=True)
    
    # 샘플 수에 맞게 K 조정
    period_k = min(best_k, len(period_pred))
    
    # Top-K Sharpe 계산
    period_sharpe, period_port_return = calculate_sharpe_ratio_topk(period_pred, period_meta, df3_pd, period_k)
    
    # Benchmark Sharpe 계산
    period_benchmark, benchmark_return = get_benchmark_performance(period_meta, df3_pd)
    
    # 개선도 계산
    if abs(period_benchmark) > 1e-9:
        improvement = (period_sharpe - period_benchmark) / abs(period_benchmark) * 100
    else:
        improvement = 0.0
    
    # 결과 출력
    print(f"{period_name:15} | {mask.sum():>10,} | {period_sharpe:>15.4f} | {period_benchmark:>15.4f} | {improvement:>9.1f}%")
    
    # 결과 저장
    temporal_results[period_name] = {
        'n_samples': int(mask.sum()),
        'top_k_sharpe': float(period_sharpe),
        'benchmark_sharpe': float(period_benchmark),
        'improvement_pct': float(improvement)
    }

print("-" * 80)
print(f"✅ 시기별 평가 완료!")



print(f"\n📊 벤치마크 계산 중...")
benchmark_val_sharpe, benchmark_val_return = get_benchmark_performance(meta_val, df3_pd)
benchmark_test_sharpe, benchmark_test_return = get_benchmark_performance(meta_test, df3_pd)

print(f"\n📊 벤치마크 (모든 대출 승인) 성과:")
print(f"  Validation Sharpe: {benchmark_val_sharpe:.4f}")
print(f"  Validation Return: {benchmark_val_return:.4f}")
print(f"  Test Sharpe:       {benchmark_test_sharpe:.4f}")
print(f"  Test Return:       {benchmark_test_return:.4f}")

# ============================================================================
# ============================================================================
# STEP 9: 최종 결과 요약
# ============================================================================
print("\n" + "="*80)
print("✅ ANN 모델 학습 및 평가 완료!")
print("="*80)
print(f"📌 원본 데이터: {len(X_processed):,}건")
print(f"📌 선택된 Feature 개수: {len(SELECTED_FEATURES)}")
print(f"📌 모델 구조: {[128, 64, 32]} (hidden layers)")
print(f"📌 Optimizer: Adam (lr=0.001)")
print(f"📌 평가 방식: Sharpe")
print(f"")
val_port_return = return_results[best_k]
print(f"📈 Validation 성과:")
print(f"   최적 K: {best_k:,}개 ({(best_k/len(y_val_pred_prob)*100):.1f}%)")
print(f"   Top-K Sharpe:    {best_sharpe:.4f}")
print(f"   Top-K Return:    {val_port_return:.4f}")
print(f"   Benchmark Sharpe: {benchmark_val_sharpe:.4f}")
if abs(benchmark_val_sharpe) > 0:
    improvement = ((best_sharpe - benchmark_val_sharpe) / abs(benchmark_val_sharpe) * 100)
    print(f"   개선도: {improvement:+.1f}%")
print(f"")
print(f"📈 Test 성과:")
print(f"   Top-K Sharpe:    {test_sharpe:.4f}")
print(f"   Top-K Return:    {test_port_return:.4f}")
print(f"   Benchmark Sharpe: {benchmark_test_sharpe:.4f}")
if abs(benchmark_test_sharpe) > 0:
    improvement = ((test_sharpe - benchmark_test_sharpe) / abs(benchmark_test_sharpe) * 100)
    print(f"   개선도: {improvement:+.1f}%")
print("="*80)

# ============================================================================
# STEP 10: Bootstrap 테스트 및 최종 저장
# ============================================================================
from datetime import datetime
import json

print("\n" + "="*80)
print("🔄 Bootstrap 테스트 시작")
print("="*80)

# STEP 10-1. Bootstrap 샘플링 설정
# ============================================================================
# ⚙️ [조절 포인트 6] Bootstrap 샘플링 설정
# ============================================================================
# 🔧 조절 방법:
#    - N_BOOTSTRAP: 반복 횟수 (클수록 더 정확하지만 느림)
#    - 기본값: 1000 (권장 범위: 500~5000)
#    - 예시: 빠른 테스트는 100~500, 정확한 결과는 2000~5000

# Bootstrap 설정
N_BOOTSTRAP = 1000                  # 🔧 반복 횟수 조절 (기본값 1000, 범위: 100~5000)
np.random.seed(42)

# 결과 저장용 리스트
bootstrap_sharpes = []
bootstrap_irrs = []

print(f"\n⏳ Test 데이터로 {N_BOOTSTRAP}회 Bootstrap 샘플링 중...\n")

for i in tqdm(range(N_BOOTSTRAP), desc="Bootstrap 진행", ncols=100):
    # Bootstrap 샘플링 (복원 추출)
    n_test = len(y_test_pred_prob)
    bootstrap_indices = np.random.choice(n_test, size=n_test, replace=True)
    
    # 샘플링된 데이터
    boot_pred_prob = y_test_pred_prob[bootstrap_indices]
    boot_meta = meta_test.iloc[bootstrap_indices].reset_index(drop=True)
    
    # Sharpe Ratio 계산 (최적 K 사용)
    boot_sharpe, boot_port_return = calculate_sharpe_ratio_topk(boot_pred_prob, boot_meta, df3_pd, best_k)
    bootstrap_sharpes.append(boot_sharpe)
    bootstrap_irrs.append(boot_port_return)  # ← 추가

bootstrap_sharpes = np.array(bootstrap_sharpes)

# 통계량 계산
mean_sharpe = np.mean(bootstrap_sharpes)
std_sharpe = np.std(bootstrap_sharpes)
ci_lower = np.percentile(bootstrap_sharpes, 2.5)
ci_upper = np.percentile(bootstrap_sharpes, 97.5)

print(f"\n✅ Bootstrap 완료!\n")
print(f"📊 Bootstrap Sharpe 분포 (1000회):")
print(f"  평균:        {mean_sharpe:.4f}")
print(f"  표준편차:    {std_sharpe:.4f}")
print(f"  95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  Min:         {np.min(bootstrap_sharpes):.4f}")
print(f"  Max:         {np.max(bootstrap_sharpes):.4f}")

# 추가
bootstrap_irrs = np.array(bootstrap_irrs)
mean_irr = np.mean(bootstrap_irrs)
std_irr = np.std(bootstrap_irrs)
ci_irr_lower = np.percentile(bootstrap_irrs, 2.5)
ci_irr_upper = np.percentile(bootstrap_irrs, 97.5)

print(f"\n📊 Bootstrap 평균 IRR 분포 (1000회):")
print(f"  평균:         {mean_irr:.4f}")
print(f"  표준편차:     {std_irr:.4f}")
print(f"  95% 신뢰구간: [{ci_irr_lower:.4f}, {ci_irr_upper:.4f}]")
print(f"  Min:          {np.min(bootstrap_irrs):.4f}")
print(f"  Max:          {np.max(bootstrap_irrs):.4f}")

# STEP 10-2. 실험 결과 저장 (CSV, JSON, TXT, PNG)
# ============================================================================
print("\n" + "="*80)
print("💾 실험 결과 저장 중...")
print("="*80)

# 타임스탬프 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "experiment_results"

# 디렉토리 생성
import os
os.makedirs(output_dir, exist_ok=True)

# STEP 10-2-1. Feature 중요도 저장 (CSV)
# ============================================================================
feature_importance_file = f"{output_dir}/feature_importance_{timestamp}.csv"
importances_df.to_csv(feature_importance_file, index=False, encoding='utf-8-sig')
print(f"✅ Feature 중요도 저장: {feature_importance_file}")

# STEP 10-2-2. 실험 설정 및 결과 저장 (JSON)
# ============================================================================
experiment_config = {
    "timestamp": timestamp,
    "data_info": {
        "total_samples": int(len(X_processed)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "n_features": len(SELECTED_FEATURES),
        "feature_selection_method": "Random Forest Importance (top_30)"
    },
    "model_config": {
        "model_type": "ANN (Artificial Neural Network)",
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 256,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "loss_function": "binary_crossentropy",
        "actual_epochs_trained": len(history.history['loss'])
    },
    "evaluation_config": {
        "metric": "Sharpe Ratio",
        "top_k_strategy": True,
        "optimal_k": int(best_k),
        "optimal_k_selection_ratio": float(best_k / len(y_val_pred_prob) * 100)
    },
    "validation_results": {
        "optimal_k": int(best_k),
        "top_k_sharpe": float(best_sharpe),
        "benchmark_sharpe": float(benchmark_val_sharpe),
        "improvement_pct": float((best_sharpe - benchmark_val_sharpe) / abs(benchmark_val_sharpe) * 100) if abs(benchmark_val_sharpe) > 0 else None
    },
    "test_results": {
        "single_test_sharpe": float(test_sharpe),
        "benchmark_sharpe": float(benchmark_test_sharpe),
        "improvement_pct": float((test_sharpe - benchmark_test_sharpe) / abs(benchmark_test_sharpe) * 100) if abs(benchmark_test_sharpe) > 0 else None,
        "temporal_robustness": temporal_results  # 🔥 이 줄 추가
    },
    "bootstrap_results": {
        "n_iterations": N_BOOTSTRAP,
        "mean_sharpe": float(mean_sharpe),
        "std_sharpe": float(std_sharpe),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "min_sharpe": float(np.min(bootstrap_sharpes)),
        "max_sharpe": float(np.max(bootstrap_sharpes))
    },
    "selected_features": SELECTED_FEATURES
}

config_file = f"{output_dir}/experiment_config_{timestamp}.json"
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(experiment_config, f, indent=2, ensure_ascii=False)
print(f"✅ 실험 설정 저장: {config_file}")

# STEP 10-2-3. Bootstrap 분포 저장 (CSV)
# ============================================================================
bootstrap_df = pd.DataFrame({
    'iteration': range(1, N_BOOTSTRAP + 1),
    'sharpe_ratio': bootstrap_sharpes
})
bootstrap_file = f"{output_dir}/bootstrap_distribution_{timestamp}.csv"
bootstrap_df.to_csv(bootstrap_file, index=False, encoding='utf-8-sig')
print(f"✅ Bootstrap 분포 저장: {bootstrap_file}")

# STEP 10-2-4. 종합 보고서 저장 (TXT)
# ============================================================================
report_file = f"{output_dir}/experiment_report_{timestamp}.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ANN 대출 승인 모델 실험 보고서\n")
    f.write("="*80 + "\n")
    f.write(f"실험 일시: {timestamp}\n\n")
    
    f.write("1. 데이터 정보\n")
    f.write("-" * 80 + "\n")
    f.write(f"  전체 샘플 수:     {len(X_processed):,}건\n")
    f.write(f"  Train 샘플:       {len(X_train):,}건 ({len(X_train)/len(X_processed)*100:.1f}%)\n")
    f.write(f"  Validation 샘플:  {len(X_val):,}건 ({len(X_val)/len(X_processed)*100:.1f}%)\n")
    f.write(f"  Test 샘플:        {len(X_test):,}건 ({len(X_test)/len(X_processed)*100:.1f}%)\n")
    f.write(f"  Feature 개수:     {len(SELECTED_FEATURES)}\n")
    f.write(f"  Feature 선택:     Random Forest Importance (top_30)\n\n")
    
    f.write("2. 모델 하이퍼파라미터\n")
    f.write("-" * 80 + "\n")
    f.write(f"  모델 타입:        ANN (Artificial Neural Network)\n")
    f.write(f"  Hidden Layers:    {[128, 64, 32]}\n")
    f.write(f"  Dropout Rate:     0.3\n")
    f.write(f"  Optimizer:        Adam (lr=0.001)\n")
    f.write(f"  Batch Size:       256\n")
    f.write(f"  Max Epochs:       100\n")
    f.write(f"  Early Stopping:   patience=10\n")
    f.write(f"  실제 학습 Epochs: {len(history.history['loss'])}\n")
    f.write(f"  Loss Function:    binary_crossentropy\n\n")
    
    f.write("3. Validation 결과\n")
    f.write("-" * 80 + "\n")
    f.write(f"  최적 K:           {best_k:,}개 ({best_k/len(y_val_pred_prob)*100:.1f}%)\n")
    f.write(f"  Top-K Sharpe:     {best_sharpe:.4f}\n")
    f.write(f"  Portfolio Return: {test_port_return:.4f}\n")  # ← 이 줄 추가
    f.write(f"  Benchmark Sharpe: {benchmark_val_sharpe:.4f}\n")
    if abs(benchmark_val_sharpe) > 0:
        improvement = (best_sharpe - benchmark_val_sharpe) / abs(benchmark_val_sharpe) * 100
        f.write(f"  개선도:           {improvement:+.1f}%\n")
    f.write("\n")
    
    f.write("4. Test 결과 (단일 실행)\n")
    f.write("-" * 80 + "\n")
    f.write(f"  Top-K Sharpe:     {test_sharpe:.4f}\n")
    f.write(f"  Benchmark Sharpe: {benchmark_test_sharpe:.4f}\n")
    if abs(benchmark_test_sharpe) > 0:
        improvement = (test_sharpe - benchmark_test_sharpe) / abs(benchmark_test_sharpe) * 100
        f.write(f"  개선도:           {improvement:+.1f}%\n")
    f.write("\n")

    f.write("4-1. Test 시기별 분석 (Temporal Robustness)\n")
    f.write("-" * 80 + "\n")
    for period_name, results in temporal_results.items():
        f.write(f"\n  [{period_name}]\n")
        if results['n_samples'] == 0:
            f.write(f"    샘플 수:          N/A (데이터 부족)\n")
        else:
            f.write(f"    샘플 수:          {results['n_samples']:,}건\n")
            f.write(f"    Top-K Sharpe:     {results['top_k_sharpe']:.4f}\n")
            f.write(f"    Benchmark Sharpe: {results['benchmark_sharpe']:.4f}\n")
            f.write(f"    개선도:           {results['improvement_pct']:+.1f}%\n")
    f.write("\n")

    f.write("5. Bootstrap 결과 (1000회 반복)\n")
    f.write("-" * 80 + "\n")
    f.write(f"  평균 Sharpe:      {mean_sharpe:.4f}\n")
    f.write(f"  표준편차:         {std_sharpe:.4f}\n")
    f.write(f"  95% 신뢰구간:     [{ci_lower:.4f}, {ci_upper:.4f}]\n")
    f.write(f"  Min Sharpe:       {np.min(bootstrap_sharpes):.4f}\n")
    # 변경
    f.write(f"  Max Sharpe:       {np.max(bootstrap_sharpes):.4f}\n\n")

    f.write("5-1. Bootstrap 평균 IRR 분포 (1000회 반복)\n")  # ← 추가
    f.write("-" * 80 + "\n")
    f.write(f"  평균 IRR:         {mean_irr:.4f}\n")
    f.write(f"  표준편차:         {std_irr:.4f}\n")
    f.write(f"  95% 신뢰구간:     [{ci_irr_lower:.4f}, {ci_irr_upper:.4f}]\n")
    f.write(f"  Min IRR:          {np.min(bootstrap_irrs):.4f}\n")
    f.write(f"  Max IRR:          {np.max(bootstrap_irrs):.4f}\n\n")
    
    f.write("6. 선택된 Features (Top 30)\n")
    f.write("-" * 80 + "\n")
    for i, feat in enumerate(SELECTED_FEATURES, 1):
        imp = importances_df[importances_df['feature'] == feat]['importance'].values[0]
        f.write(f"  {i:2d}. {feat:40s} (중요도: {imp:.4f})\n")
    
    f.write("\n" + "="*80 + "\n")

print(f"✅ 종합 보고서 저장: {report_file}")

# STEP 10-2-5. 히스토그램 시각화 (PNG)
# ============================================================================
plt.figure(figsize=(12, 6))

# Bootstrap 분포 히스토그램
plt.hist(bootstrap_sharpes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(mean_sharpe, color='red', linestyle='--', linewidth=2, label=f'평균: {mean_sharpe:.4f}')
plt.axvline(ci_lower, color='orange', linestyle='--', linewidth=1.5, label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
plt.axvline(ci_upper, color='orange', linestyle='--', linewidth=1.5)
plt.axvline(test_sharpe, color='green', linestyle='-', linewidth=2, label=f'단일 Test: {test_sharpe:.4f}')

plt.xlabel('Sharpe Ratio', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.title(f'Bootstrap Sharpe 분포 (N={N_BOOTSTRAP})', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

histogram_file = f"{output_dir}/bootstrap_histogram_{timestamp}.png"
plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
print(f"✅ 히스토그램 저장: {histogram_file}")
plt.close()

print("\n" + "="*80)
print(f"✅ 모든 결과가 '{output_dir}/' 디렉토리에 저장되었습니다!")
print("="*80)
print(f"\n📁 저장된 파일 목록:")
print(f"  1. {feature_importance_file}")
print(f"  2. {config_file}")
print(f"  3. {bootstrap_file}")
print(f"  4. {report_file}")
print(f"  5. {histogram_file}")
print("="*80)
