
# src/config.py
"""
프로젝트 전역 설정 및 상수 관리
"""

import os

# ==============================================================================
# 경로 설정
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 경로
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'external')

# 원본 데이터 파일
TRAIN_DATA_FILE = os.path.join(BASE_DIR, 'lending_club_2020_train.csv')
PROCESSED_TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_final.parquet')
FEATURE_CANDIDATES_FILE = os.path.join(BASE_DIR, '260129_feature 변수후보.xlsx')

# 국채 데이터 (FRED)
GS3_DATA_FILE = os.path.join(EXTERNAL_DATA_DIR, 'GS3.csv')
GS5_DATA_FILE = os.path.join(EXTERNAL_DATA_DIR, 'GS5.csv')

# 모델 저장 경로
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 로그 및 결과 저장
EXPERIMENTS_LOG_FILE = os.path.join(BASE_DIR, 'experiments_log.csv')

# 디렉토리 자동 생성
for dir_path in [PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==============================================================================
# 모델 설정
# ==============================================================================

# 사용할 모델 목록 (ANN만 사용)
MODELS_TO_USE = ['ANN']

# 하이퍼파라미터 설정
MODEL_PARAMS = {
    'ANN': {
        'hidden_layer_sizes': (256, 128, 64),  # MLPClassifier 은닉층 노드 수 (튜플)
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'alpha': 0.001,  # L2 정규화 (Dropout 대체)
        'max_iter': 1000,
        'early_stopping': True,
        'n_iter_no_change': 10,
        'random_state': 42,
        'batch_size': 512,
    }
}

# ==============================================================================
# 전처리 설정
# ==============================================================================

# 부도(Default) 정의
TARGET_BAD_STATUSES = [
    'Charged Off',
    'Default',
    'Late (31-120 days)',
]

TARGET_GOOD_STATUSES = [
    'Fully Paid',
]

# ⭐ 테스트용: 대부분의 변수 제외 (10개 변수만 사용)
EXCLUDE_COLS = [
    'target', 'loan_status', 
    'issue_d', 'issue_d_parsed', 
    'total_pymnt', 'funded_amnt', 'term', 'int_rate', 
    'loan_return', 'portfolio_ret', 'invest', 'loan_ret', 'rf_ret',
    'earliest_cr_line_parsed',
    'last_fico_range_high', 'last_fico_range_low', 
    'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
    'last_pymnt_amnt', 'last_pymnt_d', 'next_pymnt_d', 
    'out_prncp', 'out_prncp_inv',
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
    'hardship_amount', 'hardship_start_date', 'hardship_end_date', 
    'payment_plan_start_date', 'hardship_length', 'hardship_dpd', 
    'hardship_loan_status', 'hardship_payoff_balance_amount', 
    'hardship_last_payment_amount', 'deferral_term',
    'orig_projected_additional_accrued_interest',

    # 제외할 변수(leakage 방지)
    # ⭐ 테스트용 추가: 거의 모든 신용도 변수 제외
    'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 
    'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 
    'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq', 
    'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 
    'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 
    'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 
    'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec', 
    'pub_rec_bankruptcies', 'tax_liens', 'tot_cur_bal', 'tot_hi_cred_lim', 
    'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 
    'total_rev_hi_lim',
    'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 
    'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 
    'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
    'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line', 
    'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 
    'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts', 
    'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 
    'sec_app_mths_since_last_major_derog',
    'open_act_il_log', 'open_il_12m_log', 'total_bal_il_log',
    'sec_app_revol_util_log', 'all_util_log',
]

# 금융 관련 변수 (수익률 계산용, 학습에는 사용 안함)
FINANCE_COLS = ['total_pymnt', 'funded_amnt', 'term', 'int_rate', 'issue_d']

# 제거할 변수
DROP_COLS = [
    'id', 'member_id', 'url', 'desc', 'policy_code', 'zip_code', 'addr_state',
    'emp_title', 'title', 
    'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d',
    'collection_recovery_fee', 'recoveries', 'total_rec_prncp', 'total_rec_int', 
    'total_rec_late_fee', 'pymnt_plan', 'out_prncp', 'out_prncp_inv',
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
    'debt_settlement_flag', 'settlement_status', 'settlement_date'
]

# 범주형 변수
CATEGORICAL_FEATURES = [
    'home_ownership', 'purpose', 'application_type', 
    'initial_list_status', 'grade', 'sub_grade', 'verification_status'
]

# ==============================================================================
# 학습/평가 설정
# ==============================================================================

# 데이터 샘플링 (테스트용)
USE_SAMPLE = False
SAMPLE_FRAC = 0.3

# Train-Validation-Test Split 비율
TRAIN_RATIO = 0.6  # 60% for training
VAL_RATIO = 0.2    # 20% for validation
TEST_RATIO = 0.2   # 20% for testing

# 투자 임계값 (부도 확률이 이 이상이면 투자하지 않음)
INVESTMENT_THRESHOLD = 0.15

# 국채 수익률 (fallback)
DEFAULT_RF_3YR = 0.061  # 3년 만기
DEFAULT_RF_5YR = 0.104  # 5년 만기

# ==============================================================================
# 로깅 설정
# ==============================================================================

LOG_FORMAT = 'csv'  # csv 또는 json
LOG_COLUMNS = [
    'Date', 'Model', 'Split', 'AUC', 'Sharpe', 
    'Avg_Return', 'Duration', 'Params', 'Memo'
]
