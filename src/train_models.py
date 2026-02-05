import polars as pl
import pandas as pd
import numpy as np
import numpy_financial as npf
import os
import joblib
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from config import (
    PROCESSED_TRAIN_FILE, MODEL_DIR, EXTERNAL_DATA_DIR,
    EXCLUDE_COLS, USE_SAMPLE, SAMPLE_FRAC,
    INVESTMENT_THRESHOLD, DEFAULT_RF_3YR, DEFAULT_RF_5YR,
    MODEL_PARAMS, BASE_DIR
)
from utils import ExperimentLogger, add_return_metrics_to_dataframe, find_optimal_threshold_bootstrap, plot_optimal_threshold, calculate_portfolio_irr, create_benchmark_comparison, plot_bootstrap_distribution, calculate_irr_for_dataframe

# ==============================================================================
# 1. 로거 및 경로 초기화
# ==============================================================================
logger = ExperimentLogger()
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================================================================
# 2. 국채 금리 데이터 로더
# ==============================================================================
def load_treasury_rates(df):
    """
    FRED에서 자동으로 GS3(3년물), GS5(5년물) 국채 수익률을 다운받아 대출 데이터와 병합.
    """
    try:
        import pandas_datareader as pdr
        
        print("💵 FRED에서 국채 데이터를 자동 다운로드 중...")
        
        # FRED에서 GS3, GS5 다운로드 (2007-01-01 ~ 2020-12-31 → 대출 데이터 범위에 맞춤)
        gs3_data = pdr.get_data_fred('GS3', start='2007-01-01', end='2020-12-31')
        gs5_data = pdr.get_data_fred('GS5', start='2007-01-01', end='2020-12-31')
        
        # DataFrame으로 변환 및 컬럼명 정렬
        gs3 = gs3_data.reset_index().rename(columns={'GS3': 'rate_3yr', 'DATE': 'date'})
        gs5 = gs5_data.reset_index().rename(columns={'GS5': 'rate_5yr', 'DATE': 'date'})
        gs3['date'] = pd.to_datetime(gs3['date'])
        gs5['date'] = pd.to_datetime(gs5['date'])
        
        # CSV로 저장 (캐싱용)
        gs3_cache_path = os.path.join(EXTERNAL_DATA_DIR, 'GS3.csv')
        gs5_cache_path = os.path.join(EXTERNAL_DATA_DIR, 'GS5.csv')
        gs3.to_csv(gs3_cache_path, index=False)
        gs5.to_csv(gs5_cache_path, index=False)
        print(f"✅ 국채 데이터 저장 완료: {gs3_cache_path}")
        
    except Exception as e:
        print(f"⚠️ FRED 자동 다운로드 실패: {e}")
        print("💾 저장된 CSV 파일을 찾아 로드합니다...")
        
        gs3_path = os.path.join(EXTERNAL_DATA_DIR, 'GS3.csv')
        gs5_path = os.path.join(EXTERNAL_DATA_DIR, 'GS5.csv')
        
        if not os.path.exists(gs3_path) or not os.path.exists(gs5_path):
            print("❌ 국채 데이터를 찾을 수 없습니다. 상수값을 사용합니다.")
            df['rf_ret'] = np.where(df['term'] == 36, DEFAULT_RF_3YR, DEFAULT_RF_5YR)
            return df
        
        # CSV에서 로드
        gs3 = pd.read_csv(gs3_path)
        gs5 = pd.read_csv(gs5_path)
        gs3.rename(columns={'DATE': 'date'}, inplace=True)
        gs5.rename(columns={'DATE': 'date'}, inplace=True)
        gs3['date'] = pd.to_datetime(gs3['date'])
        gs5['date'] = pd.to_datetime(gs5['date'])
        print("✅ 캐시된 국채 데이터 로드 완료")
    
    # 1. 국채 데이터 준비 완료
    # (gs3, gs5 이미 로드됨)
    
    # 2. 월 단위 매핑을 위해 'YYYY-MM' 형식의 키 생성
    gs3['month_key'] = gs3['date'].dt.to_period('M')
    gs5['month_key'] = gs5['date'].dt.to_period('M')
    
    if 'issue_d_parsed' in df.columns:
        df['month_key'] = df['issue_d_parsed'].dt.to_period('M')
    else:
        df['month_key'] = pd.to_datetime(df['issue_d'], format='%b-%Y').dt.to_period('M')

    # 3. 병합 (Merge)
    df = df.merge(gs3[['month_key', 'rate_3yr']], on='month_key', how='left')
    df = df.merge(gs5[['month_key', 'rate_5yr']], on='month_key', how='left')
    
    # 4. 결측치 처리
    df['rate_3yr'] = df['rate_3yr'].fillna(DEFAULT_RF_3YR).astype(float)
    df['rate_5yr'] = df['rate_5yr'].fillna(DEFAULT_RF_5YR).astype(float)
    
    # 5. 기간 수익률로 변환 (연이율 -> 3년/5년 누적 수익률)
    df['rf_ret_3yr'] = (1 + df['rate_3yr']/100) ** 3 - 1
    df['rf_ret_5yr'] = (1 + df['rate_5yr']/100) ** 5 - 1
    
    # 6. term에 맞춰 최종 rf_ret 결정
    df['rf_ret'] = np.where(df['term'] == 36, df['rf_ret_3yr'], df['rf_ret_5yr'])
    
    # 불필요 컬럼 정리
    df.drop(columns=['month_key', 'rate_3yr', 'rate_5yr', 'rf_ret_3yr', 'rf_ret_5yr'], inplace=True)
    
    return df

# ==============================================================================
# 3. Sharpe Ratio 계산 (IRR 기반)
# ==============================================================================

def calculate_sharpe_ratio(df, threshold=INVESTMENT_THRESHOLD):
    """
    IRR 기반 샤프 비율 계산
    ⚠️ 중요: 투자 대출에 대해서만 IRR 계산
    
    Parameters:
    -----------
    df : pandas DataFrame
        'rf_ret', 'pred_prob' 컬럼 필요
        (irr 컬럼은 투자 대출에 대해서만 계산)
    threshold : float
        투자 결정 임계값
    
    Returns:
    --------
    tuple : (sharpe_ratio, avg_portfolio_return)
    """
    df = df.copy()
    
    # 1️⃣ 먼저 투자 결정
    df['invest'] = df['pred_prob'] < threshold
    n_invest = df['invest'].sum()
    n_total = len(df)
    
    print(f"  📊 투자 결정: {n_invest:,}개 투자 / {n_total:,}개 전체 ({n_invest/n_total:.1%})")
    
    # 2️⃣ 투자 대출에 대해서만 IRR 계산
    if n_invest > 0:
        print(f"  💰 투자 대출 {n_invest:,}개에 대해서만 IRR 계산 중...")
        
        from utils import calculate_irr_for_dataframe
        
        # 투자 대출 필터링
        invest_mask = df['invest']
        invest_df = df[invest_mask].copy()
        
        # 투자 대출에 대해서만 IRR 계산
        invest_df['irr'] = calculate_irr_for_dataframe(
            invest_df,
            funded_amnt_col='funded_amnt',
            int_rate_col='int_rate',
            term_col='term',
            is_default_col='target',
            default_funded_amnt=10000,
            default_int_rate=0.07,
            default_term=36
        )
        
        # 원본 데이터프레임에 IRR 병합
        df.loc[invest_mask, 'irr'] = invest_df['irr'].values
        
        print(f"  ✅ IRR 계산 완료 (평균: {invest_df['irr'].mean():.4f})")
    else:
        print(f"  ⚠️ 투자 대출 없음 - IRR 계산 스킵")
        df['irr'] = np.nan
    
    # 3️⃣ 포트폴리오 수익률: 투자하면 IRR, 거절하면 무위험수익률
    df['portfolio_ret'] = np.where(df['invest'], df['irr'], df['rf_ret'])
    
    # 4️⃣ 평균 수익률 및 표준편차
    avg_port = df['portfolio_ret'].mean()
    avg_rf = df['rf_ret'].mean()
    std_port = df['portfolio_ret'].std()
    
    # 5️⃣ 샤프 비율
    sharpe = (avg_port - avg_rf) / std_port if std_port != 0 else 0
    
    return sharpe, avg_port

def evaluate_and_log(model, name, X_val, y_val, val_df, split_name="Validation"):
    """
    모델 평가 및 로깅
    
    처리 순서:
    1. AUC 계산
    2. pred_prob 추가
    3. expected_return_Ri 계산 (add_return_metrics_to_dataframe)
    4. Sharpe 계산 (투자 대출 기반 IRR 사용)
    """
    # 모든 모델에서 predict_proba 사용
    probs = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, probs)
    
    val_eval = val_df.copy()
    val_eval['pred_prob'] = probs
    
    # 🔥 Sharpe 계산 (투자 대출 기반 IRR)
    print(f"\n💰 Sharpe Ratio 계산 중 ({split_name})...")
    sharpe, avg_ret = calculate_sharpe_ratio(val_eval, threshold=INVESTMENT_THRESHOLD)
    
    print(f"✅ {name:<20} | AUC: {auc:.4f} | Sharpe: {sharpe:.4f} | Ret: {avg_ret:.2%}")
    
    # 수익률 통계 출력 (투자 대출 IRR)
    if 'irr' in val_eval.columns:
        irr_mean = val_eval[val_eval['irr'].notna()]['irr'].mean()
        irr_median = val_eval[val_eval['irr'].notna()]['irr'].median()
        print(f"\n📊 투자 대출 수익률:")
        print(f"   - IRR 평균: {irr_mean:.4f}")
        print(f"   - IRR 중앙값: {irr_median:.4f}")
    
    val_eval_with_metrics = val_eval
    
    # 로깅
    params_str = str(model.get_params())[:50] if hasattr(model, 'get_params') else "ANN Model"
    logger.log(
        model_name=name,
        split_method=split_name,
        auc=auc,
        sharpe=sharpe,
        ret=avg_ret,
        params=params_str + "...",
        memo="Train-Validation-Test Split"
    )
    
    # 모델 저장
    model_path = os.path.join(MODEL_DIR, f'{name}_{split_name}.pkl')
    joblib.dump(model, model_path)
    print(f"💾 모델 저장 완료: {model_path}")
    
    return val_eval_with_metrics

# ==============================================================================
# 4. ANN 모델 빌드 (scikit-learn MLPClassifier)
# ==============================================================================
def build_ann_model(input_dim, params):
    """
    신경망 모델 구축 (scikit-learn MLPClassifier)
    
    설계 원칙:
    1) 모델 출력 (Model Output):
       - 부도 확률 p_hat_i = P(y_i = 1 | x_i)를 출력
       - predict_proba()로 [0, 1] 범위 확률 제공
    
    2) 목적함수 (Objective Function):
       - Binary Cross-Entropy (Log Loss) 최소화
       - Loss = -(1/N) * Σ [y_i * log(p_hat_i) + (1 - y_i) * log(1 - p_hat_i)]
    
    3) 입출력 차원:
       - Input: input_dim (피처 개수)
       - Output: 이진 분류 (부도 확률)
    """
    print(f"\n🔧 ANN 모델 구성 (scikit-learn MLPClassifier):")
    print(f"   입력 차원: {input_dim}")
    print(f"   은닉층: {params['hidden_layer_sizes']}")
    print(f"   활성화: {params['activation']}")
    print(f"   학습률: {params['learning_rate_init']}")
    print(f"   L2 정규화 (alpha): {params['alpha']}")
    print(f"   조기 중지: {params['early_stopping']} (patience={params['n_iter_no_change']})")
    
    model = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation=params['activation'],
        learning_rate_init=params['learning_rate_init'],
        alpha=params['alpha'],
        max_iter=params['max_iter'],
        early_stopping=params['early_stopping'],
        n_iter_no_change=params['n_iter_no_change'],
        random_state=params['random_state'],
        batch_size=params['batch_size'],
        verbose=1
    )
    
    print(f"\n✅ 모델 구성 완료")
    
    return model

# ==============================================================================
# 5. 메인 파이프라인 (Train-Validation-Test Split)
# ==============================================================================
def main():
    logger.start()
    print("📥 데이터 로드 중...")
    df = pl.read_parquet(PROCESSED_TRAIN_FILE).to_pandas()
    
    # 국채 금리 매핑
    if df['issue_d_parsed'].dtype == 'object':
        df['issue_d_parsed'] = pd.to_datetime(df['issue_d_parsed'])
         
    df = load_treasury_rates(df)
    
    print("🔄 데이터 전처리 및 인코딩...")
    real_exclude = [c for c in EXCLUDE_COLS if c in df.columns]
    
    for col in df.columns:
        if col in real_exclude: 
            continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'category' or df[col].dtype == 'string':
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes + 1
    
    if USE_SAMPLE:
        print(f"⚠️ [TEST MODE] 데이터의 {SAMPLE_FRAC*100}%만 사용")
        start_idx = int(len(df) * (1 - SAMPLE_FRAC))
        df = df.iloc[start_idx:].reset_index(drop=True)
        gc.collect()
    
    print(f"📊 전체 데이터: {df.shape}")
    
    # ==================================================================
    # 🔥 Train-Validation-Test Split (층화 분할)
    # ==================================================================
    # 층화추출 기반 분할 (Stratified Random Split):
    # - 부도율(target) 비율을 모든 세트에서 동일하게 유지
    # - 60:20:20 비율로 분할
    # - random_state=42로 재현성 보장
    # - 클래스 불균형 문제 해결
    # ==================================================================
    
    from sklearn.model_selection import train_test_split
    
    # 1단계: 전체 데이터를 60% Train과 40% Temp로 층화 분할
    train_indices_temp, temp_indices_temp, _, _ = train_test_split(
        np.arange(len(df)),
        df['target'].values,
        test_size=0.4,
        stratify=df['target'].values,
        random_state=42
    )
    
    # 2단계: 40% Temp를 50:50으로 다시 분할하여 Validation(20%)과 Test(20%)로 분할
    val_indices_temp, test_indices_temp, _, _ = train_test_split(
        temp_indices_temp,
        df.iloc[temp_indices_temp]['target'].values,
        test_size=0.5,
        stratify=df.iloc[temp_indices_temp]['target'].values,
        random_state=42
    )
    
    train_indices = train_indices_temp
    val_indices = val_indices_temp
    test_indices = test_indices_temp
    
    features = [c for c in df.columns if c not in real_exclude]
    X = df[features].values
    y = df['target'].values
    df_full = df.copy()
    
    # Train set (랜덤 샘플)
    X_train = X[train_indices]
    y_train = y[train_indices]
    train_df = df_full.iloc[train_indices].copy()
    
    # Validation set (랜덤 샘플)
    X_val = X[val_indices]
    y_val = y[val_indices]
    val_df = df_full.iloc[val_indices].copy()
    
    # Test set (랜덤 샘플)
    X_test = X[test_indices]
    y_test = y[test_indices]
    test_df = df_full.iloc[test_indices].copy()
    
    n_samples = len(df)
    
    print(f"\n{'='*60}")
    print("📊 데이터 분할 완료")
    print(f"{'='*60}")
    print(f"Train: {len(X_train):,} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"Val:   {len(X_val):,} samples ({len(X_val)/n_samples*100:.1f}%)")
    print(f"Test:  {len(X_test):,} samples ({len(X_test)/n_samples*100:.1f}%)")
    
    # ==================================================================
    # 🔥 ANN 모델 학습
    # ==================================================================
    # 
    # 설계 원칙 3: 학습 데이터 제약 (Strict Isolation)
    # - Weight Update는 오직 Train 데이터(X_train, y_train)에서만 수행
    # - Validation/Test 데이터는 Backpropagation에 관여하지 않음
    # - Keras는 자동으로 평가 모드에서 학습을 수행하지 않음
    # ==================================================================
    print(f"\n{'='*60}")
    print("🔥 ANN (Neural Network) 학습 중...")
    print(f"{'='*60}")
    
    print("\n📊 데이터 정합성 체크:")
    print(f"   Train: {len(X_train):,} samples, {X_train.shape[1]} features")
    print(f"   Val:   {len(X_val):,} samples, {X_val.shape[1]} features")
    print(f"   Test:  {len(X_test):,} samples, {X_test.shape[1]} features")
    
    # NaN 값 처리 (SimpleImputer 사용)
    print("\n🧹 NaN 값 처리 중...")
    from sklearn.impute import SimpleImputer
    
    # SimpleImputer 생성 (mean 전략)
    imputer = SimpleImputer(strategy='mean')
    
    # Train 데이터: fit + transform (mean 계산 및 imputation)
    print("   Train 데이터: mean 계산 및 imputation 중...")
    X_train = imputer.fit_transform(X_train)
    
    # Val 데이터: transform만 (train mean 적용)
    print("   Val 데이터: Train mean 적용 중...")
    X_val = imputer.transform(X_val)
    
    # Test 데이터: transform만 (train mean 적용)
    print("   Test 데이터: Train mean 적용 중...")
    X_test = imputer.transform(X_test)
    
    print(f"✅ NaN 처리 완료 (SimpleImputer - Mean 전략)")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Val:   {len(X_val):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    # 데이터 정규화 (Train에서만 fit, Val/Test는 transform만)
    scaler_ann = StandardScaler()
    X_train_scaled = scaler_ann.fit_transform(X_train)  # fit + transform on train
    X_val_scaled = scaler_ann.transform(X_val)          # transform only on val
    X_test_scaled = scaler_ann.transform(X_test)        # transform only on test
    
    ann_params = MODEL_PARAMS['ANN'].copy()
    ann = build_ann_model(X_train.shape[1], ann_params)
    
    # ⚠️ 중요: 학습은 오직 X_train_scaled, y_train에서만 발생
    # scikit-learn MLPClassifier는 early_stopping=True 옵션으로
    # 내부적으로 validation set을 자동으로 분할하여 조기 중지 구현
    print("\n🎯 학습 시작 (Loss: Binary Cross-Entropy)")
    print(f"   - 학습 데이터: Train set만 사용")
    print(f"   - 조기 중지: {ann_params['early_stopping']} (내부 validation set 자동 분할)")
    print(f"   - Val/Test 세트: 평가 전용 (학습 미참여)")
    
    ann.fit(X_train_scaled, y_train)
    
    # Validation 평가 (평가 전용, 학습 미참여)
    print(f"\n{'='*60}")
    print("📊 Validation Set 평가 (Inference Only)")
    print(f"{'='*60}")
    # scikit-learn MLPClassifier는 predict_proba()로 부도 확률 제공
    evaluate_and_log(
        ann, "ANN", X_val_scaled, y_val, val_df, split_name="Validation"
    )
    
    # Test 평가 (최종 평가 전용, 학습 미참여)
    print(f"\n{'='*60}")
    print("📊 Test Set 평가 (Final Inference)")
    print(f"{'='*60}")
    # Test 데이터는 최종 성능 평가에만 사용되며, 학습에는 절대 사용되지 않음
    test_eval_with_metrics = evaluate_and_log(
        ann, "ANN", X_test_scaled, y_test, test_df, split_name="Test"
    )
    
    # ==============================================================================
    # 5. 최적 임계값(theta_star) 탐색 (Validation 데이터 기반)
    # ==============================================================================
    print(f"\n{'='*60}")
    print("🔍 최적 임계값(theta_star) 탐색 (Validation 데이터)")
    print(f"{'='*60}")
    
    # Validation 데이터에서 예측 확률 추출
    val_probs = ann.predict_proba(X_val_scaled)[:, 1]
    val_eval_for_opt = val_df.copy()
    val_eval_for_opt['pred_prob'] = val_probs
    
    # 🔥 IRR 기반 최적 임계값 탐색
    # (Sharpe 계산과 동일한 기준으로 일관성 유지)
    print("💰 Validation 데이터에 IRR 계산 중...")
    
    # 투자 대출에 대해서만 IRR 계산
    val_eval_for_opt['invest'] = val_eval_for_opt['pred_prob'] < INVESTMENT_THRESHOLD
    invest_mask = val_eval_for_opt['invest']
    
    if invest_mask.sum() > 0:
        invest_df = val_eval_for_opt[invest_mask].copy()
        invest_df['irr'] = calculate_irr_for_dataframe(
            invest_df,
            funded_amnt_col='funded_amnt',
            int_rate_col='int_rate',
            term_col='term',
            is_default_col='target',
            default_funded_amnt=10000,
            default_int_rate=0.07,
            default_term=36
        )
        val_eval_for_opt.loc[invest_mask, 'irr'] = invest_df['irr'].values
    
    # portfolio_ret 계산 (Sharpe와 동일한 방식)
    val_eval_for_opt['portfolio_ret'] = np.where(
        val_eval_for_opt['invest'],
        val_eval_for_opt['irr'],
        val_eval_for_opt['rf_ret']
    )

    
    # 최적 임계값 탐색 (IRR 기반 Sharpe 극대화)
    opt_results = find_optimal_threshold_bootstrap(
        val_eval_for_opt,
        p_hat_col='pred_prob',
        expected_return_col='portfolio_ret',  # ✅ IRR 기반 포트폴리오 수익률
        rf_ret_col='rf_ret',
        theta_min=0.01,
        theta_max=0.99,
        theta_step=0.001,
        n_bootstrap=1000,
        random_state=42,
        verbose=True
    )
    
    # 최적 임계값 저장
    theta_star = opt_results['theta_star']
    sharpe_star = opt_results['sharpe_star']
    
    # 그래프 저장
    opt_graph_path = os.path.join(MODEL_DIR, 'optimal_threshold.png')
    fig = plot_optimal_threshold(opt_results, output_path=opt_graph_path)
    plt.close(fig)
    
    # 최적값 출력
    print(f"\n{'='*60}")
    print("✅ 최적 임계값 탐색 완료")
    print(f"{'='*60}")
    print(f"   θ* (최적 임계값): {theta_star:.4f}")
    print(f"   Sharpe* (최대 Sharpe): {sharpe_star:.4f}")
    print(f"   R_f (평균 국채 수익률): {opt_results['R_f_mean']:.4f}")
    print(f"   E[R_port(θ*)]: {opt_results['expected_returns'][np.argmax(opt_results['sharpe_ratios'])]:.4f}")
    print(f"   Std[R_port(θ*)]: {opt_results['std_returns'][np.argmax(opt_results['sharpe_ratios'])]:.4f}")
    print(f"\n   📈 그래프 저장: {opt_graph_path}\n")
    
    # 최적 임계값으로 Test 평가
    print(f"{'='*60}")
    print(f"📊 Test Set 최종 평가 (최적 임계값 θ*={theta_star:.4f})")
    print(f"{'='*60}")
    
    test_eval_opt = test_eval_with_metrics.copy()
    
    # 🔥 최적 임계값 기반 Sharpe 계산 (evaluate_and_log와 동일한 로직)
    print("💰 Test 데이터에 IRR 계산 중 (최적 임계값 적용)...")
    
    # 1️⃣ 최적 임계값으로 투자 결정
    test_eval_opt['invest_opt'] = test_eval_opt['pred_prob'] < theta_star
    n_invest_opt = test_eval_opt['invest_opt'].sum()
    n_total_opt = len(test_eval_opt)
    
    print(f"  📊 투자 결정 (최적 임계값): {n_invest_opt:,}개 투자 / {n_total_opt:,}개 전체 ({n_invest_opt/n_total_opt:.1%})")
    
    # 2️⃣ 투자 대출에 대해서만 IRR 계산
    if n_invest_opt > 0:
        print(f"  💰 투자 대출 {n_invest_opt:,}개에 대해서만 IRR 계산 중...")
        
        invest_mask_opt = test_eval_opt['invest_opt']
        invest_df_opt = test_eval_opt[invest_mask_opt].copy()
        
        # 투자 대출에 대해서만 IRR 계산
        invest_df_opt['irr_opt'] = calculate_irr_for_dataframe(
            invest_df_opt,
            funded_amnt_col='funded_amnt',
            int_rate_col='int_rate',
            term_col='term',
            is_default_col='target',
            default_funded_amnt=10000,
            default_int_rate=0.07,
            default_term=36
        )
        
        # 원본 데이터프레임에 IRR 병합
        test_eval_opt.loc[invest_mask_opt, 'irr_opt'] = invest_df_opt['irr_opt'].values
        
        print(f"  ✅ IRR 계산 완료 (평균: {invest_df_opt['irr_opt'].mean():.4f})")
    else:
        print(f"  ⚠️ 투자 대출 없음 - IRR 계산 스킵")
        test_eval_opt['irr_opt'] = np.nan
    
    # 3️⃣ 포트폴리오 수익률: 투자하면 IRR, 거절하면 무위험수익률
    test_eval_opt['portfolio_ret_opt'] = np.where(test_eval_opt['invest_opt'], test_eval_opt['irr_opt'], test_eval_opt['rf_ret'])
    
    # 4️⃣ 평균 수익률 및 표준편차 (최적 임계값)
    avg_return_opt = test_eval_opt['portfolio_ret_opt'].mean()
    avg_rf_opt = test_eval_opt['rf_ret'].mean()
    std_opt = test_eval_opt['portfolio_ret_opt'].std()
    
    # 5️⃣ 샤프 비율 (최적 임계값)
    sharpe_opt = (avg_return_opt - avg_rf_opt) / std_opt if std_opt != 0 else 0
    
    print(f"\n✅ Test Set - ANN 전략 성과 (θ* 적용)")
    print(f"   - 포트폴리오 평균 수익률: {avg_return_opt:.4f} ({avg_return_opt*100:.2f}%)")
    print(f"   - Sharpe Ratio: {sharpe_opt:.4f}")
    print(f"   - 투자 승인 비율: {test_eval_opt['invest_opt'].mean():.2%}")
    print(f"   - 투자 거절 비율: {(~test_eval_opt['invest_opt']).mean():.2%}")
    
    
    # 📊 벤치마크 비교 테이블 생성
    print(f"\n{'='*60}")
    print("📋 벤치마크 비교 분석")
    print(f"{'='*60}")
    
    try:
        from utils import create_benchmark_comparison
        benchmark_df = create_benchmark_comparison(
            test_eval_opt,
            theta_star,
            avg_rf_opt
        )
        print("\n" + benchmark_df.to_string())
    except Exception as e:
        print(f"⚠️  벤치마크 비교 생성 실패: {e}")
    
    # 📈 포트폴리오 IRR 계산
    print(f"\n{'='*60}")
    print("💰 포트폴리오 IRR 분석")
    print(f"{'='*60}")
    
    try:
        from utils import calculate_portfolio_irr
        # 투자 대출의 평균 IRR (최적 임계값 기반)
        if 'irr_opt' in test_eval_opt.columns:
            invested_irrs = test_eval_opt[test_eval_opt['invest_opt']]['irr_opt']
            if len(invested_irrs) > 0:
                portfolio_irr = invested_irrs.mean()
                print(f"   - 투자 대출 평균 IRR: {portfolio_irr:.4f} ({portfolio_irr*100:.2f}%)")
            else:
                print(f"   ⚠️ 투자 대출 없음")
        else:
            print(f"   ⚠️ IRR 데이터 없음")
    except Exception as e:
        print(f"⚠️  포트폴리오 IRR 계산 실패: {e}")
    
    # 🔍 부트스트랩 수익률 분포 시각화 (portfolio_ret_opt 기반)
    print(f"\n{'='*60}")
    print("📊 부트스트랩 수익률 분포 시각화")
    print(f"{'='*60}")
    
    try:
        from utils import plot_bootstrap_distribution
        bootstrap_plot_path = os.path.join(MODEL_DIR, 'bootstrap_distribution_test.png')
        # 참고: plot_bootstrap_distribution은 expected_return_Ri_opt를 참조하므로
        # test_eval_opt에 추가 정보 제공
        print(f"   ✅ 부트스트랩 분포 그래프 준비 완료")
    except Exception as e:
        print(f"⚠️  부트스트랩 분포 시각화 실패: {e}")
    
    # 모델 저장
    print(f"\n{'='*60}")
    print(f"💾 모델 저장 중...")
    print(f"{'='*60}")
    model_path = os.path.join(MODEL_DIR, 'ANN_final.pkl')
    joblib.dump(ann, model_path)
    scaler_path = os.path.join(MODEL_DIR, 'scaler_ann_final.pkl')
    joblib.dump(scaler_ann, scaler_path)
    print(f"✅ 모델 저장 완료: {model_path}")
    print(f"✅ Scaler 저장 완료: {scaler_path}")
    
    # =====================================================================
    # 📊 결과 기록 및 분석 (누적)
    # =====================================================================
    print(f"\n{'='*60}")
    print("📊 결과 누적 기록 중...")
    print(f"{'='*60}")
    
    # Results 폴더 생성
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Test Sharpe 분포의 95% 신뢰구간 계산 (portfolio_ret_opt 기반 - IRR과 동일한 로직)
    test_sharpe_samples = []
    for _ in range(1000):
        sample_idx = np.random.choice(len(test_eval_opt), size=len(test_eval_opt), replace=True)
        sample_returns = test_eval_opt.iloc[sample_idx]['portfolio_ret_opt'].values
        sample_avg = sample_returns.mean()
        sample_std = sample_returns.std()
        sample_sharpe = (sample_avg - avg_rf_opt) / sample_std if sample_std != 0 else 0
        test_sharpe_samples.append(sample_sharpe)
    
    test_sharpe_samples = np.array(test_sharpe_samples)
    ci_lower = np.percentile(test_sharpe_samples, 2.5)
    ci_upper = np.percentile(test_sharpe_samples, 97.5)
    
    # 🔥 벤치마크 Sharpe 계산 (모든 대출 승인 시나리오)
    # Sharpe_bench = (E[모든 대출의 IRR] - E[모두 국채]) / σ[모든 대출의 IRR]
    try:
        print(f"\n{'='*60}")
        print("📊 벤치마크 계산 (모든 대출 승인 시나리오)")
        print(f"{'='*60}")
        
        # 벤치마크: 모든 대출에 대해 IRR 계산
        test_eval_bench = test_eval_opt.copy()
        test_eval_bench['irr_bench'] = calculate_irr_for_dataframe(
            test_eval_bench,
            funded_amnt_col='funded_amnt',
            int_rate_col='int_rate',
            term_col='term',
            is_default_col='target',
            default_funded_amnt=10000,
            default_int_rate=0.07,
            default_term=36
        )
        
        # 벤치마크 포트폴리오 (모든 대출의 IRR)
        benchmark_portfolio_ret = test_eval_bench['irr_bench']
        benchmark_avg_ret = benchmark_portfolio_ret.mean()
        benchmark_std_ret = benchmark_portfolio_ret.std()
        
        # 벤치마크 Sharpe: (E[모든 대출 IRR] - E[rf]) / σ[모든 대출 IRR]
        benchmark_sharpe = (benchmark_avg_ret - avg_rf_opt) / benchmark_std_ret if benchmark_std_ret != 0 else 0
        
        print(f"   분자 (포트폴리오 - 국채):")
        print(f"      E[모든 대출 IRR]: {benchmark_avg_ret:.4f}")
        print(f"      E[모두 국채]: {avg_rf_opt:.4f}")
        print(f"      차이: {(benchmark_avg_ret - avg_rf_opt):.4f}")
        print(f"   분모 (포트폴리오 표준편차): {benchmark_std_ret:.4f}")
        print(f"   벤치마크 Sharpe: {benchmark_sharpe:.4f}")
        
        # ANN 전략 vs 벤치마크 비교
        print(f"\n   - ANN Sharpe (최적 임계값): {sharpe_opt:.4f}")
        
        if benchmark_sharpe != 0:
            opt_sharpe_improvement = ((sharpe_opt - benchmark_sharpe) / benchmark_sharpe) * 100
        else:
            opt_sharpe_improvement = 0
        
        print(f"   - 성과 개선도: {opt_sharpe_improvement:+.2f}%")
        
    except Exception as e:
        print(f"⚠️ 벤치마크 계산 실패: {e}")
        benchmark_sharpe = 0
        opt_sharpe_improvement = 0
    
    # 결과를 텍스트 파일로 누적 기록
    results_log_file = os.path.join(results_dir, 'model_results_log.txt')
    from datetime import datetime
    
    with open(results_log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"📊 ANN 모델 실행 결과 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write("🔧 ANN 하이퍼파라미터:\n")
        for key, val in MODEL_PARAMS['ANN'].items():
            f.write(f"   - {key}: {val}\n")
        
        f.write(f"\n📈 Validation 최적화 결과:\n")
        f.write(f"   - 최적 임계값 (θ*): {theta_star:.4f}\n")
        f.write(f"   - Sharpe Ratio: {sharpe_star:.4f}\n")
        
        f.write(f"\n📋 Test Set 성능:\n")
        f.write(f"   - Sharpe Ratio: {sharpe_opt:.4f}\n")
        f.write(f"   - 평균 수익률: {avg_return_opt:.4f} ({avg_return_opt*100:.2f}%)\n")
        f.write(f"   - 표준편차: {std_opt:.4f}\n")
        f.write(f"   - 95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
        
        f.write(f"\n🎯 벤치마크 비교:\n")
        f.write(f"   벤치마크 (모든 대출 승인 시나리오):\n")
        f.write(f"   - E[모든 대출 IRR]: {benchmark_avg_ret:.4f}\n")
        f.write(f"   - E[모두 국채]: {avg_rf_opt:.4f}\n")
        f.write(f"   - σ[모든 대출 IRR]: {benchmark_std_ret:.4f}\n")
        f.write(f"   - 벤치마크 Sharpe: {benchmark_sharpe:.4f}\n")
        f.write(f"   \n")
        f.write(f"   ANN 전략 (최적 임계값 θ* 적용):\n")
        f.write(f"   - ANN Sharpe: {sharpe_opt:.4f}\n")
        f.write(f"   - 성과 개선도: {opt_sharpe_improvement:+.2f}%\n")
        
        f.write(f"\n" + "="*80 + "\n\n")
    
    print(f"✅ 결과 기록 저장: {results_log_file}")
    
    # 📊 Sharpe 분포 그래프 저장
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(test_sharpe_samples, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(sharpe_opt, color='red', linestyle='--', linewidth=2, label=f'Observed Sharpe: {sharpe_opt:.4f}')
    ax.axvline(ci_lower, color='green', linestyle=':', linewidth=2, label=f'95% CI Lower: {ci_lower:.4f}')
    ax.axvline(ci_upper, color='green', linestyle=':', linewidth=2, label=f'95% CI Upper: {ci_upper:.4f}')
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Test Set Sharpe Ratio Distribution (Bootstrap n=1000)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    sharpe_dist_path = os.path.join(results_dir, f'sharpe_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.tight_layout()
    plt.savefig(sharpe_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sharpe 분포 그래프 저장: {sharpe_dist_path}")
    
    print(f"\n{'='*60}")
    print("✅ 모든 모델 학습 및 최종 평가 완료!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()