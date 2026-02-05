# src/utils.py

import os
import pandas as pd
import numpy as np
import numpy_financial as npf
from datetime import datetime
import time

class ExperimentLogger:
    def __init__(self, log_file='../experiments_log.csv'):
        # src 폴더의 상위(../)에 로그 파일 생성
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file = os.path.join(base_dir, log_file)
        self.start_time = None
        
    def start(self):
        """실험 시작 시간 기록"""
        self.start_time = time.time()
        
    def log(self, model_name, split_method, auc, sharpe, ret, params, memo=""):
        """결과를 CSV에 한 줄 추가"""
        duration = time.time() - self.start_time if self.start_time else 0
        duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
        
        # 저장할 데이터 딕셔너리
        record = {
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': model_name,
            'Split': split_method,
            'AUC': round(auc, 4),
            'Sharpe': round(sharpe, 4),
            'Avg_Return': f"{ret*100:.2f}%",
            'Duration': duration_str,
            'Memo': memo,
            'Params': str(params)
        }
        
        # DataFrame 변환
        df_new = pd.DataFrame([record])
        
        # 파일이 없으면 헤더 포함 생성, 있으면 데이터만 추가(append)
        if not os.path.exists(self.log_file):
            df_new.to_csv(self.log_file, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(self.log_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            
        print(f"📝 실험 기록 저장 완료! -> {self.log_file}")

# 사용 예시:
# logger = ExperimentLogger()
# logger.start()
# ... 학습 ...
# logger.log("ANN", "Random", 0.7550, 0.0125, 0.0425, {'lr':0.001}, "최적 임계값 θ* 적용")


# ==============================================================================
# 수익률 계산 함수들
# ==============================================================================

def calculate_expected_return_Ri(y_true, p_hat, r_i, L_i, R_f, invest_threshold=0.5):
    """
    개별 수익률 (R_i) 계산 함수
    
    Parameters:
    -----------
    y_true : array-like
        실제 부도 여부 (0: 정상, 1: 부도)
    p_hat : array-like
        예측 부도 확률
    r_i : float or array-like
        대출 연이율 (예: 0.07 = 7%)
    L_i : float
        손실률 (예: 1.0 = 100% 손실)
    R_f : float or array-like
        무위험 국채 수익률 (예: 0.02 = 2%)
    invest_threshold : float
        투자 결정 임계값 (p_hat < threshold이면 투자)
    
    Returns:
    --------
    expected_return : array
        각 대출 건의 예상 수익률
    """
    y_true = np.asarray(y_true)
    p_hat = np.asarray(p_hat)
    
    # 투자 결정: 부도 확률이 임계값보다 낮으면 투자
    invest_decision = p_hat < invest_threshold
    
    # 투자 승인 시 수익률
    # - 정상 상환(y=0): 이자율 r_i
    # - 부도(y=1): 손실률 -L_i
    return_if_invest = np.where(y_true == 0, r_i, -L_i)
    
    # 최종 수익률: 투자하면 return_if_invest, 거절하면 R_f
    expected_return = np.where(invest_decision, return_if_invest, R_f)
    
    return expected_return


def calculate_monthly_payment(principal, annual_rate, term_months):
    """
    원리금 균등 상환 방식의 월 상환액 계산
    
    Parameters:
    -----------
    principal : float
        원금 (funded_amnt)
    annual_rate : float
        연이율 (예: 0.07 = 7%)
    term_months : int
        만기 (개월 수)
    
    Returns:
    --------
    monthly_payment : float
        월 상환액
    """
    if annual_rate == 0:
        return principal / term_months
    
    monthly_rate = annual_rate / 12
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
                      ((1 + monthly_rate)**term_months - 1)
    
    return monthly_payment


def calculate_irr_single_loan(funded_amnt, monthly_payment, term_months, is_default, default_month=None):
    """
    개별 대출 건의 IRR(Internal Rate of Return) 계산
    
    Parameters:
    -----------
    funded_amnt : float
        대출 실행액 (초기 현금 유출)
    monthly_payment : float
        월 상환액
    term_months : int
        만기 (개월 수)
    is_default : bool
        부도 여부 (True: 부도, False: 정상 상환)
    default_month : int, optional
        부도 발생 월 (is_default=True일 때 필요)
    
    Returns:
    --------
    irr : float
        연환산 IRR (예: 0.08 = 8%)
        계산 실패 시 np.nan 반환
    """
    # 현금흐름 초기화: CF_0 = -funded_amnt (대출 실행)
    cash_flows = [-funded_amnt]
    
    if is_default:
        # 부도 시: default_month까지만 상환
        if default_month is None or default_month <= 0:
            default_month = 1
        else:
            default_month = int(default_month)  # float을 int로 변환
        
        term_months = int(term_months)  # float을 int로 변환
        
        # default_month까지 상환액 추가
        cash_flows.extend([monthly_payment] * int(min(default_month, term_months)))
        
        # 나머지 기간은 현금흐름 0
        remaining_months = int(term_months - default_month)
        if remaining_months > 0:
            cash_flows.extend([0] * int(remaining_months))
    else:
        # 정상 상환: term_months 동안 매달 monthly_payment 수취
        cash_flows.extend([monthly_payment] * int(term_months))
    
    # IRR 계산
    try:
        monthly_irr = npf.irr(cash_flows)
        # 월별 IRR을 연환산 IRR로 변환: (1 + monthly_irr)^12 - 1
        if monthly_irr is not None and not np.isnan(monthly_irr):
            annual_irr = (1 + monthly_irr)**12 - 1
            return annual_irr
        else:
            return np.nan
    except:
        return np.nan


def calculate_irr_for_dataframe(df, 
                                 funded_amnt_col='funded_amnt',
                                 int_rate_col='int_rate',
                                 term_col='term',
                                 is_default_col='y',
                                 default_month_col=None,
                                 default_funded_amnt=10000,
                                 default_int_rate=0.07,
                                 default_term=36):
    """
    데이터프레임 전체에 대해 IRR 계산
    
    Parameters:
    -----------
    df : pd.DataFrame
        대출 데이터프레임
    funded_amnt_col : str
        대출 실행액 컬럼명
    int_rate_col : str
        이자율 컬럼명 (소수 형태, 예: 0.07)
    term_col : str
        만기 컬럼명 (개월 수)
    is_default_col : str
        부도 여부 컬럼명 (1: 부도, 0: 정상)
    default_month_col : str, optional
        부도 발생 월 컬럼명 (없으면 term의 50%로 가정)
    default_funded_amnt : float
        funded_amnt 컬럼이 없을 때 기본값
    default_int_rate : float
        int_rate 컬럼이 없을 때 기본값
    default_term : int
        term 컬럼이 없을 때 기본값
    
    Returns:
    --------
    irr_series : pd.Series
        각 대출 건의 IRR
    """
    # 컬럼 존재 여부 확인 및 기본값 설정
    funded_amnt = df[funded_amnt_col] if funded_amnt_col in df.columns else default_funded_amnt
    int_rate = df[int_rate_col] if int_rate_col in df.columns else default_int_rate
    term = df[term_col] if term_col in df.columns else default_term
    is_default = df[is_default_col]
    
    # int_rate가 퍼센트(%) 형식인 경우 소수로 변환 (예: 7 -> 0.07)
    if int_rate_col in df.columns and df[int_rate_col].max() > 1:
        int_rate = int_rate / 100
    
    # default_month 처리
    if default_month_col and default_month_col in df.columns:
        default_month = df[default_month_col]
    else:
        # 부도 발생 월을 만기의 50%로 가정
        default_month = term * 0.5
    
    # 월 상환액 계산 (벡터화)
    monthly_payments = []
    for i in range(len(df)):
        p = funded_amnt.iloc[i] if isinstance(funded_amnt, pd.Series) else funded_amnt
        r = int_rate.iloc[i] if isinstance(int_rate, pd.Series) else int_rate
        t = term.iloc[i] if isinstance(term, pd.Series) else term
        monthly_payments.append(calculate_monthly_payment(p, r, t))
    
    # IRR 계산 (apply 사용)
    def calculate_row_irr(row_idx):
        f_amt = funded_amnt.iloc[row_idx] if isinstance(funded_amnt, pd.Series) else funded_amnt
        m_pay = monthly_payments[row_idx]
        t = term.iloc[row_idx] if isinstance(term, pd.Series) else term
        is_def = is_default.iloc[row_idx]
        def_month = default_month.iloc[row_idx] if isinstance(default_month, pd.Series) else default_month
        
        return calculate_irr_single_loan(f_amt, m_pay, t, is_def == 1, def_month)
    
    irr_values = [calculate_row_irr(i) for i in range(len(df))]
    
    return pd.Series(irr_values, index=df.index)


def add_return_metrics_to_dataframe(df, 
                                      p_hat_col='pred_prob',
                                      y_col='y',
                                      funded_amnt_col='funded_amnt',
                                      int_rate_col='int_rate',
                                      term_col='term',
                                      rf_ret_col='rf_ret',
                                      invest_threshold=0.5,
                                      default_r_i=0.07,
                                      default_L_i=1.0,
                                      default_R_f=0.02,
                                      default_funded_amnt=10000,
                                      default_term=36,
                                      calculate_irr=True):
    """
    데이터프레임에 expected_return_Ri와 irr 컬럼 추가
    
    Parameters:
    -----------
    df : pd.DataFrame
        대출 데이터프레임 (원본은 수정되지 않음)
    p_hat_col : str
        예측 부도 확률 컬럼명
    y_col : str
        실제 부도 여부 컬럼명
    funded_amnt_col : str
        대출 실행액 컬럼명
    int_rate_col : str
        이자율 컬럼명
    term_col : str
        만기 컬럼명
    rf_ret_col : str
        무위험 수익률 컬럼명
    invest_threshold : float
        투자 결정 임계값
    default_r_i : float
        이자율 기본값
    default_L_i : float
        손실률 기본값
    default_R_f : float
        무위험 수익률 기본값
    default_funded_amnt : float
        대출액 기본값
    default_term : int
        만기 기본값
    calculate_irr : bool
        IRR 계산 여부 (계산량이 많으므로 선택 가능)
    
    Returns:
    --------
    df_result : pd.DataFrame
        수익률 지표가 추가된 데이터프레임
    """
    df_result = df.copy()
    
    # 1. Expected Return (R_i) 계산
    r_i = df_result[int_rate_col] if int_rate_col in df_result.columns else default_r_i
    R_f = df_result[rf_ret_col] if rf_ret_col in df_result.columns else default_R_f
    
    # int_rate가 퍼센트(%) 형식인 경우 소수로 변환
    if int_rate_col in df_result.columns and df_result[int_rate_col].max() > 1:
        r_i = r_i / 100
    
    df_result['expected_return_Ri'] = calculate_expected_return_Ri(
        y_true=df_result[y_col],
        p_hat=df_result[p_hat_col],
        r_i=r_i,
        L_i=default_L_i,
        R_f=R_f,
        invest_threshold=invest_threshold
    )
    
    # 2. IRR 계산 (선택적)
    if calculate_irr:
        print("📊 IRR 계산 중... (시간이 소요될 수 있습니다)")
        df_result['irr'] = calculate_irr_for_dataframe(
            df_result,
            funded_amnt_col=funded_amnt_col,
            int_rate_col=int_rate_col,
            term_col=term_col,
            is_default_col=y_col,
            default_funded_amnt=default_funded_amnt,
            default_int_rate=default_r_i,
            default_term=default_term
        )
        print("✅ IRR 계산 완료!")
    
    return df_result


# ==============================================================================
# 최적 임계값 탐색 함수
# ==============================================================================

def find_optimal_threshold_bootstrap(val_df,
                                     p_hat_col='pred_prob',
                                     expected_return_col='expected_return_Ri',
                                     rf_ret_col='rf_ret',
                                     theta_min=0.01,
                                     theta_max=0.50,
                                     theta_step=0.001,
                                     n_bootstrap=1000,
                                     random_state=42,
                                     verbose=True):
    """
    Sharpe Ratio를 극대화하는 최적 임계값(theta_star) 찾기 (Validation: 부트스트래핑 없이)
    
    Parameters:
    -----------
    val_df : pd.DataFrame
        Validation 데이터프레임 (portfolio_ret 컬럼 포함)
    p_hat_col : str
        예측 부도 확률 컬럼명
    expected_return_col : str
        포트폴리오 수익률 컬럼명 (일반적으로 'portfolio_ret')
    rf_ret_col : str
        무위험 수익률 컬럼명
    theta_min : float
        임계값 탐색 범위 최소값
    theta_max : float
        임계값 탐색 범위 최대값
    theta_step : float
        임계값 탐색 단위
    n_bootstrap : int
        (Validation에선 미사용, Test 단계에서 사용하기 위해 유지)
    random_state : int
        난수 시드
    verbose : bool
        진행 상황 출력 여부
    
    Returns:
    --------
    results : dict
        'theta_range': 탐색한 모든 임계값
        'sharpe_ratios': 각 임계값별 Sharpe Ratio
        'theta_star': 최적 임계값
        'sharpe_star': 최대 Sharpe Ratio
        'expected_returns': 각 임계값별 E[R_port]
        'std_returns': 각 임계값별 Std[R_port]
    """
    np.random.seed(random_state)
    
    # 데이터 준비
    p_hat = val_df[p_hat_col].values
    portfolio_ret = val_df[expected_return_col].values  # IRR-based portfolio returns
    rf_ret = val_df[rf_ret_col].values if rf_ret_col in val_df.columns else 0.02
    
    # 평균 무위험 수익률
    if isinstance(rf_ret, (int, float)):
        R_f_mean = rf_ret
    else:
        R_f_mean = np.mean(rf_ret)
    
    # 임계값 범위 생성
    theta_range = np.arange(theta_min, theta_max + theta_step, theta_step)
    sharpe_ratios = []
    expected_returns_mean = []
    std_returns = []
    
    n_samples = len(val_df)
    
    if verbose:
        print(f"🔍 최적 임계값 탐색 시작 (Validation 단계)")
        print(f"   - 탐색 범위: {theta_min:.3f} ~ {theta_max:.3f} (단위: {theta_step:.3f})")
        print(f"   - Validation 샘플: {n_samples}개")
        print(f"   - 각 θ에 대해 Sharpe 계산 (부트스트래핑 없음)\n")
    
    for i, theta in enumerate(theta_range):
        # theta별로 투자 결정: p_hat < theta 면 투자
        invest = (p_hat < theta)
        
        # 포트폴리오 수익률: 투자한 것만 portfolio_ret, 아니면 rf_ret
        R_port = np.where(invest, portfolio_ret, rf_ret)
        
        # E[R_port], Std[R_port] 계산
        E_R_port = np.mean(R_port)
        Std_R_port = np.std(R_port)
        
        # Sharpe Ratio 계산
        if Std_R_port != 0:
            sharpe = (E_R_port - R_f_mean) / Std_R_port
        else:
            sharpe = 0
        
        sharpe_ratios.append(sharpe)
        expected_returns_mean.append(E_R_port)
        std_returns.append(Std_R_port)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"   [{i+1}/{len(theta_range)}] theta={theta:.3f}, Sharpe={sharpe:.4f}")
    
    # 최적 임계값 찾기
    sharpe_ratios = np.array(sharpe_ratios)
    optimal_idx = np.argmax(sharpe_ratios)
    theta_star = theta_range[optimal_idx]
    sharpe_star = sharpe_ratios[optimal_idx]
    
    if verbose:
        print(f"\n✅ 최적 임계값 탐색 완료!")
        print(f"   - 최적 임계값 (theta_star): {theta_star:.3f}")
        print(f"   - 최대 Sharpe Ratio: {sharpe_star:.4f}")
        print(f"   - E[R_port]: {expected_returns_mean[optimal_idx]:.4f}")
        print(f"   - Std[R_port]: {std_returns[optimal_idx]:.4f}\n")
    
    results = {
        'theta_range': theta_range,
        'sharpe_ratios': sharpe_ratios,
        'expected_returns': expected_returns_mean,
        'std_returns': std_returns,
        'theta_star': theta_star,
        'sharpe_star': sharpe_star,
        'R_f_mean': R_f_mean
    }
    
    return results


def plot_optimal_threshold(results, output_path=None):
    """
    최적 임계값 탐색 결과를 그래프로 시각화
    
    Parameters:
    -----------
    results : dict
        find_optimal_threshold_bootstrap()의 반환값
    output_path : str, optional
        그래프 저장 경로 (None이면 저장 안함)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        생성된 그림 객체
    """
    import matplotlib.pyplot as plt
    
    theta_range = results['theta_range']
    sharpe_ratios = results['sharpe_ratios']
    theta_star = results['theta_star']
    sharpe_star = results['sharpe_star']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sharpe Ratio 곡선
    ax.plot(theta_range, sharpe_ratios, 'b-', linewidth=2, label='Sharpe Ratio(θ)')
    
    # 최적 지점 표시
    ax.plot(theta_star, sharpe_star, 'r*', markersize=20, label=f'최적점: θ*={theta_star:.3f}', zorder=5)
    ax.axvline(x=theta_star, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=sharpe_star, color='r', linestyle='--', alpha=0.5)
    
    # 레이블 및 제목
    ax.set_xlabel('Investment Threshold (θ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('최적 임계값(θ*) 탐색 결과\n(부트스트래핑 기반)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # 최적 지점 정보 표시
    info_text = f'θ* = {theta_star:.3f}\nSharpe = {sharpe_star:.4f}'
    ax.text(theta_star, sharpe_star + 0.02, info_text, 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # 저장
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 그래프 저장 완료: {output_path}")
    
    return fig


# ==============================================================================
# 최종 평가 함수들
# ==============================================================================

def calculate_portfolio_irr(df,
                           expected_return_col='expected_return_Ri',
                           funded_amnt_col='funded_amnt',
                           term_col='term'):
    """
    포트폴리오 전체 IRR 계산
    
    Parameters:
    -----------
    df : pd.DataFrame
        대출 데이터프레임 (expected_return_Ri 컬럼 포함)
    expected_return_col : str
        개별 수익률 컬럼명
    funded_amnt_col : str
        대출 실행액 컬럼명
    term_col : str
        만기 컬럼명 (개월 수)
    
    Returns:
    --------
    portfolio_irr : float
        포트폴리오 전체 IRR (연환산)
    """
    # 데이터 준비
    returns = df[expected_return_col].values
    amounts = df[funded_amnt_col].values if funded_amnt_col in df.columns else 10000
    terms = df[term_col].values if term_col in df.columns else 36
    
    if isinstance(amounts, (int, float)):
        amounts = np.full(len(df), amounts)
    if isinstance(terms, (int, float)):
        terms = np.full(len(df), int(terms))
    
    terms = terms.astype(int)
    
    # 가중 평균 수익률 (단순한 방식)
    total_funded = np.sum(amounts)
    weighted_return = np.sum(returns * amounts) / total_funded
    
    # 근사 IRR (연환산 수익률)
    # 더 정확한 IRR 계산을 위해 월별 현금흐름 생성
    monthly_irr = weighted_return / 12
    annual_irr = (1 + monthly_irr)**12 - 1
    
    return annual_irr


def create_benchmark_comparison(test_df,
                                p_hat_col='pred_prob',
                                expected_return_col='expected_return_Ri',
                                rf_ret_col='rf_ret',
                                theta_star=0.15,
                                funded_amnt_col='funded_amnt',
                                term_col='term'):
    """
    ANN 전략과 벤치마크 전략들의 성과를 비교
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test 데이터프레임
    p_hat_col : str
        예측 부도 확률 컬럼명
    expected_return_col : str
        개별 수익률 컬럼명
    rf_ret_col : str
        무위험 수익률 컬럼명
    theta_star : float
        최적 임계값
    funded_amnt_col : str
        대출 실행액 컬럼명
    term_col : str
        만기 컬럼명
    
    Returns:
    --------
    comparison_df : pd.DataFrame
        벤치마크 비교 결과 테이블
    """
    df = test_df.copy()
    p_hat = df[p_hat_col].values
    returns = df[expected_return_col].values
    rf_ret = df[rf_ret_col].values if rf_ret_col in df.columns else 0.02
    
    if isinstance(rf_ret, (int, float)):
        R_f_mean = rf_ret
    else:
        R_f_mean = np.mean(rf_ret)
    
    results = []
    
    # 1. ANN 전략 (θ* 적용)
    ann_decision = p_hat < theta_star
    ann_returns = returns.copy()
    ann_avg_return = np.mean(ann_returns)
    ann_std = np.std(ann_returns)
    ann_sharpe = (ann_avg_return - R_f_mean) / ann_std if ann_std != 0 else 0
    ann_approval_rate = np.mean(ann_decision)
    ai_irr = calculate_portfolio_irr(df.loc[ai_decision] if ai_approval_rate > 0 else df,
                                     expected_return_col, funded_amnt_col, term_col)
    
    results.append({
        'Strategy': 'ANN 모델 (θ*)',
        'Approval Rate': f"{ann_approval_rate:.2%}",
        'Avg Return': f"{ann_avg_return:.4f}",
        'Std Dev': f"{ann_std:.4f}",
        'Sharpe Ratio': f"{ann_sharpe:.4f}",
        'Portfolio IRR': f"{ai_irr:.4f}"
    })
    
    # 2. 벤치마크: 모든 대출 승인
    all_approve = np.ones_like(ai_decision, dtype=bool)
    all_returns = returns.copy()
    all_avg_return = np.mean(all_returns)
    all_std = np.std(all_returns)
    all_sharpe = (all_avg_return - R_f_mean) / all_std if all_std != 0 else 0
    all_irr = calculate_portfolio_irr(df, expected_return_col, funded_amnt_col, term_col)
    
    results.append({
        'Strategy': '모든 대출 승인',
        'Approval Rate': '100.00%',
        'Avg Return': f"{all_avg_return:.4f}",
        'Std Dev': f"{all_std:.4f}",
        'Sharpe Ratio': f"{all_sharpe:.4f}",
        'Portfolio IRR': f"{all_irr:.4f}"
    })
    
    comparison_df = pd.DataFrame(results)
    return comparison_df


def plot_bootstrap_distribution(test_df,
                               p_hat_col='pred_prob',
                               expected_return_col='expected_return_Ri',
                               rf_ret_col='rf_ret',
                               theta_star=0.15,
                               n_bootstrap=1000,
                               random_state=42,
                               output_path=None):
    """
    Test 세트에서 부트스트랩 수익률 분포 히스토그램 생성
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test 데이터프레임
    p_hat_col : str
        예측 부도 확률 컬럼명
    expected_return_col : str
        개별 수익률 컬럼명
    rf_ret_col : str
        무위험 수익률 컬럼명
    theta_star : float
        최적 임계값
    n_bootstrap : int
        부트스트래핑 반복 횟수
    random_state : int
        난수 시드
    output_path : str, optional
        그래프 저장 경로
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        생성된 그림 객체
    bootstrap_means : np.ndarray
        부트스트랩 샘플의 평균들
    """
    import matplotlib.pyplot as plt
    
    np.random.seed(random_state)
    
    p_hat = test_df[p_hat_col].values
    returns = test_df[expected_return_col].values
    rf_ret = test_df[rf_ret_col].values if rf_ret_col in test_df.columns else 0.02
    
    if isinstance(rf_ret, (int, float)):
        R_f_mean = rf_ret
    else:
        R_f_mean = np.mean(rf_ret)
    
    n_samples = len(test_df)
    bootstrap_means = []
    
    # 부트스트래핑
    for i in range(n_bootstrap):
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_returns = returns[bootstrap_idx]
        bootstrap_mean = np.mean(bootstrap_returns)
        bootstrap_means.append(bootstrap_mean)
    
    bootstrap_means = np.array(bootstrap_means)
    
    # 히스토그램 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 왼쪽: ANN 전략 부트스트랩 분포
    ax1.hist(bootstrap_means, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(bootstrap_means), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(bootstrap_means):.4f}')
    ax1.axvline(R_f_mean, color='green', linestyle='--', linewidth=2, label=f'Risk-free: {R_f_mean:.4f}')
    ax1.set_xlabel('Portfolio Return', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title(f'ANN 전략 수익률 분포\n(θ*={theta_star:.3f}, Bootstrap={n_bootstrap})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 오른쪽: 비교 통계
    stats_text = f"""
    최적 임계값: θ* = {theta_star:.4f}
    투자 승인 비율: {(p_hat < theta_star).mean():.2%}
    
    부트스트랩 통계 (n={n_bootstrap}):
    ━━━━━━━━━━━━━━━━━━━━━━━━
    평균 (Mean): {np.mean(bootstrap_means):.6f}
    표준편차 (Std): {np.std(bootstrap_means):.6f}
    최소값 (Min): {np.min(bootstrap_means):.6f}
    최대값 (Max): {np.max(bootstrap_means):.6f}
    
    95% 신뢰구간:
    [{np.percentile(bootstrap_means, 2.5):.6f}, 
     {np.percentile(bootstrap_means, 97.5):.6f}]
    
    무위험 수익률: {R_f_mean:.6f}
    초과 수익률: {np.mean(bootstrap_means) - R_f_mean:.6f}
    """
    
    ax2.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')
    
    plt.tight_layout()
    
    # 저장
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 부트스트랩 분포 그래프 저장 완료: {output_path}")
    
    return fig, bootstrap_means