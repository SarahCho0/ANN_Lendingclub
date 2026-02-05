# 🚀 Lending Club 신용평가 프로젝트 - 설정 가이드

## 📋 프로젝트 목표

Lending Club 데이터로 **ANN 모델**을 학습하여 부도 예측 및 **Sharpe Ratio 최대화 투자 임계값(θ*)** 도출

---

## 🔄 전체 파이프라인 흐름

```
1️⃣ 데이터 분할 (층화추출)
   Train(60%) / Validation(20%) / Test(20%)
   
2️⃣ ANN 모델 학습 (Train 데이터)
   
3️⃣ Validation 평가 + 최적 임계값(θ*) 탐색
   → Sharpe Ratio 최대화하는 θ* 찾기
   
4️⃣ Test 평가 (θ* 적용)
   → 최종 Sharpe, 포트폴리오 수익률, 신뢰구간
   
5️⃣ 벤치마크 비교
   → 모든 대출 승인 시나리오 vs ANN 전략
```

---

## ⚙️ 환경 설정

### 가상환경 생성 (첫 설정)
```bash
python3 -m venv ~/ann_venv_new
source ~/ann_venv_new/bin/activate
pip install -r requirements.txt
```

### 이후 실행
```bash
source ~/ann_venv_new/bin/activate
python src/train_models.py
```

---

## 🎯 핵심 알고리즘: Sharpe Ratio 계산 (IRR 기반)

### Sharpe Ratio 공식
$$\text{Sharpe} = \frac{E[R_p] - E[R_f]}{\sigma[R_p]}$$

**핵심**: 투자 결정에 따른 포트폴리오 구성
```python
# 부도 확률 < θ인 대출만 투자
invest = (pred_prob < θ)

# 포트폴리오 수익률 구성
portfolio_return = {
    IRR          if 투자 (pred_prob < θ)
    rf_return    if 거절 (pred_prob ≥ θ)
}

# Sharpe 계산
Sharpe = (mean(portfolio_return) - mean(rf_return)) / std(portfolio_return)
```

### IRR(내부수익률) 계산

**원리금균등상환** 기반 현금흐름:
```
월 상환액 = P × [r(1+r)^n] / [(1+r)^n - 1]
현금흐름 = [-P, A, A, ..., A] (부도 시 조기 종료)

월별 IRR → 연율화: annual_irr = (1 + monthly_irr)^12 - 1
```

### 최적 임계값(θ*) 탐색

**목표**: Sharpe Ratio 최대화하는 θ 찾기

**Validation 단계 (θ* 결정)**:
1. Validation 데이터로 990개 θ값(1%~99%, 0.1% 단위) 각각 테스트
2. 각 θ에 대해 Sharpe Ratio 계산
3. Sharpe 최대인 **θ* 1개 선택**

**Test 단계 (신뢰구간 계산)**:
1. 선택된 θ*를 Test 데이터에 적용
2. 1,000회 부트스트래핑 실행 → Sharpe 분포 생성
3. **95% 신뢰구간** 계산 (하위 2.5%, 상위 97.5% 백분위수)

---

## 📊 최종 평가 (Test Set)

### 1️⃣ 포트폴리오 성과
```
- 평균 수익률: E[portfolio_return]
- Sharpe Ratio: (mean - rf_mean) / std
- 투자 승인 비율: 전체 대출 중 투자 비율
```

### 2️⃣ 벤치마크 비교 (모든 대출 승인 시나리오)
```
벤치마크 Sharpe = (E[모든 대출 IRR] - E[국채]) / σ[모든 대출 IRR]

성과 개선도 = (ANN Sharpe - 벤치마크) / 벤치마크 × 100%
```

### 3️⃣ 신뢰구간
```
95% 신뢰구간: [CI_lower, CI_upper]
→ Test Sharpe 분포의 하위 2.5%, 상위 97.5% 백분위수
```

---

## 🏃 실행 방법

```bash
# 1. 가상환경 활성화
source ~/ann_venv_new/bin/activate

# 2. 전체 파이프라인 실행 (자동 수행)
python src/train_models.py

# 3. 결과 확인
cat results/model_results_log.txt
```

**생성 파일**:
- `models/ANN_final.pkl` - 학습된 모델
- `models/optimal_threshold.png` - θ* 시각화
- `results/model_results_log.txt` - **누적 결과 기록** ⭐

---

## 📝 v0.7 주요 특징

- ✅ **일관된 Sharpe 계산**: Validation, 최적 임계값 탐색, Test 모두 **같은 IRR 기반 Sharpe** 사용
- ✅ **벤치마크**: 모든 대출 승인 vs ANN 선택적 투자 비교
- ✅ **효율성**: 투자 대출만 IRR 계산 → 연산량 50% 감소
- ✅ **신뢰성**: 부트스트랩 신뢰구간으로 결과 검증

---

**프로젝트 상태**: ✅ Ready to Run
