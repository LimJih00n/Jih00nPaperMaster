# Topic Coverage-based Demonstration Retrieval for In-Context Learning - 수학적 기초 분석

## 🔢 핵심 수식 분해 (5개 수식)

### 수식 1: 토픽 커버리지 기반 관련성 점수 (Topic Coverage-aware Relevance)

```python
# 수학적 정의
r(x, d) = ∑(t∈T) (t̂_x,t · t̂_d,t) / t̂_LM,t = ⟨t̂_x ⊘ t̂_LM, t̂_d⟩

# 물리적 의미
"이 수식은 '테스트 입력 x와 후보 시연 d 간의 토픽 레벨 관련성'을 계산하는 것"

# 직관적 비유
"필요한 지식 영역(required topics)을 모델이 약한 부분 위주로 가중치를 주어, 
 그 영역을 잘 커버하는 시연을 우선 선택하는 것"

# 구체적 예시 ('herbivore' 질문)
test_input = "Non-human organisms that mainly consume plants are known as what?"
required_topics = {"herbivore": 0.87, "carnivore": 0.91, "omnivore": 0.90}
model_knowledge = {"herbivore": 0.75, "carnivore": 0.72, "omnivore": 0.85}
demonstration_topics = {"herbivore": 0.96, "plant": 0.27}

# 관련성 점수 계산
r(x, d) = (0.87 * 0.96 / 0.75) + (0 * 0 / 0.72) + (0 * 0 / 0.85)
       = 1.11  # 높은 점수 → herbivore 관련 시연 선택

# 차원 분석
t̂_x: [|T|] → 테스트 입력의 토픽 분포 벡터
t̂_d: [|T|] → 시연의 토픽 분포 벡터  
t̂_LM: [|T|] → 모델의 토픽별 지식 벡터
r(x, d): scalar → 최종 관련성 점수
```

### 수식 2: 토픽별 모델 지식 평가 (Topical Knowledge Assessment)

```python
# 수학적 정의
t̂_LM,t = (∑(d∈D) t̂_d,t · zero-shot(d)) / (∑(d∈D) t̂_d,t)

where zero-shot(d) = 1[y = arg max_ŷ p_LM(ŷ|x)]

# 물리적 의미
"이 수식은 '모델이 특정 토픽 t에 대해 얼마나 잘 아는지'를 측정하는 것"

# 직관적 비유
"각 토픽별로 모델의 '시험 점수'를 매기는 것 - 
 해당 토픽 관련 문제들을 zero-shot으로 얼마나 잘 푸는지"

# 구체적 예시 ('carnivore' 토픽)
carnivore_demonstrations = [
    {"topic_weight": 0.87, "zero_shot_correct": 1},  # 맞춤
    {"topic_weight": 0.65, "zero_shot_correct": 1},  # 맞춤
    {"topic_weight": 0.43, "zero_shot_correct": 0}   # 틀림
]

t̂_LM,carnivore = (0.87*1 + 0.65*1 + 0.43*0) / (0.87 + 0.65 + 0.43)
                = 1.52 / 1.95 = 0.78  # 78% 정확도

# 차원 분석
t̂_d,t: scalar ∈ [0,1] → 시연 d에서 토픽 t의 중요도
zero-shot(d): binary {0,1} → 해당 시연을 zero-shot으로 맞췄는지
t̂_LM,t: scalar ∈ [0,1] → 토픽 t에 대한 모델의 지식 수준
```

### 수식 3: 누적 토픽 커버리지 (Cumulative Topic Coverage)

```python
# 수학적 정의
t̂_d ← (t̂_{d∪D'_x} - t̂_{D'_x})

where t̂_{d∪D'_x} = f(e_{d∪D'_x}) and e_{d∪D'_x} = (e_d + ∑_{d'∈D'_x} e_{d'}) / (1 + |D'_x|)

# 물리적 의미
"이 수식은 '이미 선택된 시연들이 커버하지 않은 새로운 토픽 영역'을 계산하는 것"

# 직관적 비유
"이미 가진 퍼즐 조각들을 제외하고, 새로운 조각이 채워줄 수 있는 빈 공간을 찾는 것"

# 구체적 예시 (동물 관련 질문)
previous_demonstrations = [
    "carnivore 관련 시연",  # carnivore: 0.8 커버됨
    "omnivore 관련 시연"   # omnivore: 0.9 커버됨
]

candidate_demonstration = "herbivore 관련 시연"
original_coverage = {"carnivore": 0.7, "omnivore": 0.6, "herbivore": 0.9}
previous_coverage = {"carnivore": 0.8, "omnivore": 0.9, "herbivore": 0.1}

# 새로운 기여도 계산
new_contribution = {
    "carnivore": max(0, 0.7 - 0.8) = 0,    # 이미 충분히 커버됨
    "omnivore": max(0, 0.6 - 0.9) = 0,     # 이미 충분히 커버됨  
    "herbivore": max(0, 0.9 - 0.1) = 0.8   # 새로운 기여!
}

# 차원 분석
t̂_d: [|T|] → 시연의 원래 토픽 분포
t̂_{D'_x}: [|T|] → 이전 시연들의 누적 토픽 커버리지
t̂_{d∪D'_x}: [|T|] → 현재 시연을 포함한 전체 커버리지
updated_t̂_d: [|T|] → 중복 제거된 새로운 기여 토픽 분포
```

### 수식 4: 구별성 인식 학습 신호 (Distinctiveness-aware Training Signal)

```python
# 수학적 정의
DST(d, t) = exp(BM25(d, t)) / (1 + ∑_{d'∈D_d} exp(BM25(d', t)))

t_{d,t} = DST(d, t) / max_{t'∈T_d} DST(d, t')

# 물리적 의미
"이 수식은 '특정 시연에서 해당 토픽이 얼마나 독특하고 구별되는지'를 측정하는 것"

# 직관적 비유
"교실에서 특정 학생이 가진 특별한 재능의 '희귀성'을 평가하는 것 - 
 다른 학생들도 가진 흔한 재능보다 독특한 재능에 높은 점수"

# 구체적 예시 ('photosynthesis' 토픽)
demonstration_d = "Plants convert sunlight into energy through photosynthesis"
nearby_demonstrations = [
    "Animals eat plants for energy",      # photosynthesis: BM25=0.1
    "Energy flows through food chains",   # photosynthesis: BM25=0.0
    "Chlorophyll helps in photosynthesis" # photosynthesis: BM25=0.8
]

# 구별성 점수 계산
DST(d, "photosynthesis") = exp(0.9) / (1 + exp(0.1) + exp(0.0) + exp(0.8))
                          = 2.46 / (1 + 1.11 + 1.0 + 2.23) = 0.46

# 정규화된 소프트 레이블
t_{d,"photosynthesis"} = 0.46 / max(모든 토픽 DST 점수들) = 0.46 / 0.52 = 0.88

# 차원 분석
BM25(d, t): scalar → 시연 d에서 토픽 t의 어휘적 중요도
DST(d, t): scalar ∈ [0,1] → 토픽 t에 대한 시연 d의 구별성
t_{d,t}: scalar ∈ [0,1] → 정규화된 소프트 트레이닝 타겟
```

### 수식 5: 토픽 예측기 손실 함수 (Topic Predictor Loss)

```python
# 수학적 정의
L_TP = -∑_{d∈D} [∑_{t∈T_d} t_{d,t} log t̂_{d,t} + ∑_{t∉T_d} log(1 - t̂_{d,t})]

# 물리적 의미
"이 수식은 '토픽 예측기가 각 시연의 토픽 분포를 얼마나 정확히 예측하는지'를 측정하는 것"

# 직관적 비유
"학생이 문서를 보고 주제를 맞히는 시험에서, 정답과 예측 간의 차이를 페널티로 주는 것"

# 구체적 예시 (생물학 시연)
demonstration = "Herbivores are animals that primarily eat plants"
true_topics = {"herbivore": 0.9, "animal": 0.6, "plant": 0.4}
predicted_topics = {"herbivore": 0.8, "animal": 0.7, "plant": 0.3}
non_topics = {"carnivore": 0.1, "mineral": 0.05}

# 손실 계산
positive_loss = -(0.9 * log(0.8) + 0.6 * log(0.7) + 0.4 * log(0.3))
              = -(0.9 * (-0.22) + 0.6 * (-0.36) + 0.4 * (-1.20))
              = -(-0.198 - 0.216 - 0.48) = 0.894

negative_loss = -(log(1-0.1) + log(1-0.05))
              = -(log(0.9) + log(0.95))
              = -(-0.105 - 0.051) = 0.156

total_loss = 0.894 + 0.156 = 1.05

# 차원 분석
t_{d,t}: scalar ∈ [0,1] → 실제 토픽 중요도 (소프트 레이블)
t̂_{d,t}: scalar ∈ [0,1] → 예측된 토픽 중요도
L_TP: scalar ≥ 0 → 전체 이진 교차 엔트로피 손실
```

## 🧮 수학적 직관 (Why does it work?)

### 왜 이렇게 설계했는가?

1. **토픽 기반 분해의 이론적 근거**:
```python
# 논문의 이론적 정당화 (수식 8)
p(x|d) = p(x) · ∑_t [p(t|x) · p(t|d) / p(t)]
       = p(x) · ∑_t [required_topics · covered_topics / topical_knowledge]
```

2. **불확실성 최소화와의 연결**:
- 기존 ConE 방법: H(x|d) 최소화 (비싸고 느림)
- TopicK: 토픽 모델링으로 분해하여 효율적으로 동일한 목적 달성

3. **다양성과 관련성의 균형**:
- 관련성: t̂_x,t (필요한 토픽에 높은 가중치)
- 다양성: 누적 커버리지로 중복 방지
- 모델 인식: t̂_LM,t (모델이 약한 부분 우선)

### 다른 대안들과 비교

| 방법 | 장점 | 단점 | TopicK 개선점 |
|------|------|------|---------------|
| Similarity-based | 빠름, 단순함 | 모델 무관, 표면적 | 모델 지식 고려 |
| Uncertainty-based | 모델 맞춤 | 느림, 다양성 부족 | 효율성 + 다양성 |
| Random | 매우 빠름 | 성능 낮음 | 체계적 선택 |

### 수학적 성질과 특징

1. **스케일 불변성**: z-score 정규화로 토픽별 점수 범위 통일
2. **단조성**: 더 관련 있는 토픽일수록 높은 점수
3. **서브모듈성**: 누적 커버리지가 diminishing returns 특성
4. **계산 복잡도**: O(K × |T|) - LLM 추론보다 훨씬 효율적

## 📊 그래디언트 분석

```python
# Topic Predictor 역전파 수식 도출
∂L_TP/∂t̂_{d,t} = {
    -t_{d,t}/t̂_{d,t}           if t ∈ T_d (positive class)
    1/(1-t̂_{d,t})              if t ∉ T_d (negative class)
}

# MLP 파라미터에 대한 그래디언트
∂L_TP/∂W = ∑_d ∂L_TP/∂t̂_d · ∂t̂_d/∂W
where t̂_d = softmax(W₃ · ReLU(W₂ · ReLU(W₁ · e_d)))

# 그래디언트 흐름 시각화
"""
Input embedding (e_d) 
    ↓ ∂L/∂e_d = W₁ᵀ · grad
Hidden Layer 1 (W₁)
    ↓ ∂L/∂W₁ = e_d · grad₁ᵀ  
Hidden Layer 2 (W₂)
    ↓ ∂L/∂W₂ = h₁ · grad₂ᵀ
Output Layer (W₃)
    ↓ ∂L/∂W₃ = h₂ · grad₃ᵀ
Topic Predictions (t̂_d)
    ↓ ∂L/∂t̂_{d,t} (위에서 계산)
Loss (L_TP)
"""

# 안정적인 학습을 위한 그래디언트 클리핑
grad_norm = ||∇L_TP||₂
if grad_norm > max_grad_norm:
    ∇L_TP ← ∇L_TP · (max_grad_norm / grad_norm)
```

이 수학적 기초 분석을 통해 TopicK의 각 구성요소가 어떻게 작동하고 왜 효과적인지 명확히 이해할 수 있습니다.