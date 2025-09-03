# Attention Is All You Need - 수학적 기초 완전분해

## 🔢 핵심 수식 완전분해 (3개 핵심 수식)

### 수식 1: Scaled Dot-Product Attention 💎

```python
# 수학적 정의
Attention(Q,K,V) = softmax(QK^T/√d_k)V

# 물리적 의미
"이 수식은 'Query가 어떤 Key에 얼마나 집중할지를 계산하여 Value의 가중합을 만드는' 것"

# 직관적 비유 
"스포트라이트(Query)가 무대의 배우들(Key)을 비추면서, 밝게 비춘 배우의 연기(Value)를 더 많이 보는 것"

# 구체적 예시: "I love you" 처리 과정
sentence = ["I", "love", "you"]
# Query = "love"가 다른 단어들과의 관련성을 찾는 상황

step1_scores = {
    "love → I":    Q_love @ K_I    = 0.1,    # 약한 관련성
    "love → love": Q_love @ K_love = 0.9,    # 자기 자신 (강함)  
    "love → you":  Q_love @ K_you  = 0.8     # 강한 관련성 (love의 대상)
}

step2_scaling = {
    "love → I":    0.1 / √64 = 0.0125,
    "love → love": 0.9 / √64 = 0.1125, 
    "love → you":  0.8 / √64 = 0.1000
}

step3_softmax = {
    "love → I":    exp(0.0125) / sum = 0.1,   # 10% 집중
    "love → love": exp(0.1125) / sum = 0.5,   # 50% 집중  
    "love → you":  exp(0.1000) / sum = 0.4    # 40% 집중
}

step4_weighted_sum = {
    final_representation = 0.1*V_I + 0.5*V_love + 0.4*V_you
    # "love"의 새로운 표현 = 자기 자신 50% + "you" 40% + "I" 10%
}

# 차원 분석 (seq_len=3, d_k=64 예시)
Q: [1, 3, 64]        # 배치=1, 시퀀스=3, 임베딩차원=64
K: [1, 3, 64]        # Key도 동일한 차원
V: [1, 3, 64]        # Value도 동일한 차원

QK^T: [1, 3, 64] @ [1, 64, 3] = [1, 3, 3]    # Attention Score Matrix
#      ↑Query      ↑Key전치    ↑모든 단어쌍의 유사도

Softmax([1, 3, 3]): [1, 3, 3]                # 확률분포로 변환
#  I    love  you
# [0.1   0.2   0.1]  ← "I"가 집중하는 정도
# [0.1   0.5   0.4]  ← "love"가 집중하는 정도  
# [0.2   0.3   0.5]  ← "you"가 집중하는 정도

최종결과: [1, 3, 3] @ [1, 3, 64] = [1, 3, 64]  # 새로운 표현
#        ↑Attention   ↑Value      ↑업데이트된 임베딩
```

**🤔 왜 √d_k로 나누는가?**
```python
# 문제: d_k가 클수록 내적값이 커짐 → softmax가 극단적으로 집중
d_k = 512일 때: QK^T ≈ 22.6 → softmax 거의 one-hot
d_k = 64일 때:  QK^T ≈ 8.0  → softmax 적당히 분산

# 해결: √d_k로 normalize → 항상 적정한 attention 분포 유지
scaled_score = raw_score / √d_k  # 표준편차 정규화 효과
```

### 수식 2: Multi-Head Attention 🧠

```python
# 수학적 정의
MultiHead(Q,K,V) = Concat(head_1, head_2, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

# 물리적 의미
"여러 개의 서로 다른 attention을 병렬로 계산하여 다양한 관점에서 정보를 수집하는 것"

# 직관적 비유
"8명의 서로 다른 전문가가 같은 문장을 분석하는 것"
# 전문가1: 문법적 관계 (주어-동사)
# 전문가2: 의미적 관계 (동사-목적어)  
# 전문가3: 위치적 관계 (인접성)
# ... 등등

# 구체적 예시: "I love you"를 8개 head로 분석
sentence = ["I", "love", "you"]

# Head 1: 주어-동사 관계에 특화 (W_1^Q, W_1^K, W_1^V 학습됨)
head_1_attention = {
    "I":    [0.8, 0.1, 0.1],  # "I"는 "love"에 집중 (주어→동사)
    "love": [0.7, 0.2, 0.1],  # "love"는 "I"에 집중 (동사→주어)
    "you":  [0.1, 0.1, 0.8]   # "you"는 자기 자신에 집중
}

# Head 2: 동사-목적어 관계에 특화
head_2_attention = {
    "I":    [0.6, 0.1, 0.3],  # "I"는 약간 "you"에도 관심
    "love": [0.1, 0.2, 0.7],  # "love"는 "you"에 집중 (동사→목적어)
    "you":  [0.1, 0.6, 0.3]   # "you"는 "love"에 집중 (목적어→동사)
}

# Head 3: 위치적 인접성에 특화
head_3_attention = {
    "I":    [0.5, 0.4, 0.1],  # 인접한 "love"에 집중
    "love": [0.3, 0.4, 0.3],  # 양쪽 모두와 균등한 관계
    "you":  [0.1, 0.4, 0.5]   # 인접한 "love"에 집중
}

# ... head 4-8도 각각 다른 패턴 학습

# 차원 분석 (h=8, d_model=512, d_k=d_v=64)
d_model = 512  # 전체 모델 차원
h = 8          # head 개수  
d_k = d_v = d_model // h = 64  # 각 head당 차원

# 각 head별 변환 행렬들
W_i^Q: [512, 64]  # Query 변환 (8개)
W_i^K: [512, 64]  # Key 변환 (8개)
W_i^V: [512, 64]  # Value 변환 (8개)
W^O:   [512, 512] # 최종 출력 변환 (1개)

# 처리 과정
입력: X [1, 3, 512]

# Step 1: 각 head별로 Q,K,V 변환
Q_i = X @ W_i^Q  # [1, 3, 512] @ [512, 64] = [1, 3, 64] (8개)
K_i = X @ W_i^K  # [1, 3, 512] @ [512, 64] = [1, 3, 64] (8개)  
V_i = X @ W_i^V  # [1, 3, 512] @ [512, 64] = [1, 3, 64] (8개)

# Step 2: 각 head별로 attention 계산  
head_i = Attention(Q_i, K_i, V_i)  # [1, 3, 64] (8개)

# Step 3: 모든 head 결과 연결
concatenated = Concat(head_1, ..., head_8)  # [1, 3, 512]
#                    ↑64 × 8 = 512

# Step 4: 최종 변환
output = concatenated @ W^O  # [1, 3, 512] @ [512, 512] = [1, 3, 512]
```

**🤔 왜 여러 head로 나누는가?**
```python
# 1개 head (512차원): 한 가지 관점으로만 attention 
single_attention = softmax(Q@K^T/√512) @ V  # 단일 관점

# 8개 head (각 64차원): 8가지 다른 관점으로 attention
multi_attention = [
    문법_관점_attention,     # head 1
    의미_관점_attention,     # head 2  
    위치_관점_attention,     # head 3
    감정_관점_attention,     # head 4
    # ... 등등
]

# 결과: 더 풍부한 표현력 + 안정적인 학습
```

### 수식 3: Positional Encoding ⏰

```python
# 수학적 정의
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# 물리적 의미  
"위치 정보가 없는 attention에게 '순서'를 알려주는 고유한 위치 신호"

# 직관적 비유
"각 자리수마다 서로 다른 주파수로 진동하는 시계"
# 1초마다 빠르게 진동 (고주파) + 1분마다 천천히 진동 (저주파)
# → 몇 초, 몇 분인지 정확히 알 수 있음

# 구체적 예시: "I love you"의 각 위치별 encoding
sentence = ["I", "love", "you"]
positions = [0, 1, 2]
d_model = 512

# 위치 0 ("I")의 positional encoding 계산
pos = 0
PE_0 = [
    sin(0/10000^(0/512)),    # i=0: sin(0) = 0
    cos(0/10000^(0/512)),    # i=0: cos(0) = 1
    sin(0/10000^(2/512)),    # i=1: sin(0) = 0  
    cos(0/10000^(2/512)),    # i=1: cos(0) = 1
    sin(0/10000^(4/512)),    # i=2: sin(0) = 0
    cos(0/10000^(4/512)),    # i=2: cos(0) = 1
    # ... 총 512차원까지
] # → [0, 1, 0, 1, 0, 1, ...]

# 위치 1 ("love")의 positional encoding
pos = 1  
PE_1 = [
    sin(1/10000^(0/512)),    # sin(1) ≈ 0.841
    cos(1/10000^(0/512)),    # cos(1) ≈ 0.540
    sin(1/10000^(2/512)),    # sin(1/10000^(2/512)) ≈ sin(0.999) ≈ 0.841
    cos(1/10000^(2/512)),    # cos(1/10000^(2/512)) ≈ cos(0.999) ≈ 0.540  
    sin(1/10000^(4/512)),    # 더 낮은 주파수
    cos(1/10000^(4/512)),    # 더 낮은 주파수
    # ...
] # → [0.841, 0.540, 0.841, 0.540, ...]

# 위치 2 ("you")의 positional encoding  
pos = 2
PE_2 = [
    sin(2/10000^(0/512)),    # sin(2) ≈ 0.909
    cos(2/10000^(0/512)),    # cos(2) ≈ -0.416
    sin(2/10000^(2/512)),    # sin(2/10000^(2/512)) ≈ 0.909
    cos(2/10000^(2/512)),    # cos(2/10000^(2/512)) ≈ -0.416
    # ...
] # → [0.909, -0.416, 0.909, -0.416, ...]

# 실제 사용: 원본 임베딩에 더하기
original_embeddings = {
    "I":    [0.1, 0.2, 0.3, 0.4, ...],  # 단어의 의미 정보
    "love": [0.5, 0.6, 0.7, 0.8, ...],  
    "you":  [0.9, 1.0, 1.1, 1.2, ...]
}

final_embeddings = {
    "I":    original_embeddings["I"] + PE_0,      # 의미 + 위치정보  
    "love": original_embeddings["love"] + PE_1,   # 의미 + 위치정보
    "you":  original_embeddings["you"] + PE_2     # 의미 + 위치정보
}

# 차원 분석
PE: [max_seq_len, d_model] = [5000, 512]  # 미리 계산된 테이블
입력_임베딩: [batch, seq_len, d_model] = [1, 3, 512]  
위치_인코딩: [seq_len, d_model] = [3, 512]  # 해당 위치만 선택
최종_임베딩: [1, 3, 512] = 입력_임베딩 + 위치_인코딩  # element-wise 덧셈
```

**🤔 왜 sin/cos 함수를 사용하는가?**
```python
# 대안 1: 단순 정수 [0, 1, 2, 3, ...] 
문제: 시퀀스 길이에 따라 값의 범위가 달라짐 → 학습 불안정

# 대안 2: 학습 가능한 위치 임베딩  
문제: 최대 길이 고정 → 더 긴 시퀀스 처리 불가

# 선택: sin/cos 주기함수
장점1: 값 범위 [-1, 1]로 고정 → 안정적 학습
장점2: 임의의 길이 시퀀스 처리 가능  
장점3: 상대적 위치 관계도 학습 가능

# 상대 위치 관계 학습
PE(pos+k) = f(PE(pos), k)  # k만큼 떨어진 위치는 함수로 표현 가능
# sin(A+B) = sin(A)cos(B) + cos(A)sin(B) 공식 활용
```

## 🧮 수학적 직관 (Why does it work?)

### 🎯 왜 Attention이 RNN보다 효과적인가?

```python
# RNN의 정보 전달 경로
"I love you" → RNN 처리
step_1: h_1 = f(h_0, "I")        # "I" 정보가 h_1에 저장
step_2: h_2 = f(h_1, "love")     # "I" 정보가 h_1을 거쳐 h_2로 (손실 발생)
step_3: h_3 = f(h_2, "you")      # "I" 정보가 h_1→h_2→h_3으로 (더 많은 손실)

결과: "you"가 "I"의 정보에 접근하려면 2단계 거쳐야 함 → 정보 손실

# Attention의 정보 전달 경로  
"I love you" → Attention 처리
"you"의 Query가 "I"의 Key와 직접 계산: Q_you @ K_I
결과: "you"가 "I"의 정보에 직접 접근 → 정보 손실 없음

# 수학적 증명
RNN 정보전달: P(정보보존) = α^n  (α < 1, n = 거리)
Attention 정보전달: P(정보보존) = softmax 가중치 (거리와 무관)
```

### 🎯 Multi-Head가 단일 Head보다 좋은 이유

```python
# 표현 공간의 다양성
Single_Head_512D = "하나의 큰 방에 모든 정보를 담기"
Multi_Head_8×64D = "8개의 전문 방에 특화된 정보를 담기"

# 수학적 관점: 저차원 특화 vs 고차원 일반화
single_head_capacity = 1 × 512²  = 262,144 parameters
multi_head_capacity  = 8 × 64²   = 32,768 parameters (per head)
total_parameters     = 8 × 32,768 = 262,144 parameters (동일)

하지만 학습 효율성:
single_head: 하나의 복잡한 함수 학습 (어려움)
multi_head:  8개의 간단한 함수 학습 (쉬움) + 조합으로 복잡성 달성
```

### 🎯 √d_k Scaling의 수학적 필연성

```python
# 내적의 분산 분석
Q, K ~ N(0, 1)이라 가정 (정규분포)
QK^T의 각 원소는 d_k개 독립변수의 합

E[QK^T] = 0              # 기댓값은 0
Var[QK^T] = d_k          # 분산은 d_k에 비례

# d_k가 커질수록 문제 발생
d_k = 64:  QK^T ~ N(0, 64)   → softmax 적당히 분산
d_k = 512: QK^T ~ N(0, 512)  → softmax 극도로 집중 (거의 one-hot)

# √d_k로 나누면
scaled_scores = QK^T / √d_k
Var[scaled_scores] = Var[QK^T] / d_k = d_k / d_k = 1  # 항상 1로 정규화!

결과: d_k에 관계없이 안정적인 attention 분포 유지
```

## 📊 그래디언트 분석

### Attention의 역전파 경로

```python
# Forward Pass
scores = QK^T / √d_k           # [batch, seq, seq]
weights = softmax(scores)      # [batch, seq, seq]  
output = weights @ V           # [batch, seq, d_v]

# Backward Pass (Chain Rule)
∂L/∂V = weights^T @ (∂L/∂output)              # Value gradient
∂L/∂weights = (∂L/∂output) @ V^T              # Attention weights gradient
∂L/∂scores = ∂softmax/∂scores ⊙ ∂L/∂weights  # Softmax gradient (복잡!)
∂L/∂Q = (∂L/∂scores @ K) / √d_k               # Query gradient
∂L/∂K = (∂L/∂scores^T @ Q) / √d_k             # Key gradient

# 핵심 포인트: Softmax의 그래디언트
∂softmax_i/∂x_j = softmax_i(δ_ij - softmax_j)
# δ_ij: Kronecker delta (i==j이면 1, 아니면 0)

이것이 Attention이 안정적으로 학습되는 이유:
1. 그래디언트가 모든 위치로 골고루 전파됨 (vanishing gradient 해결)
2. Skip connection 없이도 깊은 네트워크 학습 가능
```

### Multi-Head Attention의 그래디언트 흐름

```python
# Multi-Head는 8개의 병렬 그래디언트 경로 생성
∂L/∂input = Σ(i=1 to 8) ∂L/∂head_i ⊙ ∂head_i/∂input

장점:
1. 그래디언트 다양성: 8가지 다른 업데이트 신호
2. 안정성: 일부 head가 saturate되어도 다른 head가 보상  
3. 특화: 각 head가 다른 관점의 그래디언트 신호 제공

결과: Single-head보다 훨씬 안정적이고 빠른 수렴
```

### Positional Encoding의 그래디언트 특성

```python
# Positional Encoding은 학습되지 않음 (Fixed)
∂L/∂PE = 0  # 그래디언트 없음

하지만 입력 임베딩으로는 그래디언트 전파:
∂L/∂word_embedding = ∂L/∂(word_embedding + PE) = ∂L/∂final_embedding

효과:
1. 위치 정보는 고정되어 안정적 학습
2. 단어 임베딩은 위치를 고려한 의미로 업데이트
3. 위치-의미 결합 표현 학습

예시:
"love"의 임베딩이 위치에 따라 다르게 학습:
- "I love you"에서 동사 역할 강조
- "love is good"에서 명사 역할 강조
```

## 💡 핵심 통찰

### 1. Attention = Soft Dictionary Lookup
```python
# 전통적 딕셔너리: 정확한 키 매칭
dict["love"] = value_love  # 정확히 일치해야만 값 반환

# Attention: 유사도 기반 소프트 매칭  
attention["love"] = 0.1*value_I + 0.5*value_love + 0.4*value_you
# 모든 키와의 유사도로 가중평균된 값 반환
```

### 2. Self-Attention = 문맥적 단어 표현
```python
# 기존: 고정된 단어 임베딩
"love" → [0.1, 0.2, 0.3, ...]  # 항상 동일

# Self-Attention 후: 문맥 의존적 표현
"I love you" → "love" = [0.5, 0.6, 0.7, ...]      # 동사적 의미 강화
"love is good" → "love" = [0.2, 0.8, 0.4, ...]    # 명사적 의미 강화
```

### 3. Multi-Head = 다면적 이해
```python
# 인간의 언어 이해와 유사
문장: "The bank can guarantee deposits will eventually cover future tuition costs"

Head1: "bank" → "deposits" (금융 기관)
Head2: "bank" → "river" (지리적 의미) X
Head3: "guarantee" → "can" (문법적 보조동사)  
Head4: "deposits" → "costs" (경제적 관계)
...

결합: 문맥에 맞는 올바른 해석 = 금융 은행
```

이 수학적 기초를 바탕으로 실제 구현에서 어떻게 코드로 구현하는지 `implementation_guide.md`에서 확인해보세요! 🚀