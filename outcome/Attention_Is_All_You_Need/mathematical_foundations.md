# Transformer 이해를 위한 수학적 기초 - 완벽 가이드

## 📚 목차
1. [선형대수 기초](#1-선형대수-기초)
2. [확률론과 정보이론](#2-확률론과-정보이론)
3. [Attention 메커니즘의 수학](#3-attention-메커니즘의-수학)
4. [최적화 이론](#4-최적화-이론)
5. [실습 예제](#5-실습-예제)

---

## 1. 선형대수 기초

### 1.1 벡터와 행렬 기본

#### 벡터 (Vector)
벡터는 숫자들의 순서있는 배열입니다.

```
단어 "cat"의 임베딩 벡터 예시:
v = [0.2, -0.5, 0.8, 0.1]  (4차원 벡터)

실제 Transformer에서는:
v ∈ ℝ^512 또는 ℝ^768 (512 또는 768차원)
```

#### 행렬 (Matrix)
행렬은 벡터들의 집합으로 볼 수 있습니다.

```
문장 "I love cats"의 임베딩 행렬:
     [0.2  -0.5   0.8]  ← "I"의 임베딩
X =  [0.3   0.7  -0.2]  ← "love"의 임베딩
     [0.1  -0.4   0.9]  ← "cats"의 임베딩

크기: 3×3 (3개 단어, 3차원 임베딩)
```

### 1.2 핵심 연산

#### 행렬 곱셈 (Matrix Multiplication)
**이것이 Transformer의 핵심입니다!**

```
A × B = C

예시:
[1 2]   [5 6]   [1×5+2×7  1×6+2×8]   [19 22]
[3 4] × [7 8] = [3×5+4×7  3×6+4×8] = [43 50]

Transformer에서의 의미:
- Query × Key^T = Attention Score
- 각 단어가 다른 단어와 얼마나 관련있는지 계산
```

#### 전치 행렬 (Transpose)
행과 열을 바꾸는 연산입니다.

```
     [1 2 3]        [1 4]
A =  [4 5 6]   A^T = [2 5]
                     [3 6]

Transformer에서:
Key^T를 만들어 Query와 곱하기 위해 사용
```

#### 내적 (Dot Product)
두 벡터 간의 유사도를 측정합니다.

```
v1 · v2 = v1[0]×v2[0] + v1[1]×v2[1] + ...

예시:
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32

의미:
- 값이 클수록 두 벡터가 유사
- Attention Score 계산의 기초
```

### 1.3 차원과 크기

```
Transformer의 주요 차원:
- d_model = 512: 모델의 히든 차원
- d_k = 64: Query/Key 벡터 차원
- d_v = 64: Value 벡터 차원
- h = 8: Attention 헤드 수

관계: d_model = h × d_k
      512 = 8 × 64
```

---

## 2. 확률론과 정보이론

### 2.1 Softmax 함수
**Attention 가중치를 만드는 핵심 함수**

```
softmax(x_i) = exp(x_i) / Σ(exp(x_j))

예시:
입력: [2.0, 1.0, 0.1]
계산:
- exp(2.0) = 7.39
- exp(1.0) = 2.72  
- exp(0.1) = 1.11
- 합계 = 11.22

출력: [0.66, 0.24, 0.10]  (확률로 변환됨)

의미:
- 모든 값의 합 = 1.0
- 큰 값은 더 크게, 작은 값은 더 작게 만듦
- Attention에서 "주목할 단어" 결정
```

### 2.2 Cross-Entropy Loss
모델 학습의 목표 함수입니다.

```
L = -Σ(y_true × log(y_pred))

예시:
정답: "cat" (원-핫 벡터: [0, 1, 0])
예측: [0.2, 0.7, 0.1]

Loss = -(0×log(0.2) + 1×log(0.7) + 0×log(0.1))
     = -log(0.7) = 0.36

의미:
- 예측이 정확할수록 Loss가 작음
- 완벽한 예측: Loss = 0
```

### 2.3 확률 분포

```
언어 모델의 출력:
P(다음 단어 = "cat") = 0.3
P(다음 단어 = "dog") = 0.2
P(다음 단어 = "bird") = 0.1
...
합계 = 1.0

Perplexity (혼란도):
PPL = exp(평균 Loss)
낮을수록 좋은 모델
```

---

## 3. Attention 메커니즘의 수학

### 3.1 Scaled Dot-Product Attention
**Transformer의 핵심 공식**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

#### 단계별 분해:

**Step 1: Query, Key, Value 생성**
```
입력 X가 주어졌을 때:
Q = X × W_Q  (Query 행렬)
K = X × W_K  (Key 행렬)
V = X × W_V  (Value 행렬)

예시 (3개 단어, 4차원):
X = [[x1], [x2], [x3]]  (3×4)
W_Q, W_K, W_V는 학습 가능한 가중치 (4×4)
```

**Step 2: Attention Score 계산**
```
Scores = Q × K^T

예시:
     K1  K2  K3
Q1 [ 10   5   2]  ← Q1이 각 K와 얼마나 관련있는지
Q2 [  3  12   4]
Q3 [  1   4  15]

의미: Q1은 K1과 가장 관련 높음 (10)
```

**Step 3: 스케일링**
```
Scaled_Scores = Scores / √d_k

왜 √d_k로 나누나?
- d_k가 클수록 내적 값이 커짐
- Softmax가 포화되는 것을 방지
- Gradient vanishing 방지

예시 (d_k = 64):
Scores / √64 = Scores / 8
```

**Step 4: Softmax 적용**
```
Attention_Weights = softmax(Scaled_Scores)

     K1    K2    K3
Q1 [0.85  0.10  0.05]  ← Q1은 K1에 85% 주목
Q2 [0.15  0.70  0.15]  ← Q2는 K2에 70% 주목
Q3 [0.05  0.20  0.75]  ← Q3는 K3에 75% 주목
```

**Step 5: Value 가중합**
```
Output = Attention_Weights × V

각 Query에 대해:
Output_1 = 0.85×V1 + 0.10×V2 + 0.05×V3
(V1의 정보를 85% 반영)
```

### 3.2 Multi-Head Attention
**여러 관점에서 동시에 주목**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

각 헤드:
head_i = Attention(Q×W_Q^i, K×W_K^i, V×W_V^i)

8개 헤드 예시:
- Head 1: 문법적 관계 학습
- Head 2: 의미적 유사성 학습
- Head 3: 위치 관계 학습
- ...
```

### 3.3 Positional Encoding
**순서 정보 추가**

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

예시 (pos=0, d_model=4):
PE[0] = [sin(0/1), cos(0/1), sin(0/100), cos(0/100)]
      = [0, 1, 0, 1]

pos=1:
PE[1] = [sin(1/1), cos(1/1), sin(1/100), cos(1/100)]
      = [0.84, 0.54, 0.01, 0.99]

특징:
- 각 위치마다 고유한 패턴
- 상대 위치 계산 가능
- 학습 없이 임의 길이 처리
```

---

## 4. 최적화 이론

### 4.1 Adam Optimizer

```
Adam 업데이트 규칙:
m_t = β1 × m_(t-1) + (1-β1) × gradient
v_t = β2 × v_(t-1) + (1-β2) × gradient²
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)
θ_t = θ_(t-1) - α × m̂_t / (√v̂_t + ε)

Transformer 설정:
- β1 = 0.9 (momentum)
- β2 = 0.98 (RMSprop)
- ε = 10^-9 (numerical stability)
```

### 4.2 Learning Rate Schedule

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))

Warmup 예시 (warmup=4000):
- Step 1000: lr 증가 중
- Step 4000: lr 최대
- Step 8000: lr 감소 시작

이유:
- 초기: 작은 lr로 안정적 시작
- 중간: 빠른 학습
- 후기: 미세 조정
```

### 4.3 Gradient Clipping

```
if ||gradient|| > threshold:
    gradient = gradient × (threshold / ||gradient||)

예시:
- Gradient norm = 15
- Threshold = 1.0
- Clipped gradient = gradient × (1.0/15)

효과: Gradient explosion 방지
```

---

## 5. 실습 예제

### 예제 1: 간단한 Attention 계산

```python
import numpy as np

# 3개 단어, 4차원 임베딩
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 0, 0]])

K = np.array([[1, 1, 0, 0],
              [0, 0, 1, 1],
              [1, 0, 1, 0]])

V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

# Step 1: QK^T 계산
scores = np.matmul(Q, K.T)
print("Attention Scores:")
print(scores)
# [[2, 1, 2],
#  [1, 2, 1],
#  [2, 0, 1]]

# Step 2: 스케일링
d_k = 4
scaled_scores = scores / np.sqrt(d_k)

# Step 3: Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)
print("\nAttention Weights:")
print(attention_weights)

# Step 4: Value 가중합
output = np.matmul(attention_weights, V)
print("\nOutput:")
print(output)
```

### 예제 2: Positional Encoding 시각화

```python
def get_positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (i/d_model)))
    
    return PE

# 10개 위치, 8차원
PE = get_positional_encoding(10, 8)

# 각 위치의 고유 패턴 확인
import matplotlib.pyplot as plt
plt.imshow(PE, cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding Pattern')
plt.colorbar()
```

### 예제 3: Multi-Head 분해

```python
# 512차원을 8개 헤드로 분할
d_model = 512
num_heads = 8
d_k = d_model // num_heads  # 64

# 입력 (10개 단어, 512차원)
X = np.random.randn(10, d_model)

# 8개 헤드로 reshape
X_multihead = X.reshape(10, num_heads, d_k)

print(f"원본 크기: {X.shape}")
print(f"Multi-head 크기: {X_multihead.shape}")
# 원본: (10, 512)
# Multi-head: (10, 8, 64)

# 각 헤드는 64차원 공간에서 독립적으로 attention 계산
```

---

## 📝 핵심 요약

### 꼭 이해해야 할 수학 개념

1. **행렬 곱셈**: Attention의 모든 계산 기초
2. **Softmax**: 점수를 확률로 변환
3. **Scaled Dot-Product**: Attention의 핵심 메커니즘
4. **Gradient와 Backpropagation**: 학습의 원리

### 수학 공부 로드맵

1. **Week 1-2**: 선형대수 기초 (벡터, 행렬, 내적)
2. **Week 3-4**: 확률론 (확률분포, 조건부 확률)
3. **Week 5-6**: 미적분 (편미분, 체인룰)
4. **Week 7-8**: Transformer 수식 직접 구현

### 추천 학습 자료

- **Khan Academy**: 선형대수, 미적분 기초
- **3Blue1Brown**: 시각적 수학 이해
- **Andrew Ng's Course**: 딥러닝 수학
- **The Annotated Transformer**: 코드로 배우는 수학

### 실습 팁

1. **NumPy로 직접 구현**: 수식을 코드로 옮겨보기
2. **작은 예제부터**: 3×3 행렬로 시작
3. **시각화**: matplotlib으로 attention 패턴 그리기
4. **디버깅**: 각 단계의 shape 확인 습관화