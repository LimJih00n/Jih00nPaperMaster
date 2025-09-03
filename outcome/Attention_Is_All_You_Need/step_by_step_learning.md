# Transformer 수학 마스터하기: 8주 완성 가이드

## 🎯 목표
Transformer를 완벽히 이해하고 구현할 수 있는 수학적 기초 완성

---

## 📅 Week 1-2: 선형대수 기초

### Day 1-3: 벡터의 이해
```python
# 실습 1: 벡터 연산
import numpy as np

# 단어 임베딩 시뮬레이션
word_embeddings = {
    "I": [0.2, 0.5, -0.1, 0.8],
    "love": [0.7, -0.3, 0.4, 0.2],
    "AI": [0.9, 0.1, 0.6, -0.4]
}

# 벡터 덧셈 (문맥 결합)
context = np.array(word_embeddings["I"]) + np.array(word_embeddings["love"])
print(f"Combined context: {context}")

# 벡터 크기 (의미의 강도)
magnitude = np.linalg.norm(word_embeddings["AI"])
print(f"Magnitude of 'AI': {magnitude}")

# 연습 문제:
# 1. 두 단어의 코사인 유사도 계산하기
# 2. 3개 단어의 평균 벡터 구하기
```

### Day 4-7: 행렬 연산
```python
# 실습 2: 행렬 곱셈 이해하기
# 문장을 행렬로 표현
sentence = np.array([
    [0.2, 0.5, -0.1],  # "I"
    [0.7, -0.3, 0.4],  # "love"
    [0.9, 0.1, 0.6]    # "AI"
])

# 가중치 행렬 (변환)
W = np.array([
    [1.0, 0.5, 0.0],
    [0.0, 1.0, 0.5],
    [0.5, 0.0, 1.0]
])

# 변환된 표현
transformed = np.matmul(sentence, W)
print("Original shape:", sentence.shape)
print("Transformed:", transformed)

# 핵심 이해:
# - 행렬 곱셈 = 선형 변환
# - Transformer의 W_Q, W_K, W_V가 이런 역할
```

### Day 8-14: 내적과 전치
```python
# 실습 3: Attention Score의 기초
# Query와 Key의 내적 = 유사도

queries = np.array([[1, 0, 1], [0, 1, 0]])  # 2개 쿼리
keys = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1]])  # 3개 키

# 모든 쿼리-키 쌍의 점수 계산
scores = np.matmul(queries, keys.T)  # (2, 3) 행렬
print("Attention scores:")
print(scores)
print("\n해석:")
print("Query 1은 Key 1, 3과 유사")
print("Query 2는 Key 2와 유사")

# 실전 연습:
def compute_attention_scores(Q, K):
    """Q와 K로부터 attention score 계산"""
    scores = np.matmul(Q, K.T)
    return scores / np.sqrt(K.shape[-1])  # scaling

# 테스트
Q = np.random.randn(5, 4)  # 5개 쿼리, 4차원
K = np.random.randn(5, 4)  # 5개 키, 4차원
scores = compute_attention_scores(Q, K)
print(f"Score matrix shape: {scores.shape}")  # (5, 5)
```

---

## 📅 Week 3-4: 확률과 활성화 함수

### Day 15-18: Softmax 완벽 이해
```python
# 실습 4: Softmax 구현과 이해
def softmax(x, axis=-1):
    """Numerically stable softmax"""
    # Trick: 최댓값을 빼서 overflow 방지
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 예제: Attention 가중치
scores = np.array([
    [5.0, 2.0, 0.1],  # 첫 단어의 attention scores
    [1.0, 4.0, 2.0]   # 둘째 단어의 attention scores
])

weights = softmax(scores)
print("Attention weights:")
print(weights)
print("\n해석:")
print(f"첫 단어: {weights[0]*100}% 주목도")
print(f"둘째 단어: {weights[1]*100}% 주목도")

# Temperature 실험
def softmax_with_temp(x, temperature=1.0):
    """Temperature로 분포 조절"""
    return softmax(x / temperature)

# Temperature 효과 비교
x = np.array([1.0, 2.0, 3.0])
for temp in [0.5, 1.0, 2.0]:
    result = softmax_with_temp(x, temp)
    print(f"Temp={temp}: {result}")
    print(f"  최대값 비율: {max(result):.2%}")
```

### Day 19-21: Cross-Entropy 이해
```python
# 실습 5: Loss 계산
def cross_entropy(y_true, y_pred, epsilon=1e-7):
    """Cross-entropy loss 계산"""
    # 작은 값 더해서 log(0) 방지
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# 예제: 다음 단어 예측
vocab = ["the", "cat", "sat", "on", "mat"]
true_next = [0, 1, 0, 0, 0]  # 정답: "cat"

# 모델 예측 (좋은 경우)
good_pred = [0.1, 0.7, 0.05, 0.1, 0.05]
loss_good = cross_entropy(true_next, good_pred)
print(f"Good prediction loss: {loss_good:.3f}")

# 모델 예측 (나쁜 경우)
bad_pred = [0.4, 0.1, 0.2, 0.2, 0.1]
loss_bad = cross_entropy(true_next, bad_pred)
print(f"Bad prediction loss: {loss_bad:.3f}")

# Perplexity 계산
print(f"Good model perplexity: {np.exp(loss_good):.2f}")
print(f"Bad model perplexity: {np.exp(loss_bad):.2f}")
```

### Day 22-28: 정보이론 기초
```python
# 실습 6: Entropy와 KL Divergence
def entropy(p):
    """확률 분포의 엔트로피"""
    return -np.sum(p * np.log(p + 1e-10))

def kl_divergence(p, q):
    """KL divergence between p and q"""
    return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))

# 예제: Attention 분포 비교
uniform_attention = np.array([0.33, 0.33, 0.34])
focused_attention = np.array([0.8, 0.15, 0.05])

print(f"Uniform entropy: {entropy(uniform_attention):.3f}")
print(f"Focused entropy: {entropy(focused_attention):.3f}")
print("\n낮은 엔트로피 = 더 집중된 attention")

# KL divergence로 분포 차이 측정
kl = kl_divergence(focused_attention, uniform_attention)
print(f"KL(focused||uniform): {kl:.3f}")
```

---

## 📅 Week 5-6: Attention 메커니즘 구현

### Day 29-35: Scaled Dot-Product Attention
```python
# 실습 7: 완전한 Attention 구현
class ScaledDotProductAttention:
    def __init__(self, d_k):
        self.d_k = d_k
        self.scale = np.sqrt(d_k)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        """
        # Step 1: Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / self.scale
        
        # Step 2: Apply mask (optional)
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Step 3: Apply softmax
        attention_weights = softmax(scores, axis=-1)
        
        # Step 4: Weighted sum of values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights

# 테스트
batch_size, seq_len, d_k = 2, 4, 8
Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_k)

attention = ScaledDotProductAttention(d_k)
output, weights = attention.forward(Q, K, V)

print(f"Input shape: {Q.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nAttention pattern for first item:")
print(weights[0])
```

### Day 36-42: Multi-Head Attention
```python
# 실습 8: Multi-Head Attention 구현
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 가중치 초기화
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = np.matmul(x, self.W_Q)
        K = np.matmul(x, self.W_K)
        V = np.matmul(x, self.W_V)
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Apply attention to each head
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attention_weights = softmax(scores, axis=-1)
        context = np.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = np.matmul(context, self.W_O)
        
        return output, attention_weights

# 테스트
d_model, n_heads = 512, 8
batch_size, seq_len = 2, 10

x = np.random.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model, n_heads)
output, weights = mha.forward(x)

print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
print(f"Weights per head: {weights.shape}")
print(f"\n각 헤드는 {d_model//n_heads}차원 공간에서 attention 계산")
```

---

## 📅 Week 7: Positional Encoding과 최적화

### Day 43-46: Positional Encoding
```python
# 실습 9: Positional Encoding 마스터
class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.pe = self._create_pe_matrix(max_len, d_model)
    
    def _create_pe_matrix(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len).reshape(-1, 1)
        
        # 주파수 계산
        div_term = np.exp(np.arange(0, d_model, 2) * 
                         -(np.log(10000.0) / d_model))
        
        # Sin/Cos 적용
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def encode(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]
    
    def visualize(self, length=100, dims=128):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        plt.imshow(self.pe[:length, :dims].T, cmap='RdBu', aspect='auto')
        plt.colorbar()
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title('Positional Encoding Pattern')
        plt.show()
        
        # 특정 차원의 패턴
        plt.figure(figsize=(12, 3))
        for i in range(0, 8, 2):
            plt.plot(self.pe[:100, i], label=f'dim {i}')
        plt.legend()
        plt.xlabel('Position')
        plt.ylabel('Encoding Value')
        plt.title('Sinusoidal Patterns')
        plt.show()

# 사용 예제
pe = PositionalEncoding(d_model=512)
pe.visualize()

# 상대 위치 테스트
pos_5 = pe.pe[5]
pos_10 = pe.pe[10]
pos_15 = pe.pe[15]

# pos_15는 pos_5 + pos_10과 관련있음
print("위치 인코딩의 선형 관계 확인")
```

### Day 47-49: Learning Rate Scheduling
```python
# 실습 10: Warmup Schedule 구현
class TransformerLRSchedule:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        
    def get_lr(self, step):
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return self.d_model ** (-0.5) * min(arg1, arg2)
    
    def visualize(self, max_steps=20000):
        import matplotlib.pyplot as plt
        
        steps = np.arange(1, max_steps)
        lrs = [self.get_lr(step) for step in steps]
        
        plt.figure(figsize=(10, 4))
        plt.plot(steps, lrs)
        plt.axvline(x=self.warmup_steps, color='r', linestyle='--', 
                   label=f'Warmup ends: {self.warmup_steps}')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Transformer Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"Peak LR: {max(lrs):.6f}")
        print(f"At step: {steps[np.argmax(lrs)]}")

# 시각화
scheduler = TransformerLRSchedule(d_model=512, warmup_steps=4000)
scheduler.visualize()
```

---

## 📅 Week 8: 전체 통합과 최적화

### Day 50-53: 미니 Transformer 구현
```python
# 실습 11: 완전한 Transformer 블록
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Feed-forward network
        self.ff_w1 = np.random.randn(d_model, d_ff) * 0.1
        self.ff_w2 = np.random.randn(d_ff, d_model) * 0.1
        
        self.dropout = dropout
    
    def feed_forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        hidden = np.maximum(0, np.matmul(x, self.ff_w1))  # ReLU
        return np.matmul(hidden, self.ff_w2)
    
    def forward(self, x):
        # Multi-head attention with residual
        attn_output, _ = self.attention.forward(x)
        x = self.norm1(x + self.dropout_layer(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout_layer(ff_output))
        
        return x
    
    def dropout_layer(self, x):
        if self.dropout > 0:
            mask = np.random.binomial(1, 1-self.dropout, x.shape)
            return x * mask / (1-self.dropout)
        return x

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

### Day 54-56: 성능 분석과 최적화
```python
# 실습 12: 계산 복잡도 분석
def analyze_complexity(seq_len, d_model, n_heads):
    """Transformer의 계산 복잡도 분석"""
    
    # Self-attention 복잡도
    attention_flops = seq_len * seq_len * d_model  # QK^T
    attention_flops += seq_len * seq_len * d_model  # Softmax @ V
    
    # Multi-head projection
    projection_flops = 4 * seq_len * d_model * d_model  # Q,K,V,O projections
    
    # FFN 복잡도
    d_ff = 4 * d_model  # 일반적으로 4배
    ffn_flops = 2 * seq_len * d_model * d_ff
    
    total_flops = attention_flops + projection_flops + ffn_flops
    
    print(f"Sequence Length: {seq_len}")
    print(f"Model Dimension: {d_model}")
    print(f"Number of Heads: {n_heads}")
    print("-" * 40)
    print(f"Attention FLOPs: {attention_flops:,}")
    print(f"Projection FLOPs: {projection_flops:,}")
    print(f"FFN FLOPs: {ffn_flops:,}")
    print(f"Total FLOPs: {total_flops:,}")
    print("-" * 40)
    print(f"Memory (attention): O({seq_len}²×{d_model})")
    print(f"Time complexity: O({seq_len}²×{d_model})")
    
    return total_flops

# 다양한 설정 비교
configs = [
    (512, 768, 12),   # BERT-base
    (2048, 768, 12),  # 긴 시퀀스
    (512, 1024, 16),  # 큰 모델
]

for seq_len, d_model, n_heads in configs:
    flops = analyze_complexity(seq_len, d_model, n_heads)
    print(f"\n메모리 사용량 (대략): {flops * 4 / 1e9:.2f} GB")
    print("=" * 50)
```

---

## 🎓 최종 프로젝트

### 프로젝트 1: 작은 번역 모델
```python
# 숫자 시퀀스를 역순으로 변환하는 Transformer
class ToyTransformer:
    def __init__(self, vocab_size=10, d_model=64, n_heads=4):
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.pe = PositionalEncoding(d_model)
        self.encoder = TransformerBlock(d_model, n_heads, d_model*4)
        self.decoder = TransformerBlock(d_model, n_heads, d_model*4)
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.1
    
    def forward(self, src, tgt):
        # Embedding + Positional encoding
        src_emb = self.pe.encode(self.embedding[src])
        tgt_emb = self.pe.encode(self.embedding[tgt])
        
        # Encode
        enc_output = self.encoder.forward(src_emb)
        
        # Decode (simplified - real decoder needs cross-attention)
        dec_output = self.decoder.forward(tgt_emb)
        
        # Project to vocabulary
        logits = np.matmul(dec_output, self.output_projection)
        
        return softmax(logits, axis=-1)

# 테스트
model = ToyTransformer()
src = np.array([[1, 2, 3, 4, 5]])  # 입력: [1,2,3,4,5]
tgt = np.array([[5, 4, 3, 2, 1]])  # 목표: [5,4,3,2,1]

output = model.forward(src, tgt)
print(f"Output shape: {output.shape}")
print(f"Output probabilities:\n{output[0]}")
```

### 프로젝트 2: Attention 시각화 도구
```python
def visualize_attention(attention_weights, tokens):
    """Attention 패턴을 히트맵으로 시각화"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Multi-Head Attention Patterns')
    
    for head in range(8):
        ax = axes[head // 4, head % 4]
        im = ax.imshow(attention_weights[head], cmap='Blues')
        ax.set_title(f'Head {head+1}')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_yticklabels(tokens)
        
    plt.tight_layout()
    plt.show()

# 예제 사용
tokens = ["The", "cat", "sat", "on", "the", "mat"]
# 가상의 attention weights (6x6x8)
attention = np.random.softmax(np.random.randn(8, 6, 6), axis=-1)
visualize_attention(attention, tokens)
```

---

## 📊 학습 체크리스트

### 필수 이해 항목
- [ ] 벡터 내적이 유사도를 나타내는 이유
- [ ] 행렬 곱셈이 선형 변환인 이유
- [ ] Softmax가 확률 분포를 만드는 원리
- [ ] Scaled dot-product에서 √d_k로 나누는 이유
- [ ] Multi-head가 단일 head보다 나은 이유
- [ ] Positional encoding이 필요한 이유
- [ ] Residual connection의 역할
- [ ] Layer normalization의 효과

### 구현 능력
- [ ] NumPy로 attention 구현
- [ ] Multi-head attention 구현
- [ ] Positional encoding 생성
- [ ] Learning rate schedule 구현
- [ ] 간단한 Transformer 블록 구성

### 심화 이해
- [ ] O(n²) 복잡도 문제와 해결책
- [ ] Gradient vanishing/exploding 이해
- [ ] Attention의 해석가능성
- [ ] 다양한 positional encoding 방식
- [ ] Transformer 변형들 (BERT, GPT 등)

---

## 🚀 다음 단계

1. **PyTorch/TensorFlow 구현**: NumPy 코드를 딥러닝 프레임워크로 전환
2. **실제 데이터 적용**: 작은 번역 데이터셋으로 학습
3. **최신 논문 읽기**: Flash Attention, Linear Attention 등
4. **프로젝트 진행**: 자신만의 Transformer 변형 개발
5. **오픈소스 기여**: Hugging Face Transformers 등에 기여

## 💡 마지막 조언

> "수학은 한 번에 이해되지 않습니다. 반복하고, 구현하고, 시각화하세요.
> 코드로 직접 만져보면서 이해하는 것이 가장 빠른 길입니다."

작은 예제부터 시작해서 점진적으로 복잡도를 높여가세요. 
매일 30분씩만 투자해도 8주 후에는 Transformer를 완벽히 이해할 수 있습니다!