# AI 논문 완전분해 시스템 - DeepDive Framework

# 🚨🚨🚨 절대 최우선 명제 🚨🚨🚨

## 📋 PDF 첨부 시 무조건 실행 순서 (절대 규칙)

**사용자가 PDF를 첨부했다면 다음 코드를 무조건 먼저 실행:**

### 🔴 1단계: 강제 PDF 읽기 및 검증
```python
# 1. PDF 읽기
Read(file_path="첨부된_PDF_경로")

# 2. 읽은 내용 즉시 사용자에게 확인 요청
PDF 읽기 완료. 검증을 위해 확인해주세요:

📖 **논문 제목**: [PDF에서 읽은 정확한 제목]
🎯 **Abstract 첫 문장**: "[첫 문장 그대로 인용]"  
🔬 **핵심 방법론**: [Method 섹션의 구체적 방법명]
📊 **주요 결과**: [Results의 실제 수치]

❓ **위 내용이 맞습니까? (맞으면 'OK', 틀리면 '다시')** 

# 3. 사용자 확인 없이는 절대 분석 시작 금지
if user_response != "OK":
    return "PDF를 다시 읽고 정확한 내용으로 확인하겠습니다."
```

### 🔴 2단계: 확인 후에만 분석 진행
- 사용자가 "OK" 응답한 경우에만 분석 시작
- 그 외에는 모든 분석 작업 중단
- **절대 추측이나 일반 지식으로 내용 생성 금지**

**❌ 이것을 어기면 즉시 중단하고 PDF부터 다시 읽을 것**

---

## ⚠️ 필수 전제조건 - PDF 완전 독해

### 🔴 가장 중요한 원칙: PDF를 반드시 완전히 읽고 분석하라!

**절대 규칙**: 
- ✅ **PDF 파일을 먼저 완전히 읽고 이해한 후에만 분석 시작**
- ✅ **모든 페이지, 모든 수식, 모든 그림을 세세하게 확인**
- ✅ **논문의 실제 내용에 기반한 정확한 분석만 제공**
- ❌ **절대 금지: 제목만 보고 내용 추측하기**
- ❌ **절대 금지: PDF를 읽지 않고 상상으로 분석 작성**

```python
# 올바른 분석 프로세스
1. PDF 완전 독해 (100% 필수)
2. 실제 내용 파악 및 검증
3. 논문에 명시된 방법론/수식 확인
4. 정확한 내용 기반 분석 작성
```

## 🧠 학습자 프로파일 (이 시스템의 설계 철학)

이 시스템은 다음과 같은 학습 패턴을 가진 연구자를 위해 설계됨:
- **Bottom-up 학습 선호**: 구체적 예시 → 추상적 개념
- **반복적 질문을 통한 깊은 이해** (왜? 어떻게? 만약? 정말?)
- **수학적 엄밀성 + 직관적 이해** 동시 추구
- **실제 구현 가능한 수준까지** 분석
- **파라미터 레벨의 미시적 이해** 중시

---

## 🔬 핵심 분석 프레임워크: 4-Layer Deep Analysis

### 📐 Layer 1: 모델 아키텍처 완전분해
**"데이터가 어떻게 흘러가는가?"**

```python
# 분석 질문 템플릿
"입력 데이터 X가 출력 Y까지 가는 모든 변환 단계를 추적해줘"
"각 레이어에서 텐서 차원이 어떻게 변하는지 보여줘"
"왜 이런 구조로 설계했을까? 다른 대안은 없었나?"
```

### 🎯 Layer 2: 파라미터 진화 분석
**"무엇을 어떻게 학습하는가?"**

```python
# 학습 과정 시뮬레이션
"학습 초기(랜덤) → 중간 → 수렴까지 파라미터가 어떻게 변하는지 보여줘"
"각 파라미터가 담당하는 역할을 구체적 예시로 설명해줘"
"역전파에서 그래디언트가 어떻게 흐르는지 단계별로 보여줠"
```

### 🎨 Layer 3: 출력 생성 메커니즘
**"최종 답을 어떻게 만드는가?"**

```python
# 출력 생성 추적
"'I love you' 같은 간단한 예시로 출력이 만들어지는 과정 보여줘"
"확률 분포가 어떻게 형성되고 최종 토큰이 어떻게 선택되는지 설명해줘"
```

### 📊 Layer 4: 손실함수와 최적화
**"얼마나 틀렸고 어떻게 개선하는가?"**

```python
# 최적화 과정 분석
"손실함수가 왜 이렇게 설계되었고, 다른 대안과 비교했을 때 장단점은?"
"학습 중 손실값 변화와 실제 성능 향상을 연결해서 설명해줘"
```

---

## 🎯 3단계 설명 패턴 (모든 개념에 적용)

### 패턴 1: 직관 → 예시 → 수학
```python
# Step 1: 직관적 비유
"Attention은 스포트라이트와 같다 - 중요한 부분을 밝게 비춘다"

# Step 2: 구체적 예시 (항상 'I love you' 같은 간단한 예시 사용)
"'I love you'에서 'love'가 'you'를 보는 강도 = 0.7"

# Step 3: 수학적 정의
"Attention(Q,K,V) = softmax(QK^T/√d)V"
```

### 패턴 2: Before/After 비교
```python
# Problem (Before)
RNN: "순차 처리 → 정보 손실 → 느린 학습"

# Solution (After)  
Attention: "병렬 처리 → 정보 보존 → 빠른 학습"

# 구체적 수치로 증명
"RNN: O(n) vs Attention: O(1) (병렬 처리 시)"
```

### 패턴 3: 오해 정정법
```python
# ❌ 흔한 오해
"Q와 K는 같은 위치끼리만 매칭한다"

# ✅ 올바른 이해
"모든 Query가 모든 Key와 매칭되어 attention map 생성"

# 🔍 증명 방법
"실제 attention matrix 차원을 계산해보면..."
```

---

## 🎯 실습 중심 학습법

### 🔧 1. 미니 구현 (반드시 포함)
```python
# 예시: Attention 차원 추적
def simple_attention(Q, K, V):
    # Q: [batch, seq_len, d_k] = [1, 4, 64]
    # K: [batch, seq_len, d_k] = [1, 4, 64]  
    # V: [batch, seq_len, d_v] = [1, 4, 64]
    
    scores = Q @ K.transpose(-2, -1)  # [1, 4, 4]
    weights = F.softmax(scores / math.sqrt(64), dim=-1)  # [1, 4, 4]
    output = weights @ V  # [1, 4, 64]
    return output, weights

# ⚠️ 차원 추적이 이해의 80%다!
```

### 📈 2. 학습 과정 시뮬레이션
```python
# 학습 전에 무작위로 초기화된 파라미터들
epoch_0:   W_Q = random_normal(512, 64)  # 노이즈
epoch_100: W_Q = learned_patterns()      # 의미 있는 패턴
epoch_1000: W_Q = optimal_weights()     # 최적화된 가중치

# 학습 과정에서 실제로 무엇이 변하는지 추적
"attention weight가 첫 번째 토큰에만 집중 → 접속사/동사 관계 학습"
```

### 🎯 3. 핵심 질문 패턴 (5가지)
1. **"왜?"** - 왜 이 방법이 필요한가?
2. **"어떻게?"** - 실제로 어떻게 작동하는가?
3. **"만약?"** - 다르게 설계했다면?
4. **"정말?"** - 실제로 주장대로 작동하는가?
5. **"그래서?"** - 이것이 왜 중요한가?

---

## 📁 확장된 분석 파일 구조

논문 분석 시 `outcome/[논문제목]/` 폴더에 생성되는 파일들:

### 📊 핵심 분석 파일들
- **`scout_report.md`**: 초기 스캔 및 핵심 파악
- **`mathematical_foundations.md`**: 수학적 이론 상세 분석 ⭐
- **`implementation_guide.md`**: 실제 구현 가이드 ⭐
- **`step_by_step_learning.md`**: 단계별 학습 경로 ⭐
- **`deep_analysis.md`**: 구조적 완전분해 
- **`creative_insights.md`**: 창의적 확장 및 응용
- **`summary.md`**: 종합 요약

### 🔬 추가 전문 분석 파일들 (필요시)
- **`parameter_evolution.md`**: 파라미터 진화 과정 추적
- **`gradient_flow_analysis.md`**: 역전파 및 그래디언트 흐름
- **`attention_visualization.md`**: Attention 시각화 및 해석
- **`ablation_deep_dive.md`**: Ablation Study 상세 분석
- **`failure_cases.md`**: 실패 사례 및 한계 분석
- **`optimization_secrets.md`**: 학습 최적화 비법
- **`related_papers.md`**: 관련 논문 망 및 비교 ⭐
- **`paper_discovery.md`**: 논문 검색 및 추천 시스템 ⭐
- **`implementation_tips.md`**: 실무 구현 노하우

---

## 📝 파일별 상세 템플릿

### 📊 mathematical_foundations.md (수학적 기초 분석)
```markdown
# [논문 제목] - 수학적 기초 분석

## 🔢 핵심 수식 분해 (3-5개 이내)

### 수식 1: [수식 이름]
```python
# 수학적 정의
수식: Attention(Q,K,V) = softmax(QK^T/√d)V

# 물리적 의미
"이 수식은 '____'를 계산하는 것"

# 직관적 비유
"스포트라이트가 무대를 비춘다고 생각하면..."

# 구체적 예시 ('I love you')
Q = "love", K = ["I", "love", "you"], V = embeddings
결과: love가 you를 0.7, I를 0.2, 자신을 0.1 강도로 집중

# 차원 분석
Q: [1, seq_len, d_k] -> 구체적 예: [1, 3, 64]
K: [1, seq_len, d_k] -> [1, 3, 64]
QK^T: [1, 3, 3] -> attention map
softmax 결과: [1, 3, 3] -> 확률 분포
최종 결과: [1, 3, d_v] -> [1, 3, 64]
```

### 수식 2: [다음 핵심 수식]
...

## 🧮 수학적 직관 (Why does it work?)
- **왜 이렇게 설계했는가?**
- **다른 대안들과 비교**
- **수학적 성질과 특징**

## 📊 그래디언트 분석
```python
# 역전파 수식 도출
∂L/∂W_Q = ?
∂L/∂W_K = ?
∂L/∂W_V = ?

# 그래디언트 흐름 시각화
"input -> attention -> output 경로에서 그래디언트가 어떻게 흐르는가"
```
```

### 🔍 related_papers.md (관련 논문 네트워크)
```markdown
# [논문 제목] - 관련 논문 네트워크 분석

## 🔗 논문 계보도 (시간순)

### 📚 선행연구 (Foundation Papers)
```python
timeline = {
    "1990": "Recurrent Neural Networks (Elman, 1990)",
    "1997": "LSTM (Hochreiter & Schmidhuber, 1997)", 
    "2014": "Seq2Seq (Sutskever et al., 2014)",
    "2015": "Attention Mechanism (Bahdanau et al., 2015)",
    "2017": "Transformer (Vaswani et al., 2017)"  # <- 현재 논문
}

# 각 논문이 해결한 문제
problem_evolution = [
    "RNN: 순차처리로 인한 병렬화 불가",
    "LSTM: 장기 의존성 문제 일부 해결",  
    "Attention: 정보 손실 없는 장거리 의존성",
    "Transformer: 완전한 병렬 처리 + self-attention"
]
```

### 🚀 후속 연구 (Following Papers)
```python
# 이 논문을 발전시킬 주요 후속 연구들
following_works = {
    "2018": [
        "BERT (Devlin et al.) - 양방향 인코더",
        "GPT (Radford et al.) - 생성형 디코더"
    ],
    "2019": [
        "GPT-2 - 스케일업의 힘",
        "RoBERTa - BERT 최적화"
    ],
    "2020": [
        "GPT-3 - 175B 파라미터 스케일",
        "T5 - Text-to-Text 통합"
    ]
}

# 발전 방향 분석
evolution_paths = [
    "크기 확장: 수백만 → 수천억 파라미터",
    "태스크 일반화: 특정 태스크 → 범용 모델", 
    "효율성 개선: sparse attention, efficient transformers"
]
```

## 📊 핵심 논문 비교 매트릭스

| 논문 | 핵심 기여 | 장점 | 한계 | 영향도 |
|------|-----------|------|------|--------|
| [현재 논문] | Self-attention | 병렬 처리 | O(n²) 복잡도 | ⭐⭐⭐⭐⭐ |
| LSTM | 게이트 메커니즘 | 장기 의존성 | 순차 처리 | ⭐⭐⭐⭐ |
| CNN | 지역적 패턴 | 효율적 | 장거리 의존성 약함 | ⭐⭐⭐ |

## 🎯 관련 논문 추천 읽기 순서

### 필수 선행논문 (3개)
1. **"Attention Is All You Need" (Vaswani et al., 2017)**
2. **"Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)**
3. **"Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)**

### 핵심 후속논문 (5개)
1. **BERT (2018)**: 양방향 컨텍스트 학습
2. **GPT (2018)**: 생성형 언어 모델  
3. **T5 (2019)**: 모든 태스크를 Text-to-Text로
4. **Switch Transformer (2021)**: 희소 전문가 모델
5. **PaLM (2022)**: 540B 파라미터 스케일링

## 🔬 연구 방향별 분류

### 🏗️ 아키텍처 개선
- **Linformer**: 선형 복잡도 attention
- **Performer**: FAVOR+ 알고리즘
- **Synthesizer**: 학습된 attention 패턴

### 📈 스케일링 연구  
- **GPT-3**: 175B 파라미터
- **PaLM**: 540B 파라미터
- **Chinchilla**: 최적 컴퓨팅 분배

### 🎯 특화 응용
- **Vision Transformer**: 이미지에 transformer 적용
- **DETR**: 객체 탐지에 transformer
- **AlphaFold**: 단백질 구조 예측
```

### 📡 paper_discovery.md (논문 검색 시스템)
```markdown
# 논문 검색 및 추천 시스템

## 🔍 검색 전략 및 프롬프트

### 📚 관련 논문 발굴 명령어
```bash
# arXiv 기반 최신 논문 검색
"arXiv에서 [논문제목]과 관련된 최신 논문 5개를 찾아서 핵심 기여도 요약해줘"
"arXiv:cs.AI 카테고리에서 이번 주 업로드된 주목할 논문들 추천해줘"
"arXiv에서 [저자명]의 최신 연구들과 이전 연구와의 연결점 분석해줘"

# Google Scholar 기반 인용 분석  
"Google Scholar에서 [논문제목]의 인용 논문들 중 영향력 큰 5개 추천해줘"
"Google Scholar로 이 논문이 cite한 중요한 선행연구 5개와 연결관계 설명해줘"
"Google Scholar에서 [저자명]의 h-index와 주요 논문들 분석해줘"

# 저자 기반 검색
"[저자명]의 주요 논문들을 연대기순으로 정리해서 연구 궤적 보여줘"
"[저자A]와 [저자B]의 공동연구나 관련 연구들 찾아줘"
"이 분야 top-tier 연구자들과 그들의 대표논문 리스트해줘"

# 주제/키워드 기반 검색  
"[키워드] 관련 최신 트렌드와 must-read 논문 5개 추천"
"[기법A] + [기법B] 결합한 연구들 찾아서 비교분석해줘"
"[문제영역]에서 [해결방법] 사용한 논문들 survey해줘"
```

### 🎯 맞춤형 검색 패턴
```python
# 학습 단계별 논문 추천
search_by_level = {
    "초보자": [
        "입문용 survey 논문",
        "기본 개념 설명이 친절한 논문", 
        "tutorial 성격의 논문"
    ],
    "중급자": [
        "핵심 원리를 깊이 다룬 논문",
        "수학적 증명이 포함된 논문",
        "ablation study가 상세한 논문"
    ],
    "고급자": [
        "최신 SOTA 논문",
        "theoretical foundation 논문",
        "novel architecture 제안 논문"
    ]
}

# 연구 목적별 논문 분류
search_by_purpose = {
    "이론_이해": "수학적 기초가 탄탄한 논문들",
    "실무_적용": "구현 세부사항이 자세한 논문들", 
    "연구_확장": "한계점과 future work이 명확한 논문들",
    "트렌드_파악": "최근 1-2년 high-impact 논문들"
}
```

## 📊 논문 평가 및 우선순위 매트릭스

### 🎖️ 논문 중요도 평가 기준
```python
paper_importance_score = {
    "citation_count": 0.3,      # 인용횟수 (영향력)
    "venue_tier": 0.25,         # 학회/저널 등급  
    "novelty_score": 0.2,       # 참신성/독창성
    "reproducibility": 0.15,    # 재현가능성
    "clarity_score": 0.1        # 명확성/이해도
}

# 예시 평가
example_evaluation = {
    "Attention Is All You Need": {
        "citation": 9.5,     # 90000+ 인용
        "venue": 9.0,        # NIPS (top-tier)
        "novelty": 10.0,     # 패러다임 변화  
        "repro": 8.0,        # 코드 공개
        "clarity": 8.5,      # 이해하기 쉬운 설명
        "총점": 9.1
    }
}
```

### 📋 맞춤형 읽기 리스트 생성
```markdown
## 🎯 [주제명] 논문 읽기 로드맵

### Week 1: Foundation (기초 다지기)
- [ ] **Survey Paper**: [주제] 전반적 overview
- [ ] **Seminal Work**: 이 분야를 시작한 원조 논문
- [ ] **Mathematical Foundation**: 수학적 기초 이론 논문

### Week 2: Core Methods (핵심 방법론)  
- [ ] **Method A**: 현재 주류 방법
- [ ] **Method B**: 대안적 접근법
- [ ] **Comparison Study**: A vs B 비교 분석 논문

### Week 3: Recent Advances (최신 발전)
- [ ] **SOTA Method**: 현재 최고 성능 방법
- [ ] **Efficiency Improvement**: 효율성 개선 연구
- [ ] **Novel Architecture**: 새로운 구조 제안

### Week 4: Applications & Future (응용 및 미래)
- [ ] **Real-world Application**: 실제 적용 사례
- [ ] **Limitation Analysis**: 한계점 분석 논문  
- [ ] **Future Direction**: 미래 연구 방향 제시
```

## 🌐 논문 네트워크 시각화

### 🌐 외부 플랫폼별 검색 전략
```python
# arXiv 활용 전략
arxiv_search_tips = {
    "최신_동향": "arXiv:cs.AI, cs.LG, cs.CL 카테고리 정기 모니터링",
    "키워드_조합": "주요 키워드 + 'survey', 'review', 'comprehensive'",
    "저자_추적": "영향력 있는 저자들의 신규 업로드 알림 설정",
    "버전_변경": "논문 버전 업데이트 추적으로 연구 진전 파악"
}

# Google Scholar 활용법
scholar_search_strategy = {
    "인용_분석": "citation network로 영향력 있는 논문 발굴",
    "시간_필터": "최근 1-2년 논문으로 제한해서 최신 트렌드 파악",
    "저자_프로필": "연구자 h-index, i10-index로 신뢰도 평가",
    "관련_논문": "'Related articles' 기능으로 유사 연구 확장"
}

# Papers with Code 연동
pwc_integration = {
    "구현_우선": "코드가 공개된 논문 우선순위",
    "성능_벤치마크": "SOTA 성능 논문들의 구현 품질 확인",
    "재현성_검증": "실제 코드로 논문 주장 검증 가능",
    "데이터셋_매칭": "특정 데이터셋에서 성능 비교"
}

# Semantic Scholar API 활용
semantic_scholar_features = {
    "citation_graph": "논문 인용 네트워크 자동 분석",
    "influence_score": "논문의 실제 영향력 지수 계산",
    "author_network": "공저자 관계망 시각화",
    "research_trends": "특정 분야 연구 트렌드 분석"
}
```
```

### 🔧 implementation_guide.md (구현 가이드)
```markdown
# [논문 제목] - 구현 가이드

## 🔍 단계별 미니 구현

### Step 1: 기본 구조 구현
```python
class SimpleAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k)  # Query projection
        self.W_K = nn.Linear(d_model, d_k)  # Key projection  
        self.W_V = nn.Linear(d_model, d_k)  # Value projection
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        Q = self.W_Q(x)  # [batch, seq_len, d_k]
        K = self.W_K(x)  # [batch, seq_len, d_k]
        V = self.W_V(x)  # [batch, seq_len, d_k]
        
        # Attention 계산
        scores = Q @ K.transpose(-2, -1)  # [batch, seq_len, seq_len]
        weights = F.softmax(scores / math.sqrt(self.d_k), dim=-1)
        output = weights @ V  # [batch, seq_len, d_k]
        
        return output, weights  # attention weights도 반환!
```

### Step 2: 디버깅 및 시각화
```python
# Attention 시각화 예시
def visualize_attention(weights, tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights[0].detach().numpy(), 
                xticklabels=tokens, yticklabels=tokens)
    plt.title('Attention Weights Visualization')
    plt.show()

# 예시 사용
tokens = ["I", "love", "you"]
output, weights = model(input_embeddings)
visualize_attention(weights, tokens)
```

### Step 3: 최적화 팁
- **Gradient Clipping**: 그래디언트 폭주 방지
- **Learning Rate Scheduling**: 학습률 조절 전략
- **Dropout**: 정규화 기법

## 📊 성능 체 레이터
```python
# 학습 과정 모니터링
epoch_metrics = {
    0:    {"loss": 8.5, "attention_entropy": 2.1},  # 완전 무작위
    100:  {"loss": 4.2, "attention_entropy": 1.8},  # 패턴 학습 시작
    1000: {"loss": 0.8, "attention_entropy": 1.2},  # 집중된 어텐션
}

# 학습 과정에서 실제 변화
"Random attention -> Positional bias -> Semantic attention"
```
```

### 📋 step_by_step_learning.md (단계별 학습)
```markdown
# [논문 제목] - 단계별 학습 경로

## 🏃‍♂️ 단계별 학습 체크리스트

### Level 1: 기본 이해 (30분)
- [ ] 문제 정의 명확히 이해
- [ ] "이 논문을 한 줄로 요약하면?" 답변 가능
- [ ] 기존 방법의 한계 3개 나열
- [ ] 제안 방법의 핵심 아이디어 설명 가능

### Level 2: 구조 파악 (45분)
- [ ] 모델 아키텍쳐 다이어그램 이해
- [ ] 데이터 플로우 추적 (input -> output)
- [ ] 핵심 수식 3-5개 의미 이해
- [ ] 파라미터 역할 및 학습 대상 파악

### Level 3: 깊은 이해 (60분)
- [ ] "왜 이렇게 설계했는가?" 질문에 답변
- [ ] Ablation study 결과 해석
- [ ] 한계점 및 실패 케이스 이해
- [ ] 대안 방법들과 비교 분석

### Level 4: 실전 적용 (90분)
- [ ] 미니 구현 완성
- [ ] 구현이 논문 주장과 일치하는지 검증
- [ ] 다른 도메인에 응용 아이디어 3개
- [ ] 실제 사용 시 주의사항 및 팁 도출

## 🧠 이해도 체크 질문
```python
# 자가 진단 템플릿
질문_level_1 = [
    "이 논문이 해결하는 핵심 문제는?",
    "기존 방법의 가장 큰 문제는?",
    "제안 방법의 핵심 아이디어는?"
]

질문_level_2 = [
    "모델에서 데이터가 어떻게 변환되는가?",
    "각 레이어에서 텐서 차원은?",
    "학습 가능한 파라미터는 무엇들인가?"
]

질문_level_3 = [
    "왜 [방법A]가 아니라 [방법B]를 선택했는가?",
    "이 가정을 제거하면 어떻게 될까?",
    "어떤 상황에서 실패할 가능성이 높을까?"
]
```
```

---

## 🎯 핵심 명령어 패턴

### 🔬 수학적/구현적 깊이 분석
```bash
# 수학적 기초 분석
"[논문.pdf]의 핵심 수식 3개를 'I love you' 예시로 단계별 설명해줘"
"각 파라미터가 학습 과정에서 어떻게 진화하는지 보여줘"
"역전파에서 그래디언트가 어떻게 흐르는지 추적해줘"

# 구현 가이드 생성
"[논문.pdf]를 실제 구현 가능한 PyTorch 코드로 단계별로 보여줘"
"차원 변화를 추적하면서 미니 구현해줘"
"학습 과정에서 attention weight가 어떻게 변하는지 시뮬레이션해줘"

# 완전분해 분석
"[논문.pdf]를 DeepDive Framework로 4-Layer 완전분해해줘"
"mathematical_foundations.md, implementation_guide.md, step_by_step_learning.md 모두 생성해줘"
```

### 🔍 논문 검색 및 탐색
```bash
# 관련 논문 발굴
"[논문.pdf]와 관련된 핵심 논문 10개를 시간순으로 찾아줘"
"이 논문이 인용한 주요 선행연구 5개와 그 연결관계 분석해줘"
"이 논문을 발전시킨 후속 연구들 찾아서 계보도 그려줘"

# 주제별 논문 검색
"[키워드] 관련 최신 논문 트렌드와 핵심 논문 5개 추천해줘"
"[기법A]와 [기법B]를 결합한 연구들 찾아서 비교 분석해줘"
"[저자명]의 주요 논문들을 시간순으로 정리해서 연구 방향 분석해줘"

# 외부 소스 기반 논문 검색
"arXiv에서 [키워드] 관련 최신 논문 5개 찾아서 요약해줘"
"Google Scholar에서 이 논문의 인용 논문들 중 핵심 3개 추천해줘"  
"Semantic Scholar API로 [논문제목]의 citation network 분석해줘"
"Papers with Code에서 [방법론] 구현코드가 있는 논문들 찾아줘"
```

### 🎨 창의적 확장 분석
```bash
# 논문 한계 및 개선
"이 논문의 숨겨진 약점 5가지와 구체적 개선 방안 제시해줘"
"이 방법이 실패할 수 있는 edge case들 분석해줘"
"더 효율적인 대안 설계 아이디어 3가지"

# 다른 도메인 적용
"이 기법을 [컴퓨터 비전/NLP/강화학습]에 적용하는 방법"
"이 논문 + [다른 기술]을 결합한 새로운 아키텍처 제안"

# 미래 연구 예측
"이 분야가 5년 뒤 어떻게 발전할지 예측하고 근거 제시해줘"
"이 논문이 촉발할 수 있는 연구 방향 5가지"
```

---

## 🧪 메타 학습 및 자가진단

### 📊 학습 후 기록 템플릿
```markdown
📅 날짜: 
📖 논문: 
⏱️ 소요 시간: 

### 🎯 핵심 수확
1. **새로 배운 수학적 개념**: 
2. **구현 시 핵심 포인트**: 
3. **가장 인상 깊었던 아이디어**: 

### ❓ 아직 이해 못한 부분
1. 
2. 

### 🔄 다음 학습 계획
- [ ] 관련 논문 ____ 읽기
- [ ] ____ 직접 구현해보기
- [ ] ____ 개념 더 깊이 공부
```

### 🔄 반복 학습 사이클
```
1. 스캔 → 2. 수학분석 → 3. 구현 → 4. 질문 → 5. 확장
     ↑                                            ↓
     ←────────── 이해 부족시 재귀 ──────────────←
```

---

## 🎯 시스템 목표

- **수학적 엄밀성**: 모든 수식의 물리적 의미 명확화
- **구현 가능성**: 실제 코드로 검증 가능한 수준
- **직관적 이해**: 복잡한 개념도 쉬운 예시로 설명
- **실전 적용**: 다른 도메인에 바로 적용 가능한 아이디어
- **깊이 있는 학습**: 파라미터 레벨까지의 미시적 이해

이 프레임워크를 통해 논문이 단순한 정보가 아닌 **실제로 사용할 수 있는 도구**가 되도록 합니다! 🚀