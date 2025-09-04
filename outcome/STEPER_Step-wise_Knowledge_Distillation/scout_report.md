# STEPER 논문 초기 스캔 보고서

## 🎯 핵심 한 줄 요약
**Multi-Step Retrieval-Augmented LM에서 단계별 추론 능력을 향상시키기 위한 Step-wise Knowledge Distillation 프레임워크**

## 📊 논문 기본 정보
- **제목**: STEPER: Step-wise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented LM
- **학회**: Anonymous ACL submission
- **연구 분야**: Knowledge Distillation, Retrieval-Augmented Generation, Multi-Step Reasoning
- **모델 규모**: Teacher (Llama3.1-70B) → Student (Llama3.1-8B)

## 🔍 문제 정의 및 동기
### 기존 방법의 한계 3가지
1. **단계별 추론 능력 부족**: 기존 KD 방법들은 각 단계에서 필요한 서로 다른 추론 능력을 고려하지 않음
2. **정보량 변화 무시**: 각 retrieval step에서 접근 가능한 정보량이 달라지는데 이를 반영하지 못함
3. **일괄 처리 방식**: 모든 retrieval 결과를 한 번에 처리하여 step-by-step 추론의 본질을 놓침

### 의학 진단 비유로 본 필요성
```python
# 의사의 진단 과정 (3단계)
step_1_reasoning_initialization = "증상 기반 잠재 질병 식별"
step_2_reasoning_expansion = "추가 검사 (X-ray, 초음파) 수행" 
step_3_reasoning_aggregation = "모든 정보 종합하여 최종 진단"

# 각 단계마다 다른 추론 능력과 정보량이 필요!
```

## 🧠 핵심 아이디어
### STEPER의 3가지 추론 능력 분류
1. **Reasoning Initialization** (첫 단계)
   - 초기 retrieval 결과로 추론 시작점 설정
   - "Jim Halsey가 어떤 뮤지션의 커리어를 관리했는가?" → Roy Clark 식별

2. **Reasoning Expansion** (중간 단계)  
   - 새로운 evidence와 기존 맥락 통합
   - Roy Clark → "Hee Haw" show 연결

3. **Reasoning Aggregation** (최종 단계)
   - 모든 정보 종합하여 최종 답안 도출
   - "So the answer is: Hee Haw"

## 📐 방법론 핵심
### Step-wise Dataset 구성
```python
# First-step data: 초기 추론 학습
input: Q + P1 → output: R1

# Mid-step data: 추론 확장 학습  
input: Q + P1,P2 + R1 → output: R2

# Final-step data: 추론 집합 학습
input: Q + P1,P2,P3 + R1,R2 → output: Answer
```

### Reasoning Difficulty-Aware Training
- 각 추론 태스크의 난이도를 adaptive weighting으로 조절
- σ 파라미터로 난이도 표현, 어려운 태스크일수록 높은 가중치

## 📊 실험 결과 하이라이트
### 성능 향상 (HotpotQA 기준)
- **vanilla-KD**: 54.8% Accuracy
- **STEPER**: 61.0% Accuracy  
- **향상률**: +6.2% (상대적으로 11.3% 향상)

### 모델 확장성
- **1.5B STEPER** > **3B vanilla-KD** 성능
- **3B STEPER** ≈ **72B Teacher** 성능
- **7B STEPER** > **72B Teacher** 성능

## 🔬 핵심 인사이트
### 1. Step-wise가 핵심이다
- GPT-4 평가에서 모든 추론 능력(Initialization, Expansion, Aggregation)에서 향상 확인
- 단계별 데이터가 많을수록 성능 지속 향상

### 2. 범용성이 뛰어나다  
- IRCOT, Self-Ask 등 다양한 multi-step 프레임워크에 적용 가능
- Out-of-domain에서도 1-4% 일관된 성능 향상

### 3. 효율성이 우수하다
- 작은 모델로 큰 모델 성능 달성 → 실용성 높음
- Latency vs Accuracy trade-off에서 최적점 제공

## 🚀 혁신적 기여점
1. **이론적**: Multi-step reasoning을 3단계로 체계화
2. **방법론적**: Step-wise Knowledge Distillation 최초 제안  
3. **실무적**: 8B 모델로 70B 성능 달성하는 실용적 프레임워크

## ❓ 추후 분석할 핵심 질문들
1. 왜 step-wise 접근이 이렇게 효과적인가?
2. Reasoning Difficulty-Aware Training의 수학적 원리는?
3. 다른 도메인(CV, RL)에도 적용 가능한가?
4. Teacher 모델의 품질이 성능에 미치는 영향은?

---
**다음 단계**: 수학적 기초 완전분해로 진행 →