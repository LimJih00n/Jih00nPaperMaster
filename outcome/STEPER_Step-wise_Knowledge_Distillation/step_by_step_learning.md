# STEPER - 단계별 학습 경로

## 🏃‍♂️ 단계별 학습 체크리스트

### Level 1: 기본 이해 (30분)
- [ ] **문제 정의 명확히 이해**: Multi-Step Retrieval-Augmented LM에서 기존 KD의 한계점
- [ ] **"이 논문을 한 줄로 요약하면?" 답변 가능**: "단계별로 다른 추론 능력을 학습시키는 Knowledge Distillation"
- [ ] **기존 방법의 한계 3개 나열**:
  - ✅ 단계별 추론 능력 차이 무시
  - ✅ 각 단계별 정보량 변화 미반영  
  - ✅ 일괄 처리로 인한 step-by-step 본질 상실
- [ ] **제안 방법의 핵심 아이디어 설명 가능**: Step-wise Dataset + Reasoning Difficulty-Aware Training

#### 🧠 Level 1 자가진단 질문
```python
질문_level_1 = [
    "STEPER가 해결하는 핵심 문제는?",
    "기존 vanilla-KD의 가장 큰 문제는?", 
    "STEPER의 핵심 아이디어는?"
]

# 답변 예시
답변_예시 = {
    "문제": "Multi-step retrieval에서 각 단계별로 다른 추론 능력이 필요한데 기존 방법은 이를 구분하지 않음",
    "vanilla-KD_문제": "최종 단계 데이터만 사용해서 중간 단계의 추론 과정을 학습하지 못함", 
    "핵심_아이디어": "3가지 추론 능력(초기화/확장/집합)을 단계별 데이터로 각각 학습"
}
```

### Level 2: 구조 파악 (45분)
- [ ] **모델 아키텍처 다이어그램 이해**: Teacher LM → Step-wise Dataset → Student LM with Adaptive Weighting
- [ ] **데이터 플로우 추적 (input → output)**:
  ```python
  # 데이터 플로우 추적
  input_flow = {
      "First-step": "Q + P1 → R1 (초기 추론)",
      "Mid-step": "Q + P1,P2 + R1 → R2 (확장 추론)",
      "Final-step": "Q + P1,P2,P3 + R1,R2 → Answer (집합 추론)"
  }
  ```
- [ ] **핵심 수식 3-5개 의미 이해**:
  - ✅ Multi-step RAG: ∏P(rs|q,P≤s,R<s)·P(a|q,P≤S,R<S)
  - ✅ STEPER Loss: L = (1/3n)Σ[Linit + Lexp + Lagg]  
  - ✅ Difficulty-aware: Lfinal = Σ[1/(2σj²)Lj + log σj]
- [ ] **파라미터 역할 및 학습 대상 파악**: 
  - 공유 파라미터 (LLM backbone) + 적응적 가중치 (σ parameters)

#### 🧠 Level 2 자가진단 질문
```python
질문_level_2 = [
    "STEPER에서 데이터가 어떻게 변환되는가?",
    "각 단계에서 입력 차원과 출력은 어떻게 달라지는가?",
    "학습 가능한 파라미터는 무엇들인가?"
]

# 차원 변화 추적 예시
차원_변화 = {
    "Step_1": {
        "input": "Question + 4 passages (약 1000 tokens)",  
        "output": "First reasoning (약 50 tokens)",
        "정보량": "제한적 → 초기 추론만 가능"
    },
    "Step_2": {
        "input": "Question + 8 passages + previous reasoning (약 1500 tokens)",
        "output": "Expanded reasoning (약 80 tokens)", 
        "정보량": "중간 → 연결 추론 가능"
    },
    "Final": {
        "input": "Question + all passages + all reasoning (약 2000 tokens)",
        "output": "Final answer (약 20 tokens)",
        "정보량": "최대 → 종합 판단 가능"  
    }
}
```

### Level 3: 깊은 이해 (60분)
- [ ] **"왜 이렇게 설계했는가?" 질문에 답변**:
  - Step-wise 이유: 각 단계마다 접근 가능한 정보와 필요한 추론이 다름
  - Difficulty-aware 이유: 태스크별 난이도 차이를 자동으로 조절하기 위함
  - 3-way split 이유: 의학 진단 과정처럼 자연스러운 단계 구분
- [ ] **실험 결과 해석**: 
  - vanilla-KD 대비 9.5% 평균 향상의 원인 분석
  - GPT-4 평가에서 모든 추론 능력 향상 확인
- [ ] **한계점 및 실패 케이스 이해**:
  - Teacher 모델 품질에 의존적 
  - 필터링 방식이 단순함 (정답 일치만 확인)
  - 추론 경로 정확성 검증 부족
- [ ] **대안 방법들과 비교 분석**:
  - vs vanilla-KD: 중간 과정 학습 유무
  - vs Self-RAG: 적응적 가중치 vs 메타 토큰
  - vs IRCOT: 학생 모델 훈련 vs ICL

#### 🧠 Level 3 자가진단 질문
```python
질문_level_3 = [
    "왜 Single-step이 아니라 Multi-step을 선택했는가?",
    "Difficulty-aware training을 제거하면 어떻게 될까?",  
    "어떤 상황에서 STEPER가 실패할 가능성이 높을까?"
]

# 심층 분석 예시
심층_분석 = {
    "Multi-step_장점": "복잡한 질문에서 단계적 정보 누적으로 정확도 향상",
    "Difficulty-aware_효과": "어려운 태스크에 집중하지 않고 균형있게 학습 가능",
    "실패_케이스": [
        "Teacher 모델이 잘못된 추론을 할 때",
        "검색 결과가 매우 부정확할 때", 
        "단일 hop으로 해결 가능한 간단한 질문일 때"
    ]
}
```

### Level 4: 실전 적용 (90분)
- [ ] **미니 구현 완성**: implementation_guide.md의 코드 실제 실행
- [ ] **구현이 논문 주장과 일치하는지 검증**: 
  - Sigma 파라미터 변화 모니터링
  - 단계별 성능 향상 확인
  - Loss 감소 패턴 검증
- [ ] **다른 도메인에 응용 아이디어 3개**:
  ```python
  응용_아이디어 = {
      "Computer_Vision": "Object Detection에서 step-wise localization + classification",
      "Reinforcement_Learning": "Multi-step planning에서 단계별 action selection", 
      "Code_Generation": "단계별 코드 작성 (outline → implementation → debugging)"
  }
  ```
- [ ] **실제 사용 시 주의사항 및 팁 도출**:
  - Teacher 모델 품질 검증 필수
  - 적절한 retrieval step 수 선택 (보통 3-5개)
  - Domain-specific fine-tuning 고려

## 🧠 이해도 체크 질문 (전체)

### 📊 종합 자가진단 템플릿
```python
자가진단_체크리스트 = {
    "Level_1_기본": {
        "문제_정의": "Multi-step에서 단계별 추론 능력이 다른데 기존 KD는 이를 무시 ✅/❌",
        "핵심_해결책": "Step-wise dataset으로 3가지 추론 능력을 각각 학습 ✅/❌",
        "주요_성과": "8B 모델이 70B teacher와 동등한 성능 달성 ✅/❌"
    },
    
    "Level_2_구조": {
        "데이터_플로우": "Q+P1→R1, Q+P1,P2+R1→R2, ... ✅/❌", 
        "핵심_수식": "Multi-task loss + Difficulty-aware weighting ✅/❌",
        "파라미터_종류": "LLM backbone + σ parameters ✅/❌"
    },
    
    "Level_3_깊이": {
        "설계_근거": "의학 진단처럼 단계별로 다른 추론이 필요 ✅/❌",
        "실험_해석": "모든 추론 능력에서 일관된 향상 ✅/❌", 
        "한계_인식": "Teacher 품질 의존적, 필터링 단순함 ✅/❌"
    },
    
    "Level_4_실전": {
        "구현_완료": "실제 코드 작성 및 실행 ✅/❌",
        "검증_수행": "논문 결과 재현 ✅/❌",
        "응용_도출": "3개 이상 도메인 확장 아이디어 ✅/❌"
    }
}

# 점수 계산
def calculate_understanding_score(checklist):
    total_items = sum(len(level.items()) for level in checklist.values())
    completed_items = 0  # ✅ 개수 계산
    
    score = (completed_items / total_items) * 100
    return f"이해도 점수: {score:.1f}%"
```

## 🎯 단계별 학습 전략

### 🔄 반복 학습 사이클
```
1. 읽기 → 2. 이해 → 3. 구현 → 4. 검증 → 5. 확장
     ↑                                            ↓
     ←────── 이해 부족 시 다시 읽기 ──────────────←
```

### 📚 추천 학습 순서
1. **논문 통독** (scout_report.md 참고)
2. **수학적 이해** (mathematical_foundations.md 참고)  
3. **코드 구현** (implementation_guide.md 따라하기)
4. **실험 및 검증** (직접 결과 확인)
5. **창의적 확장** (다른 도메인 적용 고민)

### 🎮 실습 미션

#### Mission 1: Quick Implementation
```python
# 30분 도전: 기본 STEPER 프레임워크 구현
class MiniSTEPER:
    def __init__(self):
        self.sigma_init = 1.0
        self.sigma_exp = 1.0  
        self.sigma_agg = 1.0
        
    def calculate_loss(self, L_init, L_exp, L_agg):
        return (L_init/(2*self.sigma_init**2) + L_exp/(2*self.sigma_exp**2) + 
                L_agg/(2*self.sigma_agg**2) + 
                np.log(self.sigma_init) + np.log(self.sigma_exp) + np.log(self.sigma_agg))

# 구현 후 테스트
mini_steper = MiniSTEPER()
test_loss = mini_steper.calculate_loss(2.0, 3.0, 1.5)
print(f"Test loss: {test_loss}")  # 예상값과 비교
```

#### Mission 2: Ablation Study
```python
# 60분 도전: Difficulty-aware training 효과 실험
def compare_training_strategies():
    # 1. Uniform weighting (λ=1,1,1)  
    # 2. Manual weighting (λ=1.5,1,0.5)
    # 3. Difficulty-aware (adaptive σ)
    
    strategies = ['uniform', 'manual', 'difficulty_aware']
    results = []
    
    for strategy in strategies:
        # 각 전략으로 훈련 시뮬레이션
        performance = simulate_training(strategy)
        results.append(performance)
        
    # 결과 비교 및 분석
    return analyze_results(results)

comparison_results = compare_training_strategies()
```

#### Mission 3: Domain Transfer
```python
# 90분 도전: 다른 도메인에 STEPER 적용
class STEPERForCodeGeneration:
    """코드 생성에 STEPER 적용"""
    def __init__(self):
        self.reasoning_types = {
            'planning': 'Overall code structure planning',
            'implementation': 'Detailed code writing', 
            'debugging': 'Error detection and fixing'
        }
        
    def generate_stepwise_code(self, problem_description):
        # Step 1: Planning (Reasoning Initialization)
        plan = self.plan_code_structure(problem_description)
        
        # Step 2: Implementation (Reasoning Expansion) 
        code = self.implement_code(problem_description, plan)
        
        # Step 3: Debugging (Reasoning Aggregation)
        final_code = self.debug_code(code, problem_description)
        
        return final_code

# 실제 문제로 테스트
code_steper = STEPERForCodeGeneration()
result = code_steper.generate_stepwise_code("Sort an array using quicksort")
```

## 🏆 학습 완료 인증

### ✅ 최종 체크포인트
- [ ] **논문 완전 이해**: 모든 수식과 개념 설명 가능
- [ ] **코드 구현 완료**: 작동하는 STEPER 구현체 보유
- [ ] **실험 결과 검증**: 논문 주장과 일치하는 결과 확인  
- [ ] **창의적 확장**: 3개 이상 다른 도메인 응용 아이디어
- [ ] **한계점 인식**: STEPER의 약점과 개선 방향 파악

### 🎓 마스터 레벨 도전과제
1. **논문 재현**: 완전한 실험 재현 (HotpotQA에서 동일 결과)
2. **개선 제안**: STEPER의 한계를 해결하는 새로운 방법 제안
3. **도메인 적용**: 실제로 다른 분야에 STEPER 적용 및 성능 검증

이 단계별 학습 경로를 통해 STEPER를 완전히 마스터할 수 있습니다! 🚀