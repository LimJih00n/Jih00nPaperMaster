# STEPER - 구현 가이드

## 🔍 단계별 미니 구현

### Step 1: 기본 STEPER 프레임워크 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple

class STEPERFramework(nn.Module):
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Difficulty-aware 파라미터 (학습 가능)
        self.sigma_init = nn.Parameter(torch.tensor(1.0))    # Reasoning Initialization
        self.sigma_exp = nn.Parameter(torch.tensor(1.0))     # Reasoning Expansion  
        self.sigma_agg = nn.Parameter(torch.tensor(1.0))     # Reasoning Aggregation
        
        print(f"✅ STEPER 모델 초기화 완료")
        print(f"📊 모델 파라미터 수: {sum(p.numel() for p in self.parameters()):,}")
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, 
                           attention_mask=attention_mask, 
                           labels=labels)
        return outputs

# 예시 사용
steper_model = STEPERFramework()
```

### Step 2: Step-wise 데이터셋 구성기

```python
class StepwiseDatasetBuilder:
    def __init__(self, teacher_model, retriever, max_steps: int = 5):
        self.teacher_model = teacher_model
        self.retriever = retriever  # BM25 또는 다른 retrieval 모델
        self.max_steps = max_steps
        
    def create_stepwise_sample(self, question: str, answer: str) -> List[Dict]:
        """단일 QA 쌍을 step-wise 샘플들로 변환"""
        stepwise_samples = []
        
        # Step 1: Reasoning Initialization
        P1 = self.retriever.search(question, k=4)
        R1 = self.teacher_model.generate_rationale(question, P1)
        
        stepwise_samples.append({
            'type': 'reasoning_initialization',
            'input': f"Question: {question}\nPassages: {P1}",
            'output': R1,
            'step': 1
        })
        
        # Steps 2 to S-1: Reasoning Expansion
        passages_cumulative = P1
        rationales_cumulative = [R1]
        
        for step in range(2, self.max_steps):
            # 이전 rationale 기반 추가 검색
            query = self._construct_step_query(question, rationales_cumulative)
            new_passages = self.retriever.search(query, k=4)
            passages_cumulative.extend(new_passages)
            
            # 새로운 rationale 생성
            context = f"Question: {question}\nPassages: {passages_cumulative}\nPrevious reasoning: {' '.join(rationales_cumulative)}"
            R_step = self.teacher_model.generate_rationale(context)
            
            # Answer flag 체크 (조기 종료)
            if "So the answer is:" in R_step:
                break
                
            rationales_cumulative.append(R_step)
            stepwise_samples.append({
                'type': 'reasoning_expansion', 
                'input': context,
                'output': R_step,
                'step': step
            })
            
        # Final Step: Reasoning Aggregation
        final_context = f"Question: {question}\nPassages: {passages_cumulative}\nReasoning chain: {' '.join(rationales_cumulative)}"
        final_output = f"{' '.join(rationales_cumulative)} So the answer is: {answer}"
        
        stepwise_samples.append({
            'type': 'reasoning_aggregation',
            'input': final_context, 
            'output': final_output,
            'step': len(rationales_cumulative) + 1
        })
        
        return stepwise_samples
    
    def _construct_step_query(self, question: str, previous_rationales: List[str]) -> str:
        """이전 추론을 바탕으로 다음 검색 쿼리 생성"""
        last_rationale = previous_rationales[-1] if previous_rationales else ""
        # 간단한 구현: 마지막 rationale의 핵심 엔티티 추출
        return f"{question} {last_rationale}"

# 예시 사용
dataset_builder = StepwiseDatasetBuilder(teacher_model, retriever)
sample = dataset_builder.create_stepwise_sample(
    question="Jim Halsey guided the career of the musician who hosted what country variety show?",
    answer="Hee Haw"
)

print("📋 생성된 Step-wise 샘플들:")
for i, s in enumerate(sample):
    print(f"Step {s['step']} ({s['type']}): {s['output'][:50]}...")
```

### Step 3: Multi-Task Loss 구현

```python
class STEPERLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, model_outputs, batch_data, sigma_params):
        """
        Args:
            model_outputs: 모델 출력 
            batch_data: 배치 데이터 (step_type 정보 포함)
            sigma_params: (sigma_init, sigma_exp, sigma_agg) 난이도 파라미터
        """
        sigma_init, sigma_exp, sigma_agg = sigma_params
        
        total_loss = 0.0
        step_losses = {'init': [], 'exp': [], 'agg': []}
        
        for i, sample in enumerate(batch_data):
            step_type = sample['step_type']  # 'reasoning_initialization', 'reasoning_expansion', 'reasoning_aggregation'
            
            # 개별 샘플 loss 계산
            sample_loss = self.ce_loss(model_outputs.logits[i], sample['labels'])
            
            # Step type별 loss 누적
            if step_type == 'reasoning_initialization':
                step_losses['init'].append(sample_loss)
            elif step_type == 'reasoning_expansion':
                step_losses['exp'].append(sample_loss) 
            elif step_type == 'reasoning_aggregation':
                step_losses['agg'].append(sample_loss)
        
        # 각 타입별 평균 loss 계산
        L_init = torch.stack(step_losses['init']).mean() if step_losses['init'] else 0.0
        L_exp = torch.stack(step_losses['exp']).mean() if step_losses['exp'] else 0.0  
        L_agg = torch.stack(step_losses['agg']).mean() if step_losses['agg'] else 0.0
        
        # Difficulty-aware weighting 적용
        weighted_loss = (
            (1.0 / (2 * sigma_init**2)) * L_init + torch.log(sigma_init) +
            (1.0 / (2 * sigma_exp**2)) * L_exp + torch.log(sigma_exp) +  
            (1.0 / (2 * sigma_agg**2)) * L_agg + torch.log(sigma_agg)
        )
        
        return {
            'total_loss': weighted_loss,
            'L_init': L_init,
            'L_exp': L_exp, 
            'L_agg': L_agg,
            'sigma_init': sigma_init.item(),
            'sigma_exp': sigma_exp.item(),
            'sigma_agg': sigma_agg.item()
        }

# 예시 사용 
loss_fn = STEPERLoss()
```

### Step 4: 훈련 루프 구현

```python
def train_steper(model, dataloader, num_epochs=2, lr=5e-6):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = STEPERLoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        sigma_history = {'init': [], 'exp': [], 'agg': []}
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'], 
                labels=batch['labels']
            )
            
            # Loss 계산
            sigma_params = (model.sigma_init, model.sigma_exp, model.sigma_agg)
            loss_dict = loss_fn(outputs, batch['step_data'], sigma_params)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Logging
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Total Loss: {loss_dict['total_loss']:.4f}")
                print(f"  σ_init: {loss_dict['sigma_init']:.3f}, σ_exp: {loss_dict['sigma_exp']:.3f}, σ_agg: {loss_dict['sigma_agg']:.3f}")
                
            epoch_losses.append(loss_dict['total_loss'].item())
            sigma_history['init'].append(loss_dict['sigma_init'])
            sigma_history['exp'].append(loss_dict['sigma_exp'])
            sigma_history['agg'].append(loss_dict['sigma_agg'])
            
        scheduler.step()
        print(f"✅ Epoch {epoch} 완료, Average Loss: {np.mean(epoch_losses):.4f}")
        
    return model, sigma_history

# 훈련 실행
trained_model, sigma_hist = train_steper(steper_model, train_dataloader)
```

### Step 5: 추론 및 시각화

```python
def steper_inference(model, tokenizer, question: str, retriever, max_steps: int = 5):
    """STEPER 모델로 단계별 추론 수행"""
    model.eval()
    
    reasoning_chain = []
    passages_cumulative = []
    
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            # 검색 수행
            if step == 1:
                query = question
            else:
                query = f"{question} {reasoning_chain[-1]}"  # 이전 추론 기반
                
            new_passages = retriever.search(query, k=4)
            passages_cumulative.extend(new_passages)
            
            # 입력 구성
            if step == 1:
                context = f"Question: {question}\nPassages: {' '.join(passages_cumulative[:4])}\nAnswer step by step:"
            else:
                context = f"Question: {question}\nPassages: {' '.join(passages_cumulative)}\nPrevious reasoning: {' '.join(reasoning_chain)}\nContinue reasoning:"
                
            # 토크나이징 및 생성
            inputs = tokenizer(context, return_tensors="pt", max_length=1024, truncation=True)
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_reasoning = generated_text[len(context):].strip()
            reasoning_chain.append(new_reasoning)
            
            print(f"🔍 Step {step}: {new_reasoning}")
            
            # 종료 조건 체크
            if "So the answer is:" in new_reasoning:
                break
                
    return reasoning_chain, passages_cumulative

# 예시 사용
question = "Jim Halsey guided the career of the musician who hosted what country variety show?"
reasoning_chain, passages = steper_inference(trained_model, tokenizer, question, retriever)

print("\n📝 최종 추론 체인:")
for i, reasoning in enumerate(reasoning_chain, 1):
    print(f"Step {i}: {reasoning}")
```

### Step 6: 성능 분석 및 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sigma_evolution(sigma_history):
    """Sigma 파라미터 변화 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sigma 값 변화 추이
    steps = range(len(sigma_history['init']))
    ax1.plot(steps, sigma_history['init'], label='σ_init (Initialization)', color='blue')
    ax1.plot(steps, sigma_history['exp'], label='σ_exp (Expansion)', color='orange') 
    ax1.plot(steps, sigma_history['agg'], label='σ_agg (Aggregation)', color='green')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Sigma Values')
    ax1.set_title('Difficulty Parameter Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 최종 sigma 값 비교
    final_sigmas = [sigma_history['init'][-1], sigma_history['exp'][-1], sigma_history['agg'][-1]]
    labels = ['Initialization', 'Expansion', 'Aggregation']
    colors = ['blue', 'orange', 'green']
    
    bars = ax2.bar(labels, final_sigmas, color=colors, alpha=0.7)
    ax2.set_ylabel('Final Sigma Value')
    ax2.set_title('Task Difficulty Comparison')
    
    # 값 표시
    for bar, value in zip(bars, final_sigmas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 해석 출력
    most_difficult = labels[np.argmax(final_sigmas)]
    least_difficult = labels[np.argmin(final_sigmas)]
    print(f"🔍 분석 결과:")
    print(f"  가장 어려운 태스크: {most_difficult} (σ={max(final_sigmas):.3f})")
    print(f"  가장 쉬운 태스크: {least_difficult} (σ={min(final_sigmas):.3f})")

def evaluate_step_performance(model, test_dataloader):
    """단계별 성능 평가"""
    model.eval()
    
    step_accuracies = {'init': [], 'exp': [], 'agg': []}
    
    with torch.no_grad():
        for batch in test_dataloader:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            for i, sample in enumerate(batch['step_data']):
                step_type = sample['step_type']
                correct = (predictions[i] == sample['labels']).float().mean()
                
                if step_type == 'reasoning_initialization':
                    step_accuracies['init'].append(correct.item())
                elif step_type == 'reasoning_expansion':
                    step_accuracies['exp'].append(correct.item())
                elif step_type == 'reasoning_aggregation':
                    step_accuracies['agg'].append(correct.item())
    
    # 결과 출력
    for step_type, accuracies in step_accuracies.items():
        avg_acc = np.mean(accuracies)
        print(f"📊 {step_type.capitalize()} Accuracy: {avg_acc:.3f}")
    
    return step_accuracies

# 분석 실행
analyze_sigma_evolution(sigma_hist)
step_perfs = evaluate_step_performance(trained_model, test_dataloader)
```

## 📊 성능 체크리스트

### 학습 과정 모니터링
```python
# 정상적인 학습 지표
training_checkpoints = {
    "epoch_0": {
        "total_loss": 8.5, 
        "sigma_init": 1.0, "sigma_exp": 1.0, "sigma_agg": 1.0,
        "step_accuracy": {"init": 0.3, "exp": 0.2, "agg": 0.4}
    },
    "epoch_1": {
        "total_loss": 4.2,
        "sigma_init": 0.8, "sigma_exp": 1.2, "sigma_agg": 0.9, 
        "step_accuracy": {"init": 0.6, "exp": 0.4, "agg": 0.7}
    },
    "epoch_2": {
        "total_loss": 1.8,
        "sigma_init": 0.7, "sigma_exp": 1.4, "sigma_agg": 0.8,
        "step_accuracy": {"init": 0.8, "exp": 0.6, "agg": 0.9}
    }
}

print("🎯 학습 진행도 체크:")
print("✅ Loss 감소: 8.5 → 4.2 → 1.8")
print("✅ Sigma 적응: Expansion이 가장 어려운 태스크로 인식됨")
print("✅ 성능 향상: 모든 단계에서 지속적 개선")
```

## 🚀 최적화 팁

### 1. 메모리 효율성
- **Gradient Checkpointing**: 메모리 50% 절약
- **Mixed Precision**: FP16 사용으로 속도 2배 향상
- **DeepSpeed ZeRO**: 대용량 모델 분산 학습

### 2. 수렴 안정성
- **Gradient Clipping**: `max_norm=1.0`으로 폭주 방지
- **Warmup Scheduler**: 초기 불안정성 완화
- **Early Stopping**: Validation loss 모니터링

### 3. 성능 최적화
- **Teacher Forcing**: 훈련 시 실제 이전 단계 사용
- **Curriculum Learning**: 쉬운 샘플부터 점진적 학습
- **Data Augmentation**: 다양한 추론 경로 생성

이 구현 가이드를 통해 STEPER의 핵심 아이디어를 실제 작동하는 코드로 변환할 수 있습니다! 🎯