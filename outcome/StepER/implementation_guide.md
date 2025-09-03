# StepER 구현 실습 가이드

## 🚀 Step-by-Step 구현 튜토리얼

### 📦 필요한 라이브러리
```python
# 기본 라이브러리
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np

# RAG 관련
from rank_bm25 import BM25Okapi  # BM25 retriever
import faiss  # 벡터 검색용 (선택)
```

---

## 1️⃣ 데이터셋 구축

### Step 1: Teacher Model로 추론 경로 생성

```python
def generate_stepwise_data(question, documents, teacher_model, tokenizer):
    """
    Teacher 모델을 사용해 단계별 추론 데이터 생성
    """
    stepwise_data = {
        'first_step': None,
        'mid_steps': [],
        'final_step': None
    }
    
    # 1단계: 초기 추론 (Reasoning Initialization)
    prompt_step1 = f"""
    Question: {question}
    Documents: {documents[0]}  # 첫 번째 검색 결과만 사용
    
    Based on the document, start reasoning about the question.
    Reasoning:
    """
    
    first_reasoning = teacher_model.generate(prompt_step1)
    stepwise_data['first_step'] = {
        'input': prompt_step1,
        'output': first_reasoning,
        'documents_used': documents[0]
    }
    
    # 2-4단계: 추론 확장 (Reasoning Expansion)
    accumulated_reasoning = first_reasoning
    for step in range(2, 5):  # 2, 3, 4단계
        prompt_mid = f"""
        Question: {question}
        Previous reasoning: {accumulated_reasoning}
        New documents: {documents[step-1]}
        
        Continue the reasoning with new information.
        Reasoning:
        """
        
        mid_reasoning = teacher_model.generate(prompt_mid)
        stepwise_data['mid_steps'].append({
            'input': prompt_mid,
            'output': mid_reasoning,
            'step': step
        })
        accumulated_reasoning += " " + mid_reasoning
    
    # 5단계: 최종 답변 (Reasoning Aggregation)
    prompt_final = f"""
    Question: {question}
    All reasoning: {accumulated_reasoning}
    All documents: {' '.join(documents)}
    
    Provide the final answer.
    So the answer is:
    """
    
    final_answer = teacher_model.generate(prompt_final)
    stepwise_data['final_step'] = {
        'input': prompt_final,
        'output': final_answer
    }
    
    return stepwise_data
```

### Step 2: 데이터 필터링

```python
def filter_correct_samples(stepwise_data, ground_truth):
    """
    정답과 일치하는 샘플만 필터링
    """
    # 최종 답변에서 답 추출
    predicted_answer = extract_answer(stepwise_data['final_step']['output'])
    
    # 정답과 비교
    if predicted_answer.lower().strip() == ground_truth.lower().strip():
        return stepwise_data
    else:
        return None  # 틀린 답변은 제외
```

---

## 2️⃣ StepER 모델 클래스 구현

### Multi-task Learning을 위한 커스텀 모델

```python
class StepERModel(nn.Module):
    """
    StepER: 3가지 추론 능력을 학습하는 모델
    """
    def __init__(self, base_model_name="meta-llama/Llama-3.1-8B"):
        super().__init__()
        
        # 기본 언어 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 난이도 파라미터 (σ) 초기화
        self.sigma_init = nn.Parameter(torch.tensor(1.0))  # 초기화 난이도
        self.sigma_exp = nn.Parameter(torch.tensor(1.0))   # 확장 난이도
        self.sigma_agg = nn.Parameter(torch.tensor(1.0))   # 집계 난이도
        
    def forward(self, input_ids, attention_mask, task_type):
        """
        task_type: 'init', 'exp', 'agg' 중 하나
        """
        # 기본 모델로 예측
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # Teacher forcing
        )
        
        return outputs.loss, task_type
    
    def compute_weighted_loss(self, losses_dict):
        """
        난이도 인식 가중치 적용 (수식 4)
        """
        weighted_loss = 0
        
        for task_type, loss in losses_dict.items():
            if task_type == 'init':
                sigma = self.sigma_init
            elif task_type == 'exp':
                sigma = self.sigma_exp
            else:  # 'agg'
                sigma = self.sigma_agg
            
            # 수식: (1/2σ²) * L + log(σ)
            weighted_loss += (1 / (2 * sigma**2)) * loss + torch.log(sigma)
        
        return weighted_loss
```

---

## 3️⃣ 학습 루프 구현

### Custom Trainer for Multi-task Learning

```python
class StepERTrainer:
    def __init__(self, model, train_dataset, eval_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=5e-6  # 논문에서 사용한 learning rate
        )
        
    def train_step(self, batch):
        """
        한 번의 학습 스텝
        """
        self.model.train()
        losses_dict = {}
        
        # (a) Reasoning Initialization 학습
        if 'first_step' in batch:
            loss_init, _ = self.model(
                input_ids=batch['first_step_ids'],
                attention_mask=batch['first_step_mask'],
                task_type='init'
            )
            losses_dict['init'] = loss_init
        
        # (b) Reasoning Expansion 학습
        for step_data in batch.get('mid_steps', []):
            loss_exp, _ = self.model(
                input_ids=step_data['ids'],
                attention_mask=step_data['mask'],
                task_type='exp'
            )
            if 'exp' not in losses_dict:
                losses_dict['exp'] = 0
            losses_dict['exp'] += loss_exp
        
        # (c) Reasoning Aggregation 학습
        if 'final_step' in batch:
            loss_agg, _ = self.model(
                input_ids=batch['final_step_ids'],
                attention_mask=batch['final_step_mask'],
                task_type='agg'
            )
            losses_dict['agg'] = loss_agg
        
        # 난이도 인식 가중치 적용
        total_loss = self.model.compute_weighted_loss(losses_dict)
        
        # 역전파
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {
            'total_loss': total_loss.item(),
            'sigma_init': self.model.sigma_init.item(),
            'sigma_exp': self.model.sigma_exp.item(),
            'sigma_agg': self.model.sigma_agg.item()
        }
    
    def train(self, num_epochs=2):
        """
        전체 학습 과정
        """
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            for batch_idx, batch in enumerate(self.train_dataset):
                metrics = self.train_step(batch)
                
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}: Loss={metrics['total_loss']:.4f}")
                    print(f"  σ values - Init: {metrics['sigma_init']:.2f}, "
                          f"Exp: {metrics['sigma_exp']:.2f}, "
                          f"Agg: {metrics['sigma_agg']:.2f}")
            
            # 평가
            self.evaluate()
    
    def evaluate(self):
        """
        검증 데이터로 평가
        """
        self.model.eval()
        # 평가 코드...
```

---

## 4️⃣ 추론 시 사용법

### Multi-step 추론 실행

```python
def stepwise_inference(question, model, retriever, num_steps=5):
    """
    학습된 StepER 모델로 단계별 추론
    """
    all_documents = []
    all_reasoning = []
    
    # Step 1: 초기 추론
    docs_1 = retriever.search(question, k=4)
    prompt_1 = f"Question: {question}\nDocuments: {docs_1}\nReasoning:"
    
    reasoning_1 = model.generate(prompt_1, task_type='init')
    all_documents.extend(docs_1)
    all_reasoning.append(reasoning_1)
    
    # Step 2-4: 추론 확장
    for step in range(2, num_steps):
        # 이전 추론을 쿼리로 사용해 새 문서 검색
        query = all_reasoning[-1]  # 마지막 추론
        new_docs = retriever.search(query, k=4)
        
        prompt = f"""
        Question: {question}
        Previous: {' '.join(all_reasoning)}
        New docs: {new_docs}
        Continue reasoning:
        """
        
        reasoning = model.generate(prompt, task_type='exp')
        all_documents.extend(new_docs)
        all_reasoning.append(reasoning)
        
        # "So the answer is" 나오면 조기 종료
        if "So the answer is" in reasoning:
            break
    
    # Step 5: 최종 답변
    prompt_final = f"""
    Question: {question}
    All reasoning: {' '.join(all_reasoning)}
    All documents: {' '.join(all_documents)}
    So the answer is:
    """
    
    final_answer = model.generate(prompt_final, task_type='agg')
    
    return {
        'question': question,
        'reasoning_steps': all_reasoning,
        'final_answer': final_answer,
        'documents_used': all_documents
    }
```

---

## 5️⃣ 실전 팁과 주의사항

### 💡 구현 시 핵심 포인트

```python
# 1. 배치 처리 시 패딩 주의
def collate_stepwise_batch(samples):
    """
    단계별로 다른 길이의 시퀀스 처리
    """
    # 각 태스크별로 그룹화
    init_samples = [s['first_step'] for s in samples if 'first_step' in s]
    exp_samples = []
    for s in samples:
        exp_samples.extend(s.get('mid_steps', []))
    agg_samples = [s['final_step'] for s in samples if 'final_step' in s]
    
    # 각각 패딩
    # ...

# 2. 메모리 최적화
def gradient_checkpointing_setup(model):
    """
    Gradient checkpointing으로 메모리 절약
    """
    model.gradient_checkpointing_enable()
    
# 3. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(inputs)
scaler.scale(loss).backward()
```

### ⚠️ 주의사항

1. **데이터 품질**: Teacher 모델의 추론이 틀리면 안 됨
2. **단계 수 조절**: 질문 복잡도에 따라 S 값 조정 필요
3. **Retriever 성능**: BM25보다 Dense retriever 사용 고려
4. **σ 초기값**: 너무 크거나 작으면 학습 불안정

---

## 📈 예상 결과

```python
# 학습 진행 모니터링
"""
Epoch 1/2
Batch 0: Loss=2.3451
  σ values - Init: 0.95, Exp: 1.12, Agg: 1.45
  
Batch 100: Loss=1.8234  
  σ values - Init: 0.72, Exp: 1.08, Agg: 1.67
  
# σ 값 해석:
# - Init이 가장 작음 = 가장 쉬운 태스크
# - Agg가 가장 큼 = 가장 어려운 태스크
"""
```

이렇게 구현하면 논문의 핵심 아이디어를 실제로 구현할 수 있습니다!