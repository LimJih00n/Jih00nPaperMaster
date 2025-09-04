# Topic Coverage-based Demonstration Retrieval for In-Context Learning - 구현 가이드

## 🔍 단계별 미니 구현

### Step 1: 기본 토픽 예측기 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

class TopicPredictor(nn.Module):
    """경량 토픽 예측기 - 시연 임베딩을 토픽 분포로 변환"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_topics: int = 1000):
        super().__init__()
        self.num_topics = num_topics
        
        # 3-layer MLP as mentioned in paper
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_topics),
            nn.Sigmoid()  # Output probabilities for each topic
        )
        
    def forward(self, embeddings):
        # embeddings: [batch_size, input_dim] = [32, 768]
        topic_probs = self.layers(embeddings)  # [batch_size, num_topics]
        return topic_probs

# 예시 사용법
embedding_model = SentenceTransformer('all-mpnet-base-v2')
topic_predictor = TopicPredictor(input_dim=768, num_topics=1000)

# 샘플 시연들
demonstrations = [
    "Herbivores are animals that primarily eat plants",
    "Carnivores hunt and eat other animals for survival", 
    "Photosynthesis converts sunlight into chemical energy"
]

# 임베딩 생성
embeddings = embedding_model.encode(demonstrations, convert_to_tensor=True)
print(f"임베딩 차원: {embeddings.shape}")  # [3, 768]

# 토픽 분포 예측
with torch.no_grad():
    topic_distributions = topic_predictor(embeddings)
    print(f"토픽 분포 차원: {topic_distributions.shape}")  # [3, 1000]
    print(f"첫 번째 시연의 상위 5개 토픽: {torch.topk(topic_distributions[0], 5)}")
```

### Step 2: 토픽 커버리지 계산기 구현

```python
class TopicCoverageCalculator:
    """토픽 커버리지 기반 관련성 점수 계산"""
    
    def __init__(self, topic_predictor: TopicPredictor):
        self.topic_predictor = topic_predictor
        self.topical_knowledge = None  # t̂_LM - 모델의 토픽별 지식
        
    def estimate_topical_knowledge(self, demonstrations: List[str], 
                                 embeddings: torch.Tensor, 
                                 zero_shot_accuracies: List[float]):
        """논문의 수식 (5) 구현: 토픽별 모델 지식 평가"""
        
        with torch.no_grad():
            topic_distributions = self.topic_predictor(embeddings)  # [N, num_topics]
            
        # t̂_LM,t = (∑_d t̂_d,t · zero-shot(d)) / (∑_d t̂_d,t)
        zero_shot_tensor = torch.tensor(zero_shot_accuracies).unsqueeze(1)  # [N, 1]
        
        # 분자: weighted sum of zero-shot accuracies
        numerator = torch.sum(topic_distributions * zero_shot_tensor, dim=0)  # [num_topics]
        
        # 분모: sum of topic weights  
        denominator = torch.sum(topic_distributions, dim=0)  # [num_topics]
        
        # 0으로 나누기 방지
        denominator = torch.clamp(denominator, min=1e-8)
        
        self.topical_knowledge = numerator / denominator  # [num_topics]
        
        print(f"토픽별 모델 지식 범위: {self.topical_knowledge.min():.3f} ~ {self.topical_knowledge.max():.3f}")
        return self.topical_knowledge
    
    def compute_relevance_score(self, test_embedding: torch.Tensor, 
                              candidate_embedding: torch.Tensor) -> float:
        """논문의 수식 (6) 구현: 토픽 커버리지 기반 관련성 점수"""
        
        if self.topical_knowledge is None:
            raise ValueError("먼저 estimate_topical_knowledge()를 호출하세요!")
            
        with torch.no_grad():
            # 테스트 입력과 후보 시연의 토픽 분포 예측
            test_topics = self.topic_predictor(test_embedding.unsqueeze(0))[0]  # [num_topics]
            candidate_topics = self.topic_predictor(candidate_embedding.unsqueeze(0))[0]  # [num_topics]
            
            # r(x, d) = ⟨t̂_x ⊘ t̂_LM, t̂_d⟩
            knowledge_weighted_test = test_topics / torch.clamp(self.topical_knowledge, min=1e-8)
            relevance_score = torch.dot(knowledge_weighted_test, candidate_topics)
            
            return relevance_score.item()

# 예시 사용법
calculator = TopicCoverageCalculator(topic_predictor)

# 모의 zero-shot 정확도 (실제로는 LLM으로 측정)
zero_shot_scores = [0.85, 0.72, 0.91]  # 각 시연에 대한 zero-shot 성능

# 토픽별 모델 지식 추정
topical_knowledge = calculator.estimate_topical_knowledge(
    demonstrations, embeddings, zero_shot_scores
)

# 테스트 질문
test_question = "Non-human organisms that mainly consume plants are known as what?"
test_embedding = embedding_model.encode([test_question], convert_to_tensor=True)[0]

# 각 후보 시연에 대한 관련성 점수 계산
for i, demo in enumerate(demonstrations):
    score = calculator.compute_relevance_score(test_embedding, embeddings[i])
    print(f"시연 {i+1} 관련성 점수: {score:.3f}")
    print(f"내용: {demo[:50]}...")
    print()
```

### Step 3: 누적 토픽 커버리지 구현

```python
class CumulativeTopicCoverage:
    """논문의 수식 (7) 구현: 누적 토픽 커버리지로 다양성 보장"""
    
    def __init__(self, topic_predictor: TopicPredictor, embedding_model):
        self.topic_predictor = topic_predictor
        self.embedding_model = embedding_model
        self.selected_demonstrations = []
        self.selected_embeddings = []
        
    def update_coverage(self, new_demonstration: str, new_embedding: torch.Tensor) -> torch.Tensor:
        """새로운 시연 추가 시 커버리지 업데이트"""
        
        if len(self.selected_demonstrations) == 0:
            # 첫 번째 시연인 경우
            self.selected_demonstrations.append(new_demonstration)
            self.selected_embeddings.append(new_embedding)
            
            with torch.no_grad():
                return self.topic_predictor(new_embedding.unsqueeze(0))[0]
        
        # 이전 시연들의 평균 임베딩 계산
        prev_embeddings = torch.stack(self.selected_embeddings)  # [num_prev, embed_dim]
        prev_mean_embedding = torch.mean(prev_embeddings, dim=0)  # [embed_dim]
        
        # 새 시연 포함한 전체 평균 임베딩
        all_embeddings = torch.stack(self.selected_embeddings + [new_embedding])
        combined_mean_embedding = torch.mean(all_embeddings, dim=0)
        
        with torch.no_grad():
            # 이전 커버리지와 새 커버리지 계산
            prev_coverage = self.topic_predictor(prev_mean_embedding.unsqueeze(0))[0]
            combined_coverage = self.topic_predictor(combined_mean_embedding.unsqueeze(0))[0]
            
            # t̂_d ← (t̂_{d∪D'_x} - t̂_{D'_x})
            incremental_coverage = torch.clamp(combined_coverage - prev_coverage, min=0)
            
        # 선택된 시연 목록 업데이트
        self.selected_demonstrations.append(new_demonstration)
        self.selected_embeddings.append(new_embedding)
        
        return incremental_coverage
    
    def get_coverage_diversity_score(self) -> float:
        """현재 선택된 시연들의 다양성 점수"""
        if len(self.selected_demonstrations) < 2:
            return 0.0
            
        # 모든 시연 쌍 간의 토픽 분포 유사도 계산
        with torch.no_grad():
            all_embeddings = torch.stack(self.selected_embeddings)
            topic_distributions = self.topic_predictor(all_embeddings)
            
            # 코사인 유사도 계산
            similarities = F.cosine_similarity(
                topic_distributions.unsqueeze(1), 
                topic_distributions.unsqueeze(0), 
                dim=2
            )
            
            # 대각선 제외한 평균 유사도 (다양성의 역지표)
            mask = ~torch.eye(len(self.selected_demonstrations), dtype=bool)
            avg_similarity = similarities[mask].mean()
            
            # 다양성 점수 = 1 - 평균 유사도
            diversity_score = 1.0 - avg_similarity.item()
            
        return diversity_score

# 예시 사용법
coverage_tracker = CumulativeTopicCoverage(topic_predictor, embedding_model)

# 순차적 시연 선택 시뮬레이션
candidate_demos = [
    "Herbivores are animals that primarily eat plants",
    "Carnivores hunt and eat other animals for survival",
    "Omnivores eat both plants and animals",
    "Photosynthesis is the process plants use to make food",
    "Food chains show energy flow in ecosystems"
]

# 각 시연의 누적 기여도 계산
for i, demo in enumerate(candidate_demos):
    demo_embedding = embedding_model.encode([demo], convert_to_tensor=True)[0]
    incremental_coverage = coverage_tracker.update_coverage(demo, demo_embedding)
    diversity_score = coverage_tracker.get_coverage_diversity_score()
    
    print(f"\n시연 {i+1} 추가 후:")
    print(f"내용: {demo}")
    print(f"증분 커버리지 합계: {incremental_coverage.sum():.3f}")
    print(f"다양성 점수: {diversity_score:.3f}")
    print(f"상위 토픽 기여도: {torch.topk(incremental_coverage, 3)}")
```

### Step 4: 전체 TopicK 시스템 통합

```python
class TopicKRetriever:
    """TopicK 전체 시스템 구현"""
    
    def __init__(self, embedding_model_name: str = 'all-mpnet-base-v2', 
                 num_topics: int = 1000):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.topic_predictor = TopicPredictor(input_dim=768, num_topics=num_topics)
        self.coverage_calculator = None
        self.is_trained = False
        
    def train_topic_predictor(self, demonstrations: List[str], 
                            topic_labels: List[Dict[str, float]], 
                            epochs: int = 100, lr: float = 1e-4):
        """토픽 예측기 학습 - 논문의 수식 (4) 구현"""
        
        # 시연들을 임베딩으로 변환
        embeddings = self.embedding_model.encode(demonstrations, convert_to_tensor=True)
        
        # 토픽 레이블을 텐서로 변환
        topic_targets = []
        for labels in topic_labels:
            target = torch.zeros(self.topic_predictor.num_topics)
            for topic_idx, weight in labels.items():
                target[int(topic_idx)] = weight
            topic_targets.append(target)
        
        targets = torch.stack(topic_targets)  # [num_demos, num_topics]
        
        # 학습 설정
        optimizer = torch.optim.Adam(self.topic_predictor.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        self.topic_predictor.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 순전파
            predictions = self.topic_predictor(embeddings)
            
            # 이진 교차 엔트로피 손실 (논문 수식 4)
            loss = criterion(predictions, targets)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print("토픽 예측기 학습 완료!")
        
    def retrieve_demonstrations(self, test_input: str, 
                              candidate_pool: List[str], 
                              k: int = 8,
                              lambda_weight: float = 0.5) -> List[Tuple[str, float]]:
        """TopicK 메인 검색 알고리즘"""
        
        if not self.is_trained:
            raise ValueError("토픽 예측기를 먼저 학습시키세요!")
        
        # 임베딩 생성
        test_embedding = self.embedding_model.encode([test_input], convert_to_tensor=True)[0]
        candidate_embeddings = self.embedding_model.encode(candidate_pool, convert_to_tensor=True)
        
        # 토픽 커버리지 계산기 초기화
        if self.coverage_calculator is None:
            # 모의 zero-shot 정확도 (실제 구현에서는 LLM으로 측정)
            mock_zero_shot = np.random.uniform(0.6, 0.9, len(candidate_pool))
            self.coverage_calculator = TopicCoverageCalculator(self.topic_predictor)
            self.coverage_calculator.estimate_topical_knowledge(
                candidate_pool, candidate_embeddings, mock_zero_shot
            )
        
        # 누적 커버리지 추적기 초기화
        coverage_tracker = CumulativeTopicCoverage(self.topic_predictor, self.embedding_model)
        
        selected_indices = []
        remaining_indices = list(range(len(candidate_pool)))
        
        # 반복적 선택 (k개까지)
        for step in range(k):
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # 토픽 커버리지 기반 관련성 점수
                topic_score = self.coverage_calculator.compute_relevance_score(
                    test_embedding, candidate_embeddings[idx]
                )
                
                # 의미적 유사도 점수
                semantic_score = F.cosine_similarity(
                    test_embedding.unsqueeze(0), 
                    candidate_embeddings[idx].unsqueeze(0)
                ).item()
                
                # 최종 점수: r(x,d) + λ * cos(e_x, e_d)
                total_score = topic_score + lambda_weight * semantic_score
                
                # 누적 커버리지 고려 (2단계부터)
                if step > 0:
                    # 현재까지 선택된 시연들과의 토픽 중복 페널티 추가
                    coverage_penalty = self._calculate_coverage_penalty(
                        idx, selected_indices, candidate_embeddings, coverage_tracker
                    )
                    total_score -= coverage_penalty
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
            
            # 최고 점수 시연 선택
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                
                # 커버리지 업데이트
                coverage_tracker.update_coverage(
                    candidate_pool[best_idx], 
                    candidate_embeddings[best_idx]
                )
                
                print(f"Step {step+1}: 선택된 시연 {best_idx} (점수: {best_score:.3f})")
                print(f"내용: {candidate_pool[best_idx][:60]}...")
                print()
        
        # 결과 반환
        results = [(candidate_pool[idx], self.coverage_calculator.compute_relevance_score(
                    test_embedding, candidate_embeddings[idx])) 
                   for idx in selected_indices]
        
        return results
    
    def _calculate_coverage_penalty(self, candidate_idx: int, 
                                   selected_indices: List[int], 
                                   embeddings: torch.Tensor,
                                   coverage_tracker: CumulativeTopicCoverage) -> float:
        """토픽 중복에 대한 페널티 계산"""
        
        if len(selected_indices) == 0:
            return 0.0
        
        with torch.no_grad():
            # 후보 시연의 토픽 분포
            candidate_topics = self.topic_predictor(embeddings[candidate_idx].unsqueeze(0))[0]
            
            # 이미 선택된 시연들의 평균 토픽 분포
            selected_embeddings = embeddings[selected_indices]
            selected_topics = self.topic_predictor(selected_embeddings)
            avg_selected_topics = torch.mean(selected_topics, dim=0)
            
            # 토픽 오버랩 계산 (코사인 유사도)
            overlap = F.cosine_similarity(
                candidate_topics.unsqueeze(0), 
                avg_selected_topics.unsqueeze(0)
            ).item()
            
            # 오버랩이 높을수록 높은 페널티
            penalty = max(0, overlap - 0.3) * 0.5  # 임계값 0.3, 페널티 스케일 0.5
            
        return penalty

# 전체 시스템 사용 예시
def main_example():
    """TopicK 전체 시스템 사용 예시"""
    
    # 시스템 초기화
    retriever = TopicKRetriever(num_topics=100)  # 예시용으로 작은 토픽 수
    
    # 후보 시연 풀
    candidate_pool = [
        "Herbivores are animals that primarily eat plants and vegetation for energy.",
        "Carnivores are predators that hunt and consume other animals as their main food source.", 
        "Omnivores have a diverse diet that includes both plant and animal matter.",
        "Photosynthesis is the biological process by which plants convert sunlight into chemical energy.",
        "Cellular respiration breaks down glucose to produce ATP energy in living organisms.",
        "Food chains illustrate the flow of energy from producers to various levels of consumers.",
        "Decomposers like bacteria and fungi break down dead organic matter in ecosystems.",
        "Symbiosis describes close relationships between different species that can be beneficial.",
        "Evolution explains how species change and adapt over time through natural selection.",
        "Biodiversity refers to the variety of life forms found in different ecosystems."
    ]
    
    # 모의 토픽 레이블 생성 (실제로는 topic mining으로 생성)
    topic_labels = []
    for i in range(len(candidate_pool)):
        # 각 시연마다 3-5개의 관련 토픽에 가중치 부여
        labels = {}
        num_topics = np.random.randint(3, 6)
        topic_indices = np.random.choice(100, num_topics, replace=False)
        weights = np.random.uniform(0.3, 1.0, num_topics)
        
        for topic_idx, weight in zip(topic_indices, weights):
            labels[str(topic_idx)] = weight
        
        topic_labels.append(labels)
    
    # 토픽 예측기 학습
    print("토픽 예측기 학습 중...")
    retriever.train_topic_predictor(candidate_pool, topic_labels, epochs=50)
    
    # 테스트 질문
    test_question = "Non-human organisms that mainly consume plants are known as what?"
    
    # 시연 검색 실행
    print(f"\n테스트 질문: {test_question}")
    print("\nTopicK로 선택된 시연들:")
    print("=" * 60)
    
    selected_demos = retriever.retrieve_demonstrations(
        test_question, candidate_pool, k=5
    )
    
    for i, (demo, score) in enumerate(selected_demos):
        print(f"{i+1}. 관련성 점수: {score:.3f}")
        print(f"   내용: {demo}")
        print()

if __name__ == "__main__":
    main_example()
```

## 📊 성능 체크포인트

```python
def evaluate_topic_coverage(selected_demonstrations: List[str], 
                          topic_predictor: TopicPredictor,
                          embedding_model) -> Dict[str, float]:
    """선택된 시연들의 토픽 커버리지 품질 평가"""
    
    embeddings = embedding_model.encode(selected_demonstrations, convert_to_tensor=True)
    
    with torch.no_grad():
        topic_distributions = topic_predictor(embeddings)  # [num_demos, num_topics]
        
        # 1. 토픽 커버리지 (활성화된 토픽의 수)
        avg_distribution = torch.mean(topic_distributions, dim=0)
        active_topics = (avg_distribution > 0.1).sum().item()  # 임계값 0.1
        coverage_ratio = active_topics / topic_distributions.shape[1]
        
        # 2. 토픽 다양성 (시연 간 토픽 분포의 다양성)
        pairwise_similarities = F.cosine_similarity(
            topic_distributions.unsqueeze(1),
            topic_distributions.unsqueeze(0), 
            dim=2
        )
        # 대각선 제외
        mask = ~torch.eye(len(selected_demonstrations), dtype=bool)
        avg_similarity = pairwise_similarities[mask].mean().item()
        diversity_score = 1.0 - avg_similarity
        
        # 3. 토픽 엔트로피 (분포의 균등성)
        topic_entropy = -torch.sum(avg_distribution * torch.log(avg_distribution + 1e-8)).item()
        
        # 4. 토픽 집중도 (상위 토픽들이 차지하는 비중)
        sorted_topics, _ = torch.sort(avg_distribution, descending=True)
        top_k_ratio = sorted_topics[:10].sum().item()  # 상위 10개 토픽의 비중
        
    return {
        "topic_coverage_ratio": coverage_ratio,
        "topic_diversity": diversity_score, 
        "topic_entropy": topic_entropy,
        "top_k_concentration": top_k_ratio
    }

# 성능 모니터링 예시
def monitor_performance():
    """학습 및 검색 성능 모니터링"""
    
    print("=== 성능 체크포인트 ===")
    
    # 토픽 예측기 성능 체크
    sample_demos = [
        "Herbivores eat plants",
        "Carnivores eat meat", 
        "Plants use photosynthesis"
    ]
    
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    topic_predictor = TopicPredictor(num_topics=100)
    
    metrics = evaluate_topic_coverage(sample_demos, topic_predictor, embedding_model)
    
    print("토픽 커버리지 메트릭:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # 예상 성능 기준점
    print("\n예상 성능 기준:")
    print("  topic_coverage_ratio: > 0.15 (15% 이상의 토픽 활성화)")
    print("  topic_diversity: > 0.3 (30% 이상의 다양성)")
    print("  topic_entropy: > 2.0 (적절한 분포 균등성)")
    print("  top_k_concentration: < 0.8 (과도한 집중 방지)")

if __name__ == "__main__":
    monitor_performance()
```

## 🎯 최적화 팁과 실무 고려사항

### 1. 메모리 최적화
```python
# 대용량 후보 풀 처리를 위한 배치 처리
def batch_process_embeddings(texts: List[str], batch_size: int = 32):
    """메모리 효율적인 임베딩 생성"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)
```

### 2. 속도 최적화
```python
# 상위 300개 후보로 사전 필터링 (논문에서 언급)
def fast_prefiltering(test_embedding: torch.Tensor, 
                     candidate_embeddings: torch.Tensor, 
                     top_k: int = 300) -> torch.Tensor:
    """빠른 사전 필터링으로 검색 공간 축소"""
    similarities = F.cosine_similarity(
        test_embedding.unsqueeze(0), candidate_embeddings
    )
    _, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
    return top_indices
```

### 3. 하이퍼파라미터 튜닝 가이드
```python
hyperparams = {
    "learning_rate": [1e-5, 1e-4, 1e-3],          # 토픽 예측기 학습률
    "lambda_weight": [0.1, 0.5, 1.0],             # 의미적 유사도 가중치  
    "coverage_threshold": [0.1, 0.2, 0.3],        # 토픽 활성화 임계값
    "diversity_penalty": [0.3, 0.5, 0.8]          # 다양성 페널티 강도
}
```

이 구현 가이드를 통해 TopicK의 핵심 아이디어를 실제 작동하는 코드로 구현할 수 있습니다!