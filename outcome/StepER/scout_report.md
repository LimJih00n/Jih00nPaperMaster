# StepER: Step-wise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented LM - 정찰 보고서

## 🎯 Quick Scan
- **한 줄 요약**: 복잡한 질문에 답하기 위한 다단계 검색 증강 언어모델의 단계별 추론 능력을 향상시키는 지식 증류 방법
- **핵심 기여**: 단계별 데이터셋 구성과 난이도 인식 학습을 통한 효과적인 추론 능력 학습 프레임워크 StepER 제안
- **방법론**: Step-wise Knowledge Distillation + Reasoning Difficulty-Aware Training
- **실험 규모**: 3개 multi-hop QA 데이터셋 (2WikiMultiHopQA, HotpotQA, MuSiQue)에서 검증
- **성능 향상**: vanilla-KD 대비 평균 9.5% 정확도 향상, 8B 모델로 70B 교사 모델 성능 달성
- **관련성 점수**: 5/5 - 현재 LLM 경량화와 복잡한 추론 태스크 해결이 핵심 과제인 상황에서 매우 시의적절한 연구

## ⚡ 읽을 가치 판단
- **강점**: 
  - 단계별로 다른 추론 능력이 필요함을 명확히 정의 (초기화, 확장, 집계)
  - 8B 모델로 70B 성능 달성하는 뛰어난 확장성
  - 다양한 multi-step RAG 프레임워크에 적용 가능한 범용성
- **약점**: 
  - 교사 모델의 오류가 전파될 수 있는 한계
  - 추가 학습 데이터 구축 비용 필요
  - 단계별 추론 과정의 정확성 검증 방법 부재
- **추천 여부**: **정독 추천** - RAG 시스템 개선과 모델 경량화에 관심있는 연구자 필독