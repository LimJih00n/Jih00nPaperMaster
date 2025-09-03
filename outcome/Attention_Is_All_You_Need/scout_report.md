# Attention Is All You Need - 정찰 보고서

## 🎯 Quick Scan
- **한 줄 요약**: Transformer 아키텍처를 통해 RNN/CNN 없이 순수 Attention만으로 시퀀스 처리 문제를 해결하는 패러다임 전환
- **핵심 기여**: Self-Attention 메커니즘만으로 구성된 Transformer 모델 최초 제안
- **방법론**: Multi-Head Attention + Positional Encoding + Feed-Forward Networks
- **실험 규모**: WMT 2014 English-German (450만 문장 쌍), English-French (3600만 문장)
- **성능 향상**: BLEU 41.8 (EN-DE), 38.1 (EN-FR) - 당시 최고 성능 2.0 BLEU 포인트 초과
- **관련성 점수**: 5/5 - 현대 AI의 기초가 된 혁명적 논문

## ⚡ 읽을 가치 판단
- **강점**: 
  - 병렬 처리 가능으로 학습 속도 획기적 개선 (RNN 대비)
  - 장거리 의존성 문제를 직접적으로 해결
  - 단순하면서도 강력한 아키텍처로 다양한 태스크 적용 가능
  - 해석 가능한 Attention 가중치 시각화 가능

- **약점**: 
  - 시퀀스 길이 제곱에 비례하는 메모리 복잡도 O(n²)
  - 위치 정보를 위한 별도의 Positional Encoding 필요
  - 당시 기준으로 대규모 학습 데이터와 컴퓨팅 자원 요구

- **추천 여부**: **정독 필수** - NLP뿐 아니라 전체 AI 분야의 기초 논문