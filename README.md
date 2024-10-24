# 한밭대학교 컴퓨터공학과 이용빈식 캡스톤디자인팀

**팀 구성**
- 20191120 이용빈
- 20191780 육정훈

## <u>Teamate</u> Project Background
- ### 필요성
  - Grammarly 등 LLM을 활용한 글쓰기 교정 및 평가 서비스는 외국어 대상, 한국어를 지원하지 않음
  - 한국어 대상 서비스는 대부분 rule-base로 교정 능력이 떨어지는 등 여러 단점을 가짐
- ### 기존 해결책의 문제점
  - 한국어에는 다양한 문법적 예외가 존재하여, rule-base만으로는 대응에 한계가 있음
  - rule-base는 글의 전후관계를 이해한 교정이 아니므로, 문맥을 파악하지 못함
  
## System Design
  - ### System Requirements
    - (직접 학습 및 추론 시) GPU(At least A6000)
    - (서비스 이용 시) 인터넷 환경
    
## Case Study
  - ### Description
    - KoBART를 이용한 GEC 연구 한밭대학교 컴퓨터공학과 이용빈식 캡스톤디자인팀

**팀 구성**
- 20191120 이용빈
- 20191780 육정훈

## <u>Teamate</u> Project Background
- ### 필요성
  - Grammarly 등 LLM을 활용한 글쓰기 교정 및 평가 서비스는 외국어 대상, 한국어를 지원하지 않음
  - 한국어 대상 서비스는 대부분 rule-base로 교정 능력이 떨어지는 등 여러 단점을 가짐
- ### 기존 해결책의 문제점
  - 한국어에는 다양한 문법적 예외가 존재하여, rule-base만으로는 대응에 한계가 있음
  - rule-base는 글의 전후관계를 이해한 교정이 아니므로, 문맥을 파악하지 못함
  
## System Design
  - ### System Requirements
    - (직접 학습 및 추론 시) GPU(At least A6000)
    - (서비스 이용 시) 인터넷 환경
    
## Case Study
  - ### Description
    - KoBART를 이용한 GEC 연구 및 데이터셋 제공
      - link: https://github.com/soyoung97/Standard_Korean_GEC?tab=readme-ov-file

## Conclusion
  - ### 후속 연구 기여: 기존 한국어 AWE 데이터셋을 GEC에도 사용 가능하도록 확장하고, 두 Task를 동시에 학습시켰을 때 서로 긍정적인 영향을 미친다는 것을 검증하여, 이를 포함한 AWE+GEC 데이터셋과 모델을 공개한다.
  - ### 향상된 글쓰기 평가 및 교정 시스템 개발: LLM이 글의 전체 문맥을 반영하여 맞춤법 교정 및 평가를 제공하여, 기존 rule-base 시스템에 비해 고품질의 맞춤범 교정 및 글쓰기 평가 제공 
  
## Project Outcome
- ### 2024 한국컴퓨터종합학술대회 "모두를 위한 한국어 맞춤법 교정 모델"
  - ### 우수발표논문상 수상
- ### KCI "한국인-외국인 모두를 위한 한국어 맞춤법 교정 모델" 심사중
- ### NAACL 심사중
