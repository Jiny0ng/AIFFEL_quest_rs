# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 박진용
- 리뷰어 : 임정민


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
          <img width="479" height="224" alt="image" src="https://github.com/user-attachments/assets/5bba17fd-69a3-42e6-8609-5fca52662e88" />
          <img width="666" height="671" alt="image" src="https://github.com/user-attachments/assets/e0c66a8b-70c5-4aca-8d28-a0320e740fb6" />

- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
          <img width="725" height="335" alt="image" src="https://github.com/user-attachments/assets/18e103ae-384a-4578-9255-800a547de484" />
          오차 최소화를 위해 시간을 바이닝하여 파생 변수로 사용하는 새로운 시도를 했다는점에서 가장 핵심적인 부분이라 생각합니다.  

- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
          <img width="873" height="321" alt="image" src="https://github.com/user-attachments/assets/f77e45fc-ef8f-41e7-aa1d-6299b92c489d" />
          시간뿐만 아니라 온도 등 다양한 피처를 바이닝하여 파생 변수로 사용함으로써 성능을 높이려는 시도가 인상 깊었습니다.
          이 과정에서 RMSE 수치를 기록하고 성능을 비교해 나가면서 원인 분석 등 문제 해결 과정을 상세히 잘 작성했습니다. 
        
- [X]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
          <img width="873" height="638" alt="image" src="https://github.com/user-attachments/assets/7c33b4ae-1248-4876-9a83-4284169a5945" />
        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
          <img width="848" height="1023" alt="image" src="https://github.com/user-attachments/assets/3a0a2b6c-a989-4550-9cbf-fc5851721139" />


# 회고(참고 링크 및 코드 개선)
```
importances = model_SGD.featureimportances를 통해 높은 가중치를 가진 피처를 확인할 수 있다는 새로운 인사이트를 얻게 되었습니다.
성능 향상 과정에서 어떤 문제점이 어떤 원인으로 발생했는지를 자세히 기록하셔서, 리뷰어 입장에서도 도움이 많이 되었습니다.
```
