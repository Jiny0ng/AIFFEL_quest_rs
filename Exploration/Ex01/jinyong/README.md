time_bin을 통해 time을 3구간으로 바이닝해 수행하였을때
성능이 128까지 증가함을 확인

모델을 SGDRegression을 이용해봤지만 
같은 에러수치가 되었다.


temp_bin을 만들어 
temp<10
10<= temp <27
else
로 수행시 131로 에러가 증가함

리그레션의 경우 model.coef_ 를 통해 피처의 기여도를 확인할 수 있고
트리기반 모델의 경우
importances = model_SGD.feature_importances_
을 통해 높은 가중치를 가진 피처를 확인할 수 있다.


