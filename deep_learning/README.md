# 2조 힌지 최대응력 예측

1조에서 받은 FEM 결과로 모델을 학습합니다. 힌지 치수 `a`, `b`, `t`를 넣으면 최대응력을 예측하고, 3조는 이 값으로 설계 후보를 찾습니다.

현재 더미 데이터 330개로 학습, 평가, 모델 저장과 다시 불러오기를 확인했습니다. 실제 예측 오차는 1조 데이터가 들어오면 계산합니다.

## 파일별 역할

| 파일 | 하는 일 |
| --- | --- |
| [`data.py`](data.py) | FEM CSV를 읽고 치수와 응력을 mm, MPa로 변환 |
| [`train.py`](train.py) | 같은 데이터 분할과 초기 가중치로 MSE와 MAE 손실 함수 비교 |
| [`evaluate.py`](evaluate.py) | 학습 곡선, 테스트 오차, 설계별 예측값 저장 |
| [`model.py`](model.py) | 모델 정의와 저장된 모델 불러오기 |
| [`make_demo_data.py`](make_demo_data.py) | 실행 확인용 더미 CSV 생성 |

## 실행 방법

저장소 최상위 폴더에서 실행합니다.

```powershell
python -m pip install torch matplotlib
python -m deep_learning.make_demo_data data/demo_synthetic.csv
python -m deep_learning.train --csv data/demo_synthetic.csv --stress-unit MPa --output-dir results/team2-demo
python -m deep_learning.evaluate --csv data/demo_synthetic.csv --run-dir results/team2-demo
```

실제 데이터가 오면 `--csv`를 1조 파일로 바꾸고 새 결과 폴더를 지정합니다. 기본 열은 `a,b,t,sigma_max`입니다. 이름이 다르면 `--feature-columns`와 `--stress-column`으로 맞추고, 치수가 m이면 `--length-unit m`, 응력이 Pa이면 `--stress-unit Pa`를 사용합니다.

## 3조에서 사용하기

결과 폴더의 `run.json`에서 검증 오차로 선택한 모델을 확인합니다. `load_surrogate()`로 해당 `.pt` 파일을 불러온 뒤 `model.predict_stress(a, b, t)`를 호출하면 됩니다. 입력은 mm, 예측값은 MPa입니다.
