# 2조 힌지 최대응력 예측

힌지 치수 `a`, `b`, `t`를 입력하면 최대 폰 미세스 응력을 예측하는 MLP 모델입니다. 입력은 mm, 출력은 MPa입니다. 1조 FEM 결과 330개를 학습 230개, 검증 50개, 테스트 50개로 나눠 MSE와 MAE 손실 함수를 비교했습니다. 검증에서 선택한 MSE 모델의 테스트 평균 절대오차는 0.250 MPa, 최대 절대오차는 0.936 MPa였습니다.

- [1조 원본 CSV](../data/hinge_330_full.csv)
- [2조 학습 결과 폴더](results/team2-fem/)
- [두 모델의 테스트 오차](results/team2-fem/test_metrics.json)

## 파일별 역할

| 파일 | 하는 일 |
| --- | --- |
| [`data.py`](data.py) | FEM CSV를 읽고 치수와 응력을 mm, MPa로 변환 |
| [`train.py`](train.py) | 같은 데이터 분할과 초기 가중치로 MSE와 MAE 손실 함수 비교 |
| [`evaluate.py`](evaluate.py) | 학습 곡선, 테스트 오차, 설계별 예측값 저장 |
| [`model.py`](model.py) | 모델 정의와 저장된 모델 불러오기 |
| [`make_demo_data.py`](make_demo_data.py) | 실행 확인용 더미 CSV 생성 |

## FEM 데이터로 다시 학습하기

저장소 최상위 폴더에서 실행합니다. `--output-dir`에는 아직 만들어지지 않은 결과 폴더를 지정합니다.

```powershell
python -m pip install torch matplotlib
python -m deep_learning.train --csv data/hinge_330_full.csv --feature-columns a_mm b_mm t_mm --stress-column sigma_max_MPa --length-unit mm --stress-unit MPa --output-dir deep_learning/results/team2-rerun
python -m deep_learning.evaluate --csv data/hinge_330_full.csv --run-dir deep_learning/results/team2-rerun --allowable-mpa 236.667
```

## 더미 데이터로 실행 확인하기

```powershell
python -m deep_learning.make_demo_data deep_learning/demo/demo_synthetic.csv
python -m deep_learning.train --csv deep_learning/demo/demo_synthetic.csv --stress-unit MPa --output-dir deep_learning/demo/results
python -m deep_learning.evaluate --csv deep_learning/demo/demo_synthetic.csv --run-dir deep_learning/demo/results
```

## 3조에서 모델 사용하기

학습 결과 폴더의 `run.json`에서 `selected_model`을 확인합니다. [`model.py`](model.py)의 `load_surrogate()`로 해당 `.pt` 파일을 불러온 뒤 `model.predict_stress(a, b, t)`를 호출합니다. 입력은 mm, 예측값은 MPa입니다.
