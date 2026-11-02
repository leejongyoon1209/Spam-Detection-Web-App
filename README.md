# Spam Message Detection Web App

영어 SMS 메시지를 입력하면 머신러닝 모델이 `Spam`(스팸) 또는 `Ham`(정상 메시지)으로 분류하는 로컬 Flask 웹 애플리케이션입니다.

## 주요 기능

- 영어 SMS 메시지 입력
- Spam / Ham 판별 결과 표시
- 스팸일 확률 표시
- 빈 입력값 오류 처리
- 머신러닝 예측 결과가 항상 정확하지 않을 수 있다는 안내

## 사용한 기술

- Python 3.10 이상
- Flask
- scikit-learn
- pandas
- joblib
- requests
- TF-IDF 텍스트 벡터화 + Logistic Regression 분류기

## 설치 방법

PowerShell에서 `upload` 폴더로 이동한 후 실행합니다.

```powershell
python -m pip install -r requirements.txt
```

## 실행 방법

처음 실행할 때 UCI SMS Spam Collection 데이터셋을 다운로드하고 모델을 학습합니다.

```powershell
python train_model.py
python app.py
```

그 다음 브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:5000
```

`train_model.py`는 학습 후 `data/`와 `model/` 폴더를 생성합니다. 이 폴더들은 실행 중 생성되는 파일이므로 GitHub 업로드 대상에서 제외합니다.

## Spam / Ham 판별 방법

학습 데이터의 SMS 문장을 TF-IDF 방식으로 숫자 벡터로 변환한 뒤 Logistic Regression 모델을 학습합니다. 사용자가 입력한 문장도 같은 방식으로 변환하여 모델이 `spam` 또는 `ham` 중 하나를 예측합니다. 표시되는 확률과 결과는 통계적 머신러닝 예측이므로 100% 정확하지 않을 수 있습니다.

## 프로젝트 구조

```text
upload/
├─ app.py
├─ train_model.py
├─ requirements.txt
├─ .gitignore
├─ README.md
├─ templates/
│  └─ index.html
└─ static/
   └─ style.css
```
