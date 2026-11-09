# Spam Message Detection Web App

UCI SMS Spam Collection 데이터셋으로 TF-IDF + Logistic Regression을 학습하고 Flask로 제공하는 로컬 웹앱입니다.

## PowerShell 실행

```powershell
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

브라우저에서 `http://127.0.0.1:5000`을 열어 사용합니다. 평가 결과는 `model/metrics.json`에 저장됩니다.

검증된 학습 성능: accuracy 0.9821, spam precision 0.9388, spam recall 0.9262.

## Docker 실행

Docker 이미지 빌드 시 `train_model.py`가 실행되어 모델이 이미지 안에 생성됩니다.

```powershell
docker build -t spam-detection-web-app:local .
docker run --rm -p 5000:5000 spam-detection-web-app:local
```

브라우저에서 `http://127.0.0.1:5000`을 엽니다.

## GitHub Actions / Docker Hub

`main` 브랜치에 push하면 `.github/workflows/ci.yml`이 다음 이미지로 빌드하고 Docker Hub에 업로드합니다.

```text
DOCKERHUB_USERNAME/spam-detection-web-app:latest
```

GitHub 저장소 Secrets에 `DOCKERHUB_USERNAME`과 `DOCKERHUB_TOKEN`을 등록해야 합니다.

## 간단한 모델 개선

`train_model.py`에는 결제 오류·청구서·카드 인증과 관련된 현실적인 피싱 메시지 16개를 소량 보강 데이터로 추가했습니다. 기존 TF-IDF + Logistic Regression 구조는 유지하며, 재학습할 때 원본 데이터와 보강 데이터를 함께 사용합니다. 평가용 비교 문장은 보강 데이터에 포함하지 않습니다.

## 구조

- `train_model.py`: 데이터 다운로드, 학습/평가, 모델 저장
- `app.py`: 저장된 모델을 로드하는 Flask 서버
- `templates/index.html`, `static/style.css`: 웹 UI
- `data/`: 원본 데이터셋(학습 시 생성)
- `model/`: `spam_classifier.joblib`, `metrics.json`(학습 시 생성)
