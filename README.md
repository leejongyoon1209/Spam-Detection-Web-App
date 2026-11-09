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

## 구조

- `train_model.py`: 데이터 다운로드, 학습/평가, 모델 저장
- `app.py`: 저장된 모델을 로드하는 Flask 서버
- `templates/index.html`, `static/style.css`: 웹 UI
- `data/`: 원본 데이터셋(학습 시 생성)
- `model/`: `spam_classifier.joblib`, `metrics.json`(학습 시 생성)
