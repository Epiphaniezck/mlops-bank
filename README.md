# 🏦 Bank Marketing MLOps Project

Projet MLOps complet : prédire si un client va souscrire à un dépôt à terme.

## Dataset
**Bank Marketing** (UCI ML Repository) — 4 521 clients, 16 features, cible binaire `y` (yes/no).

## Stack
Python 3.11 · scikit-learn · pandas · FastAPI · Docker · GitHub Actions

## Lancer en local

```bash
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```

## Endpoints

| Méthode | URL       | Description                          |
|---------|-----------|--------------------------------------|
| GET     | /health   | Santé de l'API + état du modèle      |
| POST    | /predict  | Prédiction de souscription           |
| GET     | /docs     | Documentation Swagger UI             |

### Exemple POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42, "job": "management", "marital": "married",
    "education": "tertiary", "default": "no", "balance": 2143,
    "housing": "yes", "loan": "no", "contact": "cellular",
    "day": 5, "month": "may", "duration": 261,
    "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"
  }'
```

### Réponse

```json
{
  "will_subscribe": false,
  "label": "no",
  "probability_yes": 0.1823,
  "probability_no": 0.8177
}
```

## Docker

```bash
docker build -t bank-marketing-api .
docker run -p 8000:8000 bank-marketing-api
```

## CI/CD (GitHub Actions)

| Branche      | Jobs                                              |
|-------------|---------------------------------------------------|
| `feature/*` | Installer deps → Entraîner le modèle              |
| `develop`   | Entraîner → Build image Docker → Push Docker Hub  |

### Secrets GitHub requis
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
