# Backend - Navi Mumbai House Price Predictor API

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates model.pkl)
python train_model.py

# Start development server
uvicorn main:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/locations` | List supported locations |
| GET | `/api/model-info` | Model metrics & feature importance |
| POST | `/api/predict` | Predict house price |

## Deployment (Render)

1. Push this `backend` directory to a Git repository
2. Connect to Render and select "Blueprint"
3. Render will use `render.yaml` for configuration
4. The build step trains the model automatically

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:3000` |
| `PORT` | Server port (set by Render) | `8000` |
