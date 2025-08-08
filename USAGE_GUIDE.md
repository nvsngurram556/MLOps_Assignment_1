# California Housing ML API - Usage Guide

This guide provides step-by-step instructions for using the complete MLOps pipeline.

## 🚀 Quick Start

### 1. Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (for local development)
- Git

### 2. Clone and Setup

```bash
git clone <your-repo-url>
cd california-housing-mlops
```

### 3. Deploy Locally

```bash
./scripts/deploy.sh local
```

This will start all services:
- ML API on port 8000
- MLflow on port 5001  
- Prometheus on port 9090
- Grafana on port 3000

## 📋 Detailed Setup Guide

### Option 1: Docker Deployment (Recommended)

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Check service health:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f ml-api
   ```

### Option 2: Local Development

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data and train models:**
   ```bash
   python src/models/train.py
   ```

4. **Start MLflow server:**
   ```bash
   mlflow server --host 0.0.0.0 --port 5001
   ```

5. **Start API server:**
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

## 🎯 Using the API

### 1. Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "MedInc": 8.3014,
         "HouseAge": 21.0,
         "AveRooms": 6.238137,
         "AveBedrms": 0.971880,
         "Population": 2401.0,
         "AveOccup": 2.109842,
         "Latitude": 37.86,
         "Longitude": -122.22
       }
     }'
```

**Response:**
```json
{
  "prediction": 4.526,
  "model_version": "random_forest",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 2. Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "features_list": [
         {
           "MedInc": 8.3014,
           "HouseAge": 21.0,
           "AveRooms": 6.238137,
           "AveBedrms": 0.971880,
           "Population": 2401.0,
           "AveOccup": 2.109842,
           "Latitude": 37.86,
           "Longitude": -122.22
         },
         {
           "MedInc": 5.6431,
           "HouseAge": 18.0,
           "AveRooms": 5.817352,
           "AveBedrms": 1.073446,
           "Population": 2775.0,
           "AveOccup": 2.611321,
           "Latitude": 39.78,
           "Longitude": -121.97
         }
       ]
     }'
```

### 3. Check Model Information

```bash
curl http://localhost:8000/model/info
```

### 4. Health Check

```bash
curl http://localhost:8000/health
```

### 5. API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## 🔬 Model Training and Experiments

### 1. Train Models with MLflow Tracking

```bash
python src/models/train.py
```

This will:
- Load and preprocess the California Housing dataset
- Train multiple models (Linear Regression, Random Forest, etc.)
- Track experiments in MLflow
- Select and register the best model

### 2. View Experiments

Visit http://localhost:5001 to access the MLflow UI and view:
- Experiment runs
- Model metrics
- Model artifacts
- Model registry

### 3. Manual Model Retraining

```bash
python src/models/retrain.py --manual
```

## 📊 Monitoring and Observability

### 1. Prometheus Metrics

- **URL:** http://localhost:9090
- **Metrics available:**
  - `prediction_requests_total` - Total prediction requests
  - `prediction_duration_seconds` - Prediction latency
  - `prediction_errors_total` - Total prediction errors

### 2. Grafana Dashboard

- **URL:** http://localhost:3000
- **Login:** admin/admin
- **Pre-configured dashboard** for ML API monitoring

### 3. API Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

## 🧪 Testing

### 1. Run Unit Tests

```bash
pytest tests/ -v
```

### 2. Run Tests with Coverage

```bash
pytest tests/ --cov=src/ --cov-report=html
```

### 3. Code Quality Checks

```bash
# Linting
flake8 src/

# Code formatting
black src/

# Import sorting
isort src/
```

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline automatically:

1. **On Pull Request:**
   - Runs linting and tests
   - Checks code formatting
   - Security scanning

2. **On Push to Main:**
   - Builds and pushes Docker image
   - Deploys to staging
   - Runs model training
   - Uploads artifacts

### Required Secrets

Set these in your GitHub repository:
- `DOCKER_HUB_USERNAME`
- `DOCKER_HUB_TOKEN`
- `MLFLOW_TRACKING_URI` (optional)

## 🔄 Automated Model Retraining

### 1. Start Retraining Service

```bash
python src/models/retrain.py --service
```

This service will:
- Check for performance degradation every 6 hours
- Check for data drift
- Retrain models automatically when needed
- Schedule weekly retraining

### 2. Trigger Manual Retraining via API

```bash
curl -X POST http://localhost:8000/model/retrain
```

## 📈 Feature Engineering

The pipeline includes automatic feature engineering:

**Original features:**
- MedInc: Median income
- HouseAge: House age
- AveRooms: Average rooms per household
- AveBedrms: Average bedrooms per household
- Population: Block group population
- AveOccup: Average occupancy
- Latitude: Latitude
- Longitude: Longitude

**Engineered features:**
- RoomsPerHousehold: AveRooms / AveOccup
- BedroomsPerRoom: AveBedrms / AveRooms
- PopulationPerHousehold: Population / AveOccup

## 🛠️ Customization

### 1. Add New Models

Edit `src/models/train.py` and add your model to the `get_models()` method:

```python
def get_models(self) -> Dict[str, Any]:
    return {
        'your_model': YourModel(param1=value1, param2=value2),
        # ... existing models
    }
```

### 2. Modify Feature Engineering

Edit the `preprocess_data()` method in `src/utils/data_loader.py`:

```python
def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
    # Add your feature engineering here
    processed_df['new_feature'] = some_transformation(df)
    return processed_df
```

### 3. Custom Validation Rules

Edit `src/api/schemas.py` to add custom validation:

```python
@validator('your_field')
def validate_your_field(cls, v):
    if v < 0:
        raise ValueError('Field must be positive')
    return v
```

## 🐛 Troubleshooting

### Common Issues

1. **Docker build fails:**
   ```bash
   docker-compose down
   docker system prune -f
   docker-compose build --no-cache
   ```

2. **Model not loading:**
   - Check if model files exist in `artifacts/`