# California Housing Price Prediction - MLOps Project

A complete MLOps pipeline for predicting California housing prices using machine learning models with experiment tracking, API deployment, and monitoring.

## 🏗️ Project Structure

```
├── data/                    # Data files and preprocessing
├── src/                     # Source code
│   ├── models/             # Model training and evaluation
│   ├── api/                # FastAPI application
│   └── utils/              # Utility functions
├── tests/                  # Unit and integration tests
├── docker/                 # Docker configurations
├── monitoring/             # Prometheus/Grafana configs
├── .github/                # GitHub Actions workflows
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image definition
└── docker-compose.yml     # Multi-service orchestration
```

## 🚀 Features

- **Data Versioning**: Track datasets and preprocessing steps
- **Experiment Tracking**: MLflow for comprehensive experiment management
- **Model Registry**: Automated model selection and registration
- **API Service**: FastAPI with Pydantic validation
- **Containerization**: Docker for consistent deployments
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Logging**: Comprehensive logging system

## 🛠️ Tech Stack

- **ML Framework**: scikit-learn
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Containerization**: Docker & Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Data Validation**: Pydantic

## 📦 Installation

### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd california-housing-mlops
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up MLflow:
```bash
mlflow server --host 0.0.0.0 --port 5001
```

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

## 🎯 Usage

### Training Models

```bash
python src/models/train.py
```

### Running API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Making Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [8.3014, 21.0, 6.238137, 0.971880, 2401.0, 2.109842, 37.86, -122.22]}'
```

## 📊 Monitoring

- **MLflow UI**: http://localhost:5001
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src/
```

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline includes:
- Code linting (flake8, black)
- Unit testing
- Docker image building
- Automated deployment

## 📈 Model Performance

Current best model performance:
- Model: Random Forest
- RMSE: ~0.65
- R²: ~0.60

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License
