"""
Test suite for the FastAPI application.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app
from api.schemas import HousingFeatures

client = TestClient(app)


class TestAPI:
    """Test cases for the ML API."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
    
    @patch('api.main.model_service')
    def test_predict_endpoint_success(self, mock_model_service):
        """Test successful prediction."""
        # Mock the model service
        mock_model_service.predict.return_value = 4.526
        mock_model_service.metadata = {'model_name': 'test_model'}
        
        # Test data
        test_data = {
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
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert data["prediction"] == 4.526
    
    def test_predict_endpoint_validation_error(self):
        """Test prediction with invalid input."""
        # Invalid data (missing required field)
        test_data = {
            "features": {
                "MedInc": 8.3014,
                "HouseAge": 21.0,
                # Missing other required fields
            }
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_validation_range_error(self):
        """Test prediction with out-of-range values."""
        # Invalid data (out of range)
        test_data = {
            "features": {
                "MedInc": 50.0,  # Too high
                "HouseAge": 21.0,
                "AveRooms": 6.238137,
                "AveBedrms": 0.971880,
                "Population": 2401.0,
                "AveOccup": 2.109842,
                "Latitude": 37.86,
                "Longitude": -122.22
            }
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.model_service')
    def test_batch_predict_endpoint_success(self, mock_model_service):
        """Test successful batch prediction."""
        # Mock the model service
        mock_model_service.predict_batch.return_value = [4.526, 2.345]
        mock_model_service.metadata = {'model_name': 'test_model'}
        
        # Test data
        test_data = {
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
        }
        
        response = client.post("/predict/batch", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert len(data["predictions"]) == 2
        assert data["count"] == 2
    
    def test_batch_predict_endpoint_empty_list(self):
        """Test batch prediction with empty list."""
        test_data = {
            "features_list": []
        }
        
        response = client.post("/predict/batch", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint_too_many_items(self):
        """Test batch prediction with too many items."""
        # Create a list with more than 100 items
        test_features = {
            "MedInc": 8.3014,
            "HouseAge": 21.0,
            "AveRooms": 6.238137,
            "AveBedrms": 0.971880,
            "Population": 2401.0,
            "AveOccup": 2.109842,
            "Latitude": 37.86,
            "Longitude": -122.22
        }
        
        test_data = {
            "features_list": [test_features] * 101  # More than allowed
        }
        
        response = client.post("/predict/batch", json=test_data)
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.model_service')
    def test_model_info_endpoint(self, mock_model_service):
        """Test model info endpoint."""
        mock_model_service.model = Mock()  # Mock loaded model
        mock_model_service.metadata = {
            'model_name': 'test_model',
            'val_rmse': 0.65,
            'timestamp': '2024-01-01T12:00:00Z'
        }
        
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "rmse" in data
    
    @patch('api.main.model_service')
    def test_model_info_endpoint_no_model(self, mock_model_service):
        """Test model info endpoint when no model is loaded."""
        mock_model_service.model = None
        
        response = client.get("/model/info")
        assert response.status_code == 503  # Service unavailable
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should return Prometheus formatted metrics
    
    def test_retrain_endpoint(self):
        """Test model retrain trigger endpoint."""
        response = client.post("/model/retrain")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "timestamp" in data
        assert "status" in data


class TestSchemas:
    """Test cases for Pydantic schemas."""
    
    def test_housing_features_valid(self):
        """Test valid housing features."""
        data = {
            "MedInc": 8.3014,
            "HouseAge": 21.0,
            "AveRooms": 6.238137,
            "AveBedrms": 0.971880,
            "Population": 2401.0,
            "AveOccup": 2.109842,
            "Latitude": 37.86,
            "Longitude": -122.22
        }
        
        features = HousingFeatures(**data)
        assert features.MedInc == 8.3014
        assert features.HouseAge == 21.0
    
    def test_housing_features_invalid_range(self):
        """Test housing features with invalid range."""
        data = {
            "MedInc": 50.0,  # Too high
            "HouseAge": 21.0,
            "AveRooms": 6.238137,
            "AveBedrms": 0.971880,
            "Population": 2401.0,
            "AveOccup": 2.109842,
            "Latitude": 37.86,
            "Longitude": -122.22
        }
        
        with pytest.raises(ValueError):
            HousingFeatures(**data)
    
    def test_housing_features_bedrooms_validation(self):
        """Test custom validation for bedrooms vs rooms."""
        data = {
            "MedInc": 8.3014,
            "HouseAge": 21.0,
            "AveRooms": 2.0,
            "AveBedrms": 3.0,  # More bedrooms than rooms
            "Population": 2401.0,
            "AveOccup": 2.109842,
            "Latitude": 37.86,
            "Longitude": -122.22
        }
        
        with pytest.raises(ValueError):
            HousingFeatures(**data)


# Fixture for test data
@pytest.fixture
def sample_housing_data():
    """Sample housing data for testing."""
    return {
        "MedInc": 8.3014,
        "HouseAge": 21.0,
        "AveRooms": 6.238137,
        "AveBedrms": 0.971880,
        "Population": 2401.0,
        "AveOccup": 2.109842,
        "Latitude": 37.86,
        "Longitude": -122.22
    }


if __name__ == "__main__":
    pytest.main([__file__])