"""
Test file for similar_meal_planner.py router
Tests the similar_meal_plan endpoint functionality
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add the systems directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from routers.similar_meal_planner import DurationRequest

client = TestClient(app)

class TestSimilarMealPlannerRouter:
    """Test class for similar meal planner router functionality"""
    
    def test_similar_meal_plan_valid_request(self):
        """Test creating similar meal plans with valid input data"""
        test_data = {
            "initialMeal": ["chicken breast", "brown rice", "broccoli"],
            "days": 7
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Check that the response has the expected structure
        assert "plans" in response_data
        assert isinstance(response_data["plans"], list)
        
        # If plans are generated, check their structure
        if response_data["plans"]:
            plan = response_data["plans"][0]
            assert "date" in plan
            assert "meals" in plan
            assert isinstance(plan["meals"], list)
    
    def test_similar_meal_plan_default_days(self):
        """Test creating similar meal plans with default days (30)"""
        test_data = {
            "initialMeal": ["salmon", "quinoa"]
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "plans" in response_data
        assert isinstance(response_data["plans"], list)
        
        # Should generate plans for 30 days by default
        if response_data["plans"]:
            assert len(response_data["plans"]) <= 30  # Could be less due to processing
    
    def test_similar_meal_plan_single_food(self):
        """Test creating similar meal plans with a single initial food"""
        test_data = {
            "initialMeal": ["apple"],
            "days": 5
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "plans" in response_data
        assert isinstance(response_data["plans"], list)
    
    def test_similar_meal_plan_multiple_foods(self):
        """Test creating similar meal plans with multiple initial foods"""
        test_data = {
            "initialMeal": [
                "oatmeal", "banana", "almonds", "yogurt", "berries"
            ],
            "days": 14
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "plans" in response_data
        assert isinstance(response_data["plans"], list)
    
    def test_similar_meal_plan_empty_initial_meal(self):
        """Test creating similar meal plans with empty initial meal list"""
        test_data = {
            "initialMeal": [],
            "days": 7
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should still return a response, possibly with empty plans
        assert "plans" in response_data
        assert isinstance(response_data["plans"], list)
    
    def test_similar_meal_plan_zero_days(self):
        """Test creating similar meal plans with zero days"""
        test_data = {
            "initialMeal": ["chicken breast"],
            "days": 0
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should return empty plans or handle gracefully
        assert "plans" in response_data
        assert isinstance(response_data["plans"], list)
        assert len(response_data["plans"]) == 0
    
    def test_similar_meal_plan_large_days(self):
        """Test creating similar meal plans with a large number of days"""
        test_data = {
            "initialMeal": ["beef", "potatoes"],
            "days": 100
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "plans" in response_data
        assert isinstance(response_data["plans"], list)
    
    def test_similar_meal_plan_missing_initial_meal(self):
        """Test creating similar meal plans without initial meal field"""
        test_data = {
            "days": 7
            # Missing initialMeal field
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        # Should fail with validation error
        assert response.status_code == 422
    
    def test_similar_meal_plan_invalid_food_names(self):
        """Test creating similar meal plans with invalid/unknown food names"""
        test_data = {
            "initialMeal": ["xyz_unknown_food", "another_invalid_food"],
            "days": 5
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        # Should still work but might return fewer or no similar foods
        assert response.status_code == 200
        response_data = response.json()
        
        assert "plans" in response_data
        assert isinstance(response_data["plans"], list)
    
    def test_duration_request_model_validation(self):
        """Test the DurationRequest pydantic model validation"""
        # Valid request with explicit days
        valid_request = DurationRequest(
            initialMeal=["chicken", "rice"],
            days=14
        )
        
        assert valid_request.initialMeal == ["chicken", "rice"]
        assert valid_request.days == 14
        
        # Valid request with default days
        default_request = DurationRequest(
            initialMeal=["salmon"]
        )
        
        assert default_request.initialMeal == ["salmon"]
        assert default_request.days == 30  # Default value
    
    @patch('services.similar_meal_service.similar_meal_service.generate_similar_meal_plans')
    def test_similar_meal_plan_service_call(self, mock_generate_plans):
        """Test that the service is called with correct parameters"""
        # Mock the service response
        mock_plans = [
            {
                "date": "2025-06-09T00:00:00",
                "meals": [{"foodId": "123", "status": "not_completed"}]
            }
        ]
        mock_generate_plans.return_value = mock_plans
        
        test_data = {
            "initialMeal": ["chicken", "rice"],
            "days": 7
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        
        # Verify the service was called with correct parameters
        mock_generate_plans.assert_called_once_with(7, ["chicken", "rice"])
    
    def test_similar_meal_plan_response_structure(self):
        """Test that the response has the correct structure for generated plans"""
        test_data = {
            "initialMeal": ["oats"],
            "days": 3
        }
        
        response = client.post("/api/similar-meal-plan", json=test_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "plans" in response_data
        plans = response_data["plans"]
        
        # If plans exist, verify their structure
        for plan in plans:
            assert "date" in plan
            assert "meals" in plan
            assert isinstance(plan["meals"], list)
            
            # Check meal structure
            for meal in plan["meals"]:
                assert "foodId" in meal
                assert "status" in meal
                assert meal["status"] == "not_completed"

if __name__ == "__main__":
    pytest.main([__file__])
