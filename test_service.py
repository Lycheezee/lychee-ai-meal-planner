from services.similar_meal_service import similar_meal_service

# Test the service
try:
    print("Testing SimilarMealPlanService...")
    
    # Test with sample foods
    test_foods = ["Chicken Breast", "Rice"]
    result = similar_meal_service.generate_similar_meal_plans(days=3, start_foods=test_foods)
    
    print("✅ Service loaded successfully!")
    print(f"Generated {len(result)} days of meal plans")
    print(f"Sample result structure: {type(result)}")
    
    if result:
        print(f"First day keys: {list(result[0].keys()) if result else 'No results'}")
        if result[0].get('meals'):
            print(f"Sample meal: {result[0]['meals'][0] if result[0]['meals'] else 'No meals'}")
            
except Exception as e:
    print(f"❌ Error testing service: {e}")
    import traceback
    traceback.print_exc()
