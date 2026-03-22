import sys
import traceback
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import evaluate_model

try:
    print("\n" + "="*70)
    print(" "*15 + "MACHINE LEARNING MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Data Ingestion
    print("\n[1/3] DATA INGESTION")
    print("-" * 70)
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"✓ Training data: {train_data}")
    print(f"✓ Test data: {test_data}")

    # Data Transformation
    print("\n[2/3] DATA TRANSFORMATION")
    print("-" * 70)
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
    print(f"✓ Train array shape: {train_arr.shape}")
    print(f"✓ Test array shape: {test_arr.shape}")
    print(f"✓ Preprocessor saved: {preprocessor_path}")

    # Model Training
    print("\n[3/3] MODEL TRAINING & EVALUATION")
    print("-" * 70)
    
    from catboost import CatBoostRegressor
    from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from xgboost import XGBRegressor
    
    x_train = train_arr[:, :-1]
    y_train = train_arr[:, -1]
    x_test = test_arr[:, :-1]
    y_test = test_arr[:, -1]
    
    models = {
        "Random Forest": RandomForestRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "XGBRegressor": XGBRegressor(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "AdaBoost Regressor": AdaBoostRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Linear Regression": LinearRegression()
    }
    
    print("\nTraining models and evaluating performance...\n")
    
    results = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            'Model': model_name,
            'R² Score': round(r2, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'MSE': round(mse, 4)
        })
        
        status = "✓" if r2 >= 0.6 else "✗"
        print(f"{status} {model_name:30s} | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    
    # Find best model
    results_df = pd.DataFrame(results)
    best_idx = results_df['R² Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_r2 = results_df.loc[best_idx, 'R² Score']
    
    print("\n" + "="*70)
    print("MODEL RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("🏆 BEST PERFORMING MODEL")
    print("="*70)
    print(f"Model: {best_model_name}")
    print(f"R² Score: {best_r2:.4f}")
    print(f"File Path: artifacts/model.pkl")
    print("="*70 + "\n")
    
    if best_r2 >= 0.6:
        print("✓ Model meets quality threshold (R² >= 0.6)")
        print("✓ Model has been saved successfully")
    else:
        print("✗ Warning: Model R² score is below 0.6 threshold")
    
    print("\n")

except Exception as e:
    print("\n" + "="*70)
    print("ERROR OCCURRED")
    print("="*70)
    print(traceback.format_exc())
    sys.exit(1)
