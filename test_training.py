import sys
import traceback
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

try:
    print("=" * 50)
    print("Starting Data Ingestion...")
    print("=" * 50)
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"✓ Data Ingestion Complete")
    print(f"  Train data: {train_data}")
    print(f"  Test data: {test_data}")

    print("\n" + "=" * 50)
    print("Starting Data Transformation...")
    print("=" * 50)
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
    print(f"✓ Data Transformation Complete")
    print(f"  Train array shape: {train_arr.shape}")
    print(f"  Test array shape: {test_arr.shape}")

    print("\n" + "=" * 50)
    print("Starting Model Training...")
    print("=" * 50)
    modeltrainer = ModelTrainer()
    r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
    print(f"✓ Model Training Complete")
    print(f"  R² Score: {r2_score}")
    print(f"  Model saved to: artifacts/model.pkl")
    
    print("\n" + "=" * 50)
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 50)

except Exception as e:
    print("\n" + "=" * 50)
    print("ERROR OCCURRED:")
    print("=" * 50)
    print(traceback.format_exc())
    sys.exit(1)
