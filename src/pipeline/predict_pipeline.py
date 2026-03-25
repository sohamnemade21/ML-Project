from __future__ import annotations

import os
import sys
from functools import lru_cache

import dill
import pandas as pd

from src.exception import CustomException


FEATURE_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "reading_score",
    "writing_score",
]


@lru_cache(maxsize=1)
def _load_artifacts(model_path: str, preprocessor_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at: {preprocessor_path}")

    with open(preprocessor_path, "rb") as preprocessor_file:
        preprocessor = dill.load(preprocessor_file)

    with open(model_path, "rb") as model_file:
        model = dill.load(model_file)

    return model, preprocessor


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            model, preprocessor = _load_artifacts(self.model_path, self.preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str | None,
        race_ethnicity: str | None,
        parental_level_of_education: str | None,
        lunch: str | None,
        test_preparation_course: str | None,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            input_df = pd.DataFrame(custom_data_input_dict)
            return input_df[FEATURE_COLUMNS]
        except Exception as e:
            raise CustomException(e, sys)
