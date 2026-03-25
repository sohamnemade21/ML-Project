from __future__ import annotations

import os

from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


def _parse_score(raw_value: str | None, field_name: str) -> float:
    if raw_value is None:
        raise ValueError(f"{field_name} is required.")

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc

    if value < 0 or value > 100:
        raise ValueError(f"{field_name} must be between 0 and 100.")

    return value


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=_parse_score(request.form.get("reading_score"), "Reading score"),
            writing_score=_parse_score(request.form.get("writing_score"), "Writing score"),
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template(
            "home.html",
            results=round(float(results[0]), 2),
            form_data=request.form,
        )
    except Exception as exc:
        return render_template(
            "home.html",
            error_message=str(exc),
            form_data=request.form,
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
