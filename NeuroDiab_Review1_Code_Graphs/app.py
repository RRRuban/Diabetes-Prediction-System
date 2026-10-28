
"""NeuroDiab Streamlit application.
Educational research prototype only; not a clinical diagnostic tool.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Project and model paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    ROOT / "review2_results" / "models" / "ann_pipeline.joblib",
    ROOT / "review2_results" / "models" / "ann_review2_pipeline.joblib",
    ROOT / "models" / "ann_pipeline.joblib",
    ROOT / "models" / "ann_review2_pipeline.joblib",
]

MODEL = next((path for path in MODEL_CANDIDATES if path.is_file()), None)

# --------------------------------------------------
# 2. Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroDiab | Diabetes Risk Research Demo",
    page_icon="🩺",
    layout="centered",
)

st.title("NeuroDiab")
st.subheader("ANN-Based Diabetes Risk Prediction System")

st.warning(
    "Educational research prototype only. This output is not a diagnosis, "
    "medical advice, or a substitute for professional assessment."
)

if MODEL is None:
    st.error("No saved ANN pipeline was found.")
    st.write("Checked these locations:")
    for path in MODEL_CANDIDATES:
        st.code(str(path))
    st.info(
        "Run 01_train_ann.py or check the filenames in your models folders. "
        "The selected file must be a trained preprocessing-and-model pipeline."
    )
    st.stop()

# --------------------------------------------------
# 3. Load model safely
# --------------------------------------------------

try:
    model = joblib.load(MODEL)
    st.caption(f"Loaded model: {MODEL.relative_to(ROOT)}")
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

# --------------------------------------------------
# 4. Questionnaire form
# --------------------------------------------------

st.write(
    "Enter the questionnaire responses below. "
    "The input fields match the UCI Early Stage Diabetes Risk Prediction Dataset."
)

FIELDS = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]

with st.form("risk_form"):
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=35,
        step=1,
    )

    gender = st.selectbox("Gender", ["Male", "Female"])

    values = {}
    columns = st.columns(2)

    for index, field in enumerate(FIELDS):
        with columns[index % 2]:
            values[field] = st.selectbox(
                field,
                ["No", "Yes"],
                key=field,
            )

    submitted = st.form_submit_button(
        "Estimate model class",
        type="primary",
    )

# --------------------------------------------------
# 5. Prediction helpers
# --------------------------------------------------

def is_positive_label(label):
    """Recognize common positive-class label representations."""
    if isinstance(label, str):
        return label.strip().lower() in {
            "1", "positive", "yes", "true"
        }

    return label == 1 or label is True


# --------------------------------------------------
# 6. Predict and display result
# --------------------------------------------------

if submitted:
    row = {
        "Age": int(age),
        "Gender": gender,
        **values,
    }

    X = pd.DataFrame([row], columns=["Age", "Gender", *FIELDS])

    try:
        prediction = model.predict(X)[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.write(
            "Check that the saved pipeline was trained with the same "
            "feature names and compatible input categories."
        )
        st.stop()

    positive = is_positive_label(prediction)

    st.subheader(
        "Positive class predicted"
        if positive
        else "Negative class predicted"
    )

    # Show the model score for the positive class when available.
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)[0]
            classes = list(model.classes_)

            positive_indices = [
                index
                for index, label in enumerate(classes)
                if is_positive_label(label)
            ]

            if positive_indices:
                score = float(probabilities[positive_indices[0]])
                st.write(
                    f"Model score for the positive class: **{score:.1%}**"
                )
                st.caption(
                    "This score is model output, not a clinically validated "
                    "probability of having diabetes."
                )
            else:
                st.info(
                    "The positive class could not be identified in the "
                    "saved model's class labels, so no score was displayed."
                )
        except Exception as exc:
            st.info(f"Could not calculate the model score: {exc}")

    st.info(
        "Please consult a qualified healthcare professional for appropriate "
        "testing and interpretation, regardless of this model output."
    )
