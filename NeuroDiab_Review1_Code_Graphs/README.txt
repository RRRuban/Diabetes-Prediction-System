NEURODIAB — COMPLETE REVIEW-I + REVIEW-II PROJECT

This is the original Review-I code/graphs folder extended in-place with Review-II implementation.
The Review-I script and original graphs_from_ppt_values folder are retained. Review-II files are added beside them.

DATASET
data/diabetes_data_upload.csv — UCI Early Stage Diabetes Risk Prediction dataset (questionnaire/symptom-based).
Do not use Pima-specific cleaning rules on this dataset.

FILES
neurodiab_review1.py       Review-I audit, EDA, and graph generation from the CSV.
01_train_ann.py            ANN baseline training and evaluation.
02_compare_models.py       Train/evaluate ANN, Logistic Regression, Decision Tree, Random Forest, SVM, and KNN.
app.py                     Optional Streamlit questionnaire demo using the saved ANN pipeline.
requirements.txt           Unified dependencies for Review-I and Review-II.
graphs_from_ppt_values/    Original reconstructed graphs based on values in the earlier PPT; label as PPT-reported until independently verified.
graphs/                    Review-II evaluation graphs and related generated figures.
reports/                   Metrics, CV results, confusion matrices, prediction audit and summaries.
models/                    Saved scikit-learn pipelines.

SETUP (Windows / VS Code)
1. Extract the ZIP and open this folder in VS Code.
2. Create environment: py -m venv .venv
3. Activate: .venv\Scripts\activate
4. Install: pip install -r requirements.txt

RUN REVIEW-I
python neurodiab_review1.py --csv data/diabetes_data_upload.csv --out review1_results

RUN REVIEW-II ANN BASELINE
python 01_train_ann.py

RUN REVIEW-II COMPARISON
python 02_compare_models.py

RUN THE WEB DEMO
streamlit run app.py

The demo uses models/ann_pipeline.joblib, which is created by 02_compare_models.py. If it is missing, run the comparison script first.

METHODOLOGY NOTES
- Pipelines fit imputers, scaling, and one-hot encoding on training folds only.
- StratifiedGroupKFold groups exact duplicate full records so identical records do not cross folds.
- The comparison script uses the first grouped fold as a hold-out and reports five-fold grouped CV.
- This dataset has 520 rows and 16 questionnaire inputs plus the class target. It contains exact duplicates; they are grouped rather than silently removed.
- Metrics are internal research results, not external clinical validation. The model is not a diagnostic tool.
- Review-I graphs_from_ppt_values are reconstructed from previously reported PPT values and are not independently verified by the Review-I script. Run the script against the included CSV for dataset-derived outputs.

OUTPUTS
Review-I output tables/graphs are written to review1_results/. Review-II tables are written to reports/ and figures to graphs/. Model pipelines are saved in models/.
