"""Train and compare ANN, Logistic Regression, Decision Tree, Random Forest, SVM and KNN.
Uses exact-record grouping to reduce duplicate-record leakage. This is an educational
research prototype, not a clinically validated diagnostic tool.
"""
from pathlib import Path
import hashlib, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score,
    brier_score_loss, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'diabetes_data_upload.csv'
OUT = ROOT / 'reports'; GRAPHS = ROOT / 'graphs'; MODELS = ROOT / 'models'
for p in (OUT, GRAPHS, MODELS): p.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def record_groups(df):
    # Group identical full records, including the target, so copies do not span folds.
    canonical = df.astype(str).agg('|'.join, axis=1)
    return canonical.map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()).to_numpy()


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('numeric', numeric, num_cols), ('categorical', categorical, cat_cols)])


def main():
    warnings.filterwarnings('ignore', category=UserWarning)
    df = pd.read_csv(DATA)
    target = 'class'
    if target not in df: raise ValueError("Expected target column named 'class'.")
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(str).str.strip().str.lower().map({'positive': 1, 'negative': 0})
    if y.isna().any(): raise ValueError('Unexpected target labels; expected Positive and Negative.')
    groups = record_groups(df)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pre = build_preprocessor(X)
    models = {
      'ANN': MLPClassifier(hidden_layer_sizes=(32,16), activation='relu', solver='adam', alpha=.001,
          learning_rate_init=.001, max_iter=1000, early_stopping=True, validation_fraction=.15,
          n_iter_no_change=25, random_state=RANDOM_STATE),
      'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
      'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=RANDOM_STATE),
      'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced',
          random_state=RANDOM_STATE, n_jobs=-1),
      'SVM': SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
      'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance')
    }
    scoring = {'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'}
    rows=[]; predictions={}; fitted={}; probability={}; cv_rows=[]
    for name, estimator in models.items():
        pipe = Pipeline([('preprocess', clone(pre)), ('model', estimator)])
        cv = cross_validate(pipe, X, y, groups=groups, cv=splitter, scoring=scoring,
                            n_jobs=1, error_score='raise')
        for i in range(5):
            cv_rows.append({'model':name,'fold':i+1, **{metric: float(cv[f'test_{metric}'][i]) for metric in scoring}})
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test).astype(int)
        if hasattr(pipe, 'predict_proba'): prob = pipe.predict_proba(X_test)[:,1]
        else: prob = pipe.decision_function(X_test)
        # For probability-based calibration and PR/ROC plots, all selected estimators provide predict_proba.
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
        row={'model':name, 'accuracy':accuracy_score(y_test,pred),
             'precision':precision_score(y_test,pred,zero_division=0),
             'recall_sensitivity':recall_score(y_test,pred,zero_division=0),
             'specificity':tn/(tn+fp) if tn+fp else 0,
             'f1':f1_score(y_test,pred,zero_division=0),
             'roc_auc':roc_auc_score(y_test,prob),
             'pr_auc_average_precision':average_precision_score(y_test,prob),
             'brier_score':brier_score_loss(y_test,prob),
             'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
        rows.append(row); predictions[name]=pred; probability[name]=prob; fitted[name]=pipe
        (OUT/f'{name.lower().replace(" ","_")}_classification_report.txt').write_text(
            classification_report(y_test,pred,target_names=['Negative','Positive'],zero_division=0), encoding='utf-8')
        pd.DataFrame(confusion_matrix(y_test,pred,labels=[0,1]),index=['Actual Negative','Actual Positive'],
                     columns=['Predicted Negative','Predicted Positive']).to_csv(OUT/f'{name.lower().replace(" ","_")}_confusion_matrix.csv')
        joblib.dump(pipe, MODELS/f'{name.lower().replace(" ","_")}_pipeline.joblib')
        print(f"{name}: accuracy={row['accuracy']:.4f}, F1={row['f1']:.4f}, ROC-AUC={row['roc_auc']:.4f}")

    metrics=pd.DataFrame(rows).sort_values('f1',ascending=False)
    metrics.to_csv(OUT/'model_comparison_metrics.csv',index=False)
    pd.DataFrame(cv_rows).to_csv(OUT/'grouped_cv_scores_all_models.csv',index=False)
    cv_summary=pd.DataFrame(cv_rows).groupby('model').agg(['mean','std'])
    cv_summary.to_csv(OUT/'grouped_cv_summary_all_models.csv')
    with open(OUT/'experiment_summary.json','w',encoding='utf-8') as f:
        json.dump({'dataset_rows':len(df),'feature_count':X.shape[1], 'exact_record_groups':int(pd.Series(groups).nunique()),
          'train_rows':len(train_idx),'test_rows':len(test_idx),'positive_rows':int(y.sum()),
          'negative_rows':int((1-y).sum()),'random_state':RANDOM_STATE,
          'split_method':'first fold of StratifiedGroupKFold; exact full-record duplicates grouped',
          'caution':'Internal dataset evaluation only; not external clinical validation.'},f,indent=2)

    # Model comparison plots
    for metric, title, fname in [
      ('accuracy','Held-out Accuracy Comparison','model_accuracy_comparison.png'),
      ('precision','Held-out Precision Comparison','model_precision_comparison.png'),
      ('recall_sensitivity','Held-out Recall / Sensitivity Comparison','model_recall_comparison.png'),
      ('f1','Held-out F1-score Comparison','model_f1_comparison.png'),
      ('roc_auc','Held-out ROC-AUC Comparison','model_roc_auc_comparison.png')]:
        fig,ax=plt.subplots(figsize=(10,5)); data=metrics.sort_values(metric,ascending=False)
        ax.bar(data['model'],data[metric]); ax.set_ylim(0,1.05); ax.set_ylabel(metric.replace('_',' ').title()); ax.set_title(title)
        ax.tick_params(axis='x',rotation=25); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(GRAPHS/fname,dpi=180); plt.close(fig)

    # ROC, PR and confusion matrix figures
    fig,ax=plt.subplots(figsize=(7,6))
    for name,prob in probability.items(): RocCurveDisplay.from_predictions(y_test,prob,name=name,ax=ax)
    ax.plot([0,1],[0,1],'k--',linewidth=1); ax.set_title('ROC Curves — Grouped Hold-out'); fig.tight_layout(); fig.savefig(GRAPHS/'all_models_roc_curves.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    for name,prob in probability.items(): PrecisionRecallDisplay.from_predictions(y_test,prob,name=name,ax=ax)
    ax.set_title('Precision–Recall Curves — Grouped Hold-out'); fig.tight_layout(); fig.savefig(GRAPHS/'all_models_precision_recall_curves.png',dpi=180); plt.close(fig)
    for name,pred in predictions.items():
        fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(y_test,pred,labels=[0,1],display_labels=['Negative','Positive'],ax=ax,colorbar=False)
        ax.set_title(f'{name} Confusion Matrix'); fig.tight_layout(); fig.savefig(GRAPHS/f'{name.lower().replace(" ","_")}_confusion_matrix.png',dpi=180); plt.close(fig)

    # ANN loss curve
    ann=fitted['ANN'].named_steps['model']
    if hasattr(ann,'loss_curve_'):
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(range(1,len(ann.loss_curve_)+1),ann.loss_curve_)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Training loss'); ax.set_title('ANN Training Loss Curve'); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(GRAPHS/'ann_training_loss_comparison_run.png',dpi=180); plt.close(fig)

    # ANN permutation importance on held-out data (model-agnostic, raw feature names)
    perm=permutation_importance(fitted['ANN'],X_test,y_test,n_repeats=15,random_state=RANDOM_STATE,scoring='f1',n_jobs=1)
    imp=pd.DataFrame({'feature':X.columns,'importance_mean':perm.importances_mean,'importance_std':perm.importances_std}).sort_values('importance_mean',ascending=False)
    imp.to_csv(OUT/'ann_permutation_feature_importance.csv',index=False)
    top=imp.head(12).sort_values('importance_mean')
    fig,ax=plt.subplots(figsize=(9,6)); ax.barh(top['feature'],top['importance_mean'],xerr=top['importance_std']);
    ax.set_xlabel('Decrease in F1 after permutation'); ax.set_title('ANN Permutation Feature Importance'); ax.grid(axis='x',alpha=.25)
    fig.tight_layout(); fig.savefig(GRAPHS/'ann_permutation_feature_importance.png',dpi=180); plt.close(fig)

    # Calibration plots; predictions are model scores from the held-out set.
    from sklearn.calibration import CalibrationDisplay
    fig,ax=plt.subplots(figsize=(7,6))
    for name,prob in probability.items(): CalibrationDisplay.from_predictions(y_test,prob,n_bins=8,strategy='quantile',name=name,ax=ax)
    ax.plot([0,1],[0,1],'k--',linewidth=1); ax.set_title('Probability Calibration — Held-out Data'); fig.tight_layout(); fig.savefig(GRAPHS/'all_models_calibration.png',dpi=180); plt.close(fig)

    # Save held-out predictions for reproducibility and audit
    pred_df=X_test.copy(); pred_df['actual_class']=np.where(y_test.to_numpy()==1,'Positive','Negative')
    for name in models:
        key=name.lower().replace(' ','_'); pred_df[f'{key}_prediction']=np.where(predictions[name]==1,'Positive','Negative')
        pred_df[f'{key}_score']=probability[name]
    pred_df.to_csv(OUT/'heldout_predictions.csv',index=False)
    print('\nSaved metrics, CV scores, fitted pipelines and graphs under reports/, models/ and graphs/.')
    print(metrics[['model','accuracy','precision','recall_sensitivity','specificity','f1','roc_auc','pr_auc_average_precision','brier_score']].to_string(index=False))

if __name__ == '__main__': main()
