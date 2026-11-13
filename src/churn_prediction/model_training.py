"""Model definition, training pipelines, cross-validation, and hyperparameter tuning."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from churn_prediction.config import CV_FOLDS, RANDOM_STATE
from churn_prediction.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Instantiate candidate classifiers for benchmarking.
    
    Returns
    -------
    dict
        Dictionary mapping model names to classifier instances.
    """
    models = {
        "Dummy (Majority Class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            class_weight="balanced",
            random_state=random_state,
        ),
    }

    if XGB_AVAILABLE:
        # scale_pos_weight is approximately (73.5 / 26.5) ~ 2.77
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=2.77,
            eval_metric="logloss",
            random_state=random_state,
        )

    return models


def build_full_pipeline(classifier: Any) -> Pipeline:
    """Combine the modular feature preprocessor with a classifier into an end-to-end Pipeline.
    
    Parameters
    ----------
    classifier : estimator
        Scikit-Learn compatible classification estimator.
        
    Returns
    -------
    Pipeline
        End-to-end pipeline.
    """
    preprocessor = build_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate all candidate models using Stratified K-Fold cross validation.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    cv_folds : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Seed for K-Fold split.
        
    Returns
    -------
    pd.DataFrame
        Summary table comparing mean and standard deviations of cross-validation metrics.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    candidate_models = get_candidate_models(random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "accuracy": "accuracy",
    }

    results = []

    for name, clf in candidate_models.items():
        logger.info(f"Running {cv_folds}-fold CV for {name}...")
        pipeline = build_full_pipeline(clf)

        # Cross validate
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        res = {
            "Model": name,
            "ROC-AUC (Mean)": np.mean(scores["test_roc_auc"]),
            "ROC-AUC (Std)": np.std(scores["test_roc_auc"]),
            "PR-AUC (Mean)": np.mean(scores["test_pr_auc"]),
            "PR-AUC (Std)": np.std(scores["test_pr_auc"]),
            "Recall (Mean)": np.mean(scores["test_recall"]),
            "Recall (Std)": np.std(scores["test_recall"]),
            "Precision (Mean)": np.mean(scores["test_precision"]),
            "Precision (Std)": np.std(scores["test_precision"]),
            "F1-Score (Mean)": np.mean(scores["test_f1"]),
            "F1-Score (Std)": np.std(scores["test_f1"]),
            "Accuracy (Mean)": np.mean(scores["test_accuracy"]),
        }
        results.append(res)

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC (Mean)", ascending=False)
    return results_df


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "HistGradientBoosting",
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Perform hyperparameter optimization using GridSearchCV on selected model architecture.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    model_type : str
        Model to tune ("HistGradientBoosting", "LogisticRegression", "RandomForest", "XGBoost").
    cv_folds : int, default=5
        Number of cross validation folds.
    random_state : int, default=42
        Reproducibility seed.
        
    Returns
    -------
    Tuple[Pipeline, dict]
        Best fitted Pipeline and dictionary of best parameters.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if model_type == "HistGradientBoosting":
        base_clf = HistGradientBoostingClassifier(random_state=random_state)
        param_grid = {
            "classifier__max_iter": [100, 150],
            "classifier__learning_rate": [0.03, 0.05],
            "classifier__max_leaf_nodes": [15, 31],
            "classifier__min_samples_leaf": [30, 50],
            "classifier__class_weight": [None],
        }
    elif model_type == "XGBoost" and XGB_AVAILABLE:
        base_clf = XGBClassifier(eval_metric="logloss", random_state=random_state)
        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__scale_pos_weight": [2.0, 2.77, 3.5],
        }
    elif model_type == "LogisticRegression":
        base_clf = LogisticRegression(max_iter=1000, random_state=random_state)
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": ["balanced", None],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
    else:
        base_clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [6, 8, 10],
            "classifier__min_samples_split": [5, 10],
            "classifier__class_weight": ["balanced", None],
        }

    pipeline = build_full_pipeline(base_clf)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    logger.info(f"Initiating GridSearchCV for {model_type}...")
    grid_search.fit(X_train, y_train)

    logger.info(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
    logger.info(f"Best Parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_
