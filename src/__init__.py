"""
src/ — Core source code for the Clinical Trials Completion Risk Predictor.

This package contains the full ML pipeline:
    - config.py             → Centralized settings and constants
    - clinicaltrials_api.py → Data collection from ClinicalTrials.gov API
    - preprocess.py         → Data cleaning and feature engineering
    - train_model.py        → Model training and cross-validation
    - evaluate.py           → Model evaluation and visualization
    - predict.py            → Prediction on new trial data
"""
