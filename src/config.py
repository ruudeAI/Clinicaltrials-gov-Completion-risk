"""
config.py — Centralized configuration for the entire project.

WHY THIS FILE EXISTS:
    Instead of scattering magic numbers and hardcoded strings across every file,
    we keep them all here. If you need to change the API URL, add a feature,
    or tweak a model parameter, you only change ONE file.

HOW TO USE:
    from src.config import API_BASE_URL, RANDOM_SEED, PHASE_MAP
"""

import os

# ==============================================================================
# 1. PROJECT PATHS
# ==============================================================================
# os.path.abspath(__file__)  → full path to THIS file (config.py)
# os.path.dirname(...)       → the folder containing this file (src/)
# Going up one more level gives us the project root.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "sample")

# Output directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# ==============================================================================
# 2. API CONFIGURATION
# ==============================================================================
# ClinicalTrials.gov API v2 base URL.
# Docs: https://clinicaltrials.gov/data-api/api

API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# How many trials to fetch per API request (max 1000).
API_PAGE_SIZE = 1000

# Seconds to wait between API requests (be a good citizen!).
API_RATE_LIMIT_SECONDS = 0.5

# The fields we request from each trial record.
# Using shorthand aliases — see ClinicalTrials.gov API docs for full paths.
API_FIELDS = [
    "NCTId",                    # Unique trial identifier
    "BriefTitle",               # Short public title
    "OverallStatus",            # Recruitment status (our TARGET variable)
    "Phase",                    # Trial phase (Phase 1, 2, 3, etc.)
    "StudyType",                # INTERVENTIONAL, OBSERVATIONAL, etc.
    "EnrollmentCount",          # Number of planned participants
    "StartDate",                # When the trial started
    "PrimaryCompletionDate",    # When primary outcomes were expected
    "LeadSponsorName",          # Who is funding/running the trial
    "ConditionsModule",         # Diseases being studied
    "ArmsInterventionsModule",  # Treatment arms and interventions
    "EligibilityCriteria",      # Inclusion/exclusion criteria (free text)
    "Sex",                      # ALL, MALE, or FEMALE
    "MinimumAge",               # e.g. "18 Years"
    "MaximumAge",               # e.g. "65 Years"
    "HealthyVolunteers",        # Whether healthy people can join
    "StdAge",                   # Age groups: CHILD, ADULT, OLDER_ADULT
]

# ==============================================================================
# 3. TARGET VARIABLE DEFINITIONS
# ==============================================================================
# We group trial statuses into two classes for our binary prediction task.

# Class 0: The trial completed as planned — this is the "good" outcome.
COMPLETED_STATUSES = ["COMPLETED"]

# Class 1: The trial was stopped early — this is the "at risk" outcome.
AT_RISK_STATUSES = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]

# ==============================================================================
# 4. FEATURE ENGINEERING CONSTANTS
# ==============================================================================

# Map trial phases to numeric values for the ML model.
# Higher number = later phase = generally larger, more expensive trial.
PHASE_MAP = {
    "EARLY_PHASE1": 0,
    "PHASE1": 1,
    "PHASE2": 2,
    "PHASE3": 3,
    "PHASE4": 4,
    "NA": -1,       # Phase not applicable (e.g., observational studies)
}

# Keywords that suggest an industry (pharma/biotech) sponsor.
# We'll check if the sponsor name contains any of these (case-insensitive).
INDUSTRY_KEYWORDS = [
    "pharma", "biotech", "therapeutics", "inc.", "inc,", "corp",
    "ltd", "gmbh", "s.a.", "plc", "laboratories", "lab ",
    "biosciences", "oncology", "medicines", "drug",
    "pfizer", "novartis", "roche", "merck", "astrazeneca",
    "johnson", "abbott", "bayer", "sanofi", "gsk",
    "glaxosmithkline", "eli lilly", "amgen", "gilead",
    "bristol-myers", "abbvie", "biogen", "regeneron",
    "moderna", "takeda", "boehringer",
]

# ==============================================================================
# 5. MODEL TRAINING PARAMETERS
# ==============================================================================

# Random seed for reproducibility — using the same seed means you get
# the same train/test split and model results every time.
RANDOM_SEED = 42

# What fraction of data to hold out for testing (0.2 = 20%).
TEST_SIZE = 0.2

# Number of cross-validation folds (5 is a good default).
CV_FOLDS = 5

# ==============================================================================
# 6. DATA COLLECTION SCOPE
# ==============================================================================
# We fetch drug-related trials across 20 broad medical areas.
# search_query_source represents the query that retrieved the trial, not necessarily the official disease category.
SEARCH_QUERIES = [
    "cancer",
    "diabetes",
    "heart disease",
    "hypertension",
    "asthma",
    "depression",
    "alzheimer",
    "arthritis",
    "covid",
    "infectious disease",
    "autoimmune disease",
    "kidney disease",
    "liver disease",
    "pain",
    "epilepsy",
    "obesity",
    "migraine",
    "multiple sclerosis",
    "parkinson",
    "hiv"
]

# Configurable number of trials to fetch per condition (MVP default: 200)
TRIALS_PER_CONDITION = 200

# Centralized paths for drug-related trials
DRUG_RAW_CSV_PATH = os.path.join(DATA_RAW_DIR, "drug_trials_raw.csv")
DRUG_PROCESSED_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "drug_trials_processed.csv")
DRUG_SAMPLE_CSV_PATH = os.path.join(DATA_SAMPLE_DIR, "drug_trials_sample.csv")
DRUG_MODEL_PATH = os.path.join(MODELS_DIR, "drug_trial_completion_model.joblib")

