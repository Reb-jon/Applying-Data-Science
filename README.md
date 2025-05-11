Credit Scoring with Machine Learning
A modular framework for consumer default prediction using LendingClub, macroeconomic, and environmental data.

Overview
This project implements a machine learning–driven credit scoring framework aimed at predicting consumer loan default risk using LendingClub data and U.S. environmental and macroeconomic indicators. 
It emphasizes transparency, interpretability, and flexibility, offering configurable threshold tuning to match different business risk tolerances.

Core Features
- Preprocessing pipeline: Handles leakage control, class imbalance, and feature selection.
- Model benchmarking: Compares Random Forest, XGBoost, MLP, and LSTM architectures.
- Threshold optimisation: Investigates the trade-off between recall and precision under both F1 and G-Mean strategies.
- Ablation study: Assesses the marginal contribution of macroeconomic and environmental indicators.
- Modular design: Flexible components for future experimentation and production-ready deployment.

Repository Structure
Applying-Data-Science/
│
├── Data Preprocessing/       # Cleaning, leakage control, encoding
├── Macro_features/           # Macroeconomic variable integration and analysis
├── Main Model/               # Final model training and evaluation
├── Models/                   # Alternative architectures (XGBoost, MLP, LSTM)
├── metrics results.xlsx      # Performance metrics across models
├── Copy of model performance(1).xlsx
├── requirements.txt          # Python package dependencies for the project
└── README.md                 # You're here!

Usage
1. Clone the repo:
  git clone https://github.com/Reb-jon/Applying-Data-Science.git
  cd Applying-Data-Science
2. Install dependencies (example with pip):
   pip install -r requirements.txt
3. Run notebooks:
  Start with data preprocessing, then proceed through feature engineering and modelling steps.

Citation
If you use this work, please cite the following Zenodo DOI: 10.5281/zenodo.15384598




