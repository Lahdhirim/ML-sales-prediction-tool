# Sales Prediction Tool

This project is a **Sales Prediction Tool** that enables users to analyze and predict customer sales. It provides a fully customizable data preprocessing pipeline and machine learning hyperparameter tuning via a **Flask** and **Angular** interface.

## Features

### Data Analysis & Insights
- Exploratory Data Analysis (EDA) notebooks are available:
  - [`notebooks/data_analysis.ipynb`](notebooks/data_analysis.ipynb)
  - [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb)
- Key analysis questions addressed:
  - Ordered plot of total transactions per customer (most to least active)
  - Monthly transaction frequency for a given product ID (for 2018)
  - Top 5 highest-selling products over the last six months
  - Data seasonality effect

### Configurable Data Processing Pipeline
The pipeline consists of **three transformation levels**:
1. **Level 1:** Load and clean data (remove duplicates, transform date format)
2. **Level 2:** Compute monthly transactions per customer & define the target variable
3. **Level 3:** Compute rolling features and lag features
- Fully configurable via [`config/processing_config.json`](config/processing_config.json)

### Configurable Training Pipeline with Backtesting
- Ensures model robustness by simulating past performance
- Time-based data splits for training, validation, and testing:

| Split  | Training Period      | Validation Period  | Testing Period     |
|--------|----------------------|--------------------|--------------------|
| Split 1 | 2017-01 to 2017-12  | 2018-01 to 2018-03 | 2018-04 to 2018-06 |
| Split 2 | 2017-01 to 2018-03  | 2018-04 to 2018-06 | 2018-07 to 2018-09 |
| Split 3 | 2017-01 to 2018-06  | 2018-07 to 2018-09 | 2018-10 to 2018-12 |
| Split 4 | 2017-01 to 2018-09  | 2018-10 to 2018-12 | 2019-01 to 2019-03 |
| Split 5 | 2017-01 to 2018-12  | 2019-01 to 2019-03 | 2019-04 to 2019-06 |
| Split 6 | 2017-01 to 2019-03  | 2019-04 to 2019-06 | 2019-07 to 2019-09 |
- Various **Machine Learning algorithms** & **clustering methods** are supported
- Fully configurable via [`config/training_config.json`](config/training_config.json)
- Features can be selected via [`config/features.json`](config/features.json)

### Configurable Inference Pipeline
- Forecasts sales using trained models
- Allows configuration of:
  - **Number of models** used for aggregation
  - **Aggregation function** (e.g., weighted method prioritizes recent models)
- Configurable via [`config/inference_config.json`](config/inference_config.json)

### User Interface (Flask & Angular)
- Web interface to **edit configurations** and **run training pipeline**

---

## Getting Started

### Prerequisites
Make sure you have the following installed:
- [Python](http://python.org/downloads/)
- [Node.js](https://nodejs.org/en/download) (Optional, for UI)
- [Angular CLI](https://angular.dev/tools/cli) (Optional, for UI)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Lahdhirim/Quod_THA.git
    cd QUOD_THA
    ```
2. Install dependencies:
    - Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    - Angular UI (Optional):
        ```bash
        cd src/config_editor/angular_ui
        npm install
        ```

---

## Running the Pipelines
Three execution modes are available:

### Run Processing & Training Pipeline (Without UI)
```bash
python main.py train
```
Or simply:
```bash
python main.py
```

### Run Processing & Training Pipeline with Angular UI
```bash
python main.py train_with_ui
```

### Run Inference Pipeline
```bash
python main.py inference
```

---

## Output Files
Generated outputs are stored in `data/output`:
- **`raw_predictions.csv`** → Raw sales predictions from backtesting
- **`KPIs.xlsx`** → Performance metrics from backtesting
- **`inference.csv`** → Predictions during production phase