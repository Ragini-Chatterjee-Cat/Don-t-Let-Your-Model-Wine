# Don't Let Your Model Wine 🍷

A mini MLOps platform for training ML models and checking if they're being fair to different kinds of wines.

## Overview

This project demonstrates a complete MLOps workflow for wine quality prediction:

- **Trains** a model to predict wine quality (score from 0 to 10) based on chemical properties like alcohol, acidity, sulfates, etc.
- **Tracks experiments** with MLflow — logs parameters, metrics, and model artifacts automatically on every run
- **Packages** the model and metrics into the outputs folder for version control and reproducibility
- **Evaluates fairness** to ensure the model treats low-alcohol and high-alcohol wines equitably
- **Provides a CLI** tool (`src/cli.py`) to orchestrate the entire pipeline
- **Includes a dashboard** (Streamlit) to visualize model performance and fairness metrics
- **Dockerized** so you can run it anywhere

## Features

- Random Forest classifier for wine quality prediction
- MLflow experiment tracking — parameters, metrics, and model artifacts logged per run
- Interactive Streamlit dashboard
- CLI interface for experiment management
- Docker support for reproducible environments
- Fairness evaluation using statistical parity difference
- Model and metrics packaging for MLOps workflows

## Installation

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Ragini-Chatterjee-Cat/Don-t-Let-Your-Model-Wine.git
cd Don-t-Let-Your-Model-Wine
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Running an Experiment

Train a model and package the results:

```bash
python -m src.cli run-experiment data/WineQT.csv outputs
```

This will:
- Load the wine quality dataset
- Train a Random Forest classifier
- Log parameters, metrics, and the model to MLflow
- Save the model, metrics, and configuration to the `outputs/` directory

### Viewing the MLflow UI

After running an experiment, launch the MLflow tracking UI:

```bash
mlflow ui
```

Open `http://localhost:5000` to browse all runs, compare metrics, and download logged models.

### Checking Fairness

Evaluate if the model is fair across different alcohol levels:

```bash
python -m src.cli check-fairness data/WineQT.csv outputs/predictions.csv
```

### Launching the Dashboard

Visualize model performance and fairness metrics:

```bash
streamlit run src/dashboard.py
```

The dashboard will open at `http://localhost:8501`

## Docker Usage

### Build the Docker Image

```bash
docker build -t wine-mlops .
```

### Run an Experiment in Docker

```bash
docker run wine-mlops run-experiment data/WineQT.csv outputs
```

### Run the Dashboard in Docker

```bash
docker run -p 8501:8501 --entrypoint streamlit wine-mlops run src/dashboard.py
```

## Project Structure

```
Wine-Classification/
├── data/
│   └── WineQT.csv              # Wine quality dataset
├── src/
│   ├── cli.py                  # Command-line interface
│   ├── train.py                # Model training logic
│   ├── fairness.py             # Fairness evaluation metrics
│   ├── package_results.py      # Results packaging utilities
│   └── dashboard.py            # Streamlit dashboard
├── outputs/                    # Experiment outputs (models, metrics)
├── results/                    # Archived results
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## How It Works

### 1. Model Training (`train.py`)

- Loads wine quality data from CSV
- Splits data into 80/20 train/test sets
- Trains a Random Forest classifier
- Evaluates with accuracy and classification report
- Logs the full run to MLflow (parameters, metrics, model artifact)

### 2. Fairness Evaluation (`fairness.py`)

- Bins wines by alcohol content (< 10% vs >= 10%)
- Calculates average predictions for each group
- Computes statistical parity difference as fairness metric

### 3. Results Packaging (`package_results.py`)

- Saves trained model as pickle file
- Stores metrics in JSON format
- Saves configuration for reproducibility

### 4. CLI Tool (`cli.py`)

Built with Click for easy command management:
- `run-experiment`: End-to-end training pipeline
- `check-fairness`: Fairness evaluation workflow

### 5. Dashboard (`dashboard.py`)

Interactive Streamlit app to visualize:
- Model accuracy and performance metrics
- Fairness scores across groups
- Prediction distributions

## Dataset

The project uses the **Wine Quality Dataset** from Kaggle, which contains:
- **Features**: Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Target**: Quality score (0-10)
- **Size**: ~1,143 samples

## Dependencies

- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `mlflow` - Experiment tracking
- `click` - CLI framework
- `streamlit` - Dashboard framework
- `seaborn` - Visualization
- `numpy` - Numerical operations
- `matplotlib` - Plotting

See `requirements.txt` for complete list.

## Fairness Metrics

**Statistical Parity Difference**: Measures the absolute difference in average predictions between low-alcohol and high-alcohol wine groups. Lower values indicate more equitable treatment.

## Future Improvements

- [ ] Add more fairness metrics (equal opportunity, disparate impact)
- [ ] Implement hyperparameter tuning
- [ ] Add model explainability (SHAP values)
- [ ] Support multiple model types
- [ ] Add CI/CD pipeline
- [ ] Add MLflow Model Registry for promoting models to staging/production

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Author

Ragini Chatterjee

## Acknowledgments

- Dataset from Kaggle Wine Quality Dataset
- Built with scikit-learn, MLflow, Streamlit, and Docker
