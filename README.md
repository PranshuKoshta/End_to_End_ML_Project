# End-to-End Machine Learning Project: Student Performance Indicator

## Project Overview

This project is an end-to-end machine learning application designed to **predict student performance test scores** based on various influencing factors. It follows industry-standard best practices for building **modular, well-structured data science projects**.

The project covers essential stages of a machine learning lifecycle, including **data ingestion, data transformation, model training, evaluation, and deployment**. It also incorporates crucial aspects like **environment management, version control, exception handling, and logging** for robust and maintainable code.

## Problem Statement

The goal of this project is to understand how student performance test scores are **affected by variables** such as Gender, Ethnicity, Parental Level of Education, Lunch type, and Test Preparation Course completion. The project aims to **predict the test score** based on these features.

The dataset used consists of **1000 rows and 8 columns**, including both categorical and numerical features.

## Project Structure

The project follows a **modular and component-based architecture** to ensure code reusability and maintainability. The main source code resides in the `src` folder.

The key directories and files within `src` are:

*   **`components`**: Contains distinct modules for each stage of the ML lifecycle.
    *   `data_ingestion.py`: Handles reading the data from a source (initially CSV, planned for databases), splitting it into train and test sets, and saving the data.
    *   `data_transformation.py`: Contains code for data cleaning, preprocessing, and feature engineering, including handling missing values, categorical encoding, and scaling numerical features.
    *   `model_trainer.py`: Responsible for training various machine learning models, evaluating them, selecting the best model, and saving it.
*   **`pipeline`**: Contains scripts to orchestrate the workflow by calling the components.
    *   `train_pipeline.py`: Script for running the entire training process from data ingestion to model training.
    *   `predict_pipeline.py`: Script for handling new data inputs, preprocessing them using the saved objects, and generating predictions.
*   **`exception.py`**: Implements a **custom exception handling mechanism** to provide detailed error messages including script name and line number.
*   **`logger.py`**: Sets up a **logging configuration** to record execution flow and errors into a log file.
*   **`utils.py`**: Contains **reusable utility functions** for common tasks like saving and loading Python objects (models, preprocessors).
*   **`__init__.py`**: Present in package directories (like `src`, `components`, `pipeline`) to allow Python to recognize them as packages and enable imports.

Other important files at the root level:

*   **`setup.py`**: Used to build the entire application as a **Python package**, allowing easy installation and distribution.
*   **`requirements.txt`**: Lists all the necessary Python libraries and dependencies required for the project.
*   **`.gitignore`**: Specifies files and directories (like `venv`, `artifacts`) that should be ignored by Git and not committed to the repository.
*   **`app.py`**: Contains the Flask application code for the web interface.
*   **`notebook/`**: Contains Jupyter notebooks used for initial Data Exploration (EDA) and model experimentation.
    *   `data/student.csv`: The raw dataset file.
    *   `EDA.ipynb`: Notebook demonstrating the initial Exploratory Data Analysis.
    *   `Model Training.ipynb`: Notebook demonstrating initial model training and evaluation.
*   **`artifacts/`**: Directory created during the data ingestion and transformation process to store processed data and trained model/preprocessor objects.

## Technologies Used

*   Python
*   Flask (for web application)
*   scikit-learn (for machine learning algorithms and preprocessing) 
*   Pandas (for data manipulation) 
*   NumPy (for numerical operations) 
*   Seaborn and Matplotlib (for data visualization, used in notebooks) 
*   Various ML Algorithms (Linear Regression, Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, CatBoosting, XGBoost)
*   Grid Search CV (for Hyperparameter Tuning) 
*   Git and GitHub (for Version Control) 
*   Conda (for Environment Management)

## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd ml_projects # or your project folder name
    ```

2.  **Set up a Virtual Environment**:
    Using Conda is recommended. Open your Anaconda Prompt or terminal.
    Navigate to the project directory.
    Create a new Conda environment with Python 3.8:
    ```bash
    conda create -p venv python=3.8 -y
    ```
    Activate the environment [15]:
    ```bash
    conda activate venv
    ```
    *(The `venv` directory will be created inside your project folder).*

3.  **Install Dependencies**:
    Install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the Project as a Package**:
    The project is structured to be installable as a Python package using `setup.py`. From the project root directory, run:
    ```bash
    pip install -e .
    ```
    *(The `-e` flag installs the package in "editable" mode, meaning changes to the source code are reflected without reinstallation).*

The utils.py file contains helpful functions, such as save_object and load_object, which are used across different components (e.g., in data transformation and model training/prediction) to save and load Python objects (like the preprocessor and trained model) using pickle.
Contributing
Feel free to fork the repository, explore the code, suggest improvements, and contribute!
