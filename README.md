# End-to-End Machine Learning Project: Student Performance Indicator

## Project Overview

This project is an end-to-end machine learning application designed to **predict student performance test scores** based on various influencing factors [1]. It follows industry-standard best practices for building **modular, well-structured data science projects** from development to deployment [2].

The project covers essential stages of a machine learning lifecycle, including **data ingestion, data transformation, model training, evaluation, and deployment** [1, 2]. It also incorporates crucial aspects like **environment management, version control, exception handling, and logging** for robust and maintainable code [2].

## Problem Statement

The goal of this project is to understand how student performance test scores are **affected by variables** such as Gender, Ethnicity, Parental Level of Education, Lunch type, and Test Preparation Course completion [1]. The project aims to **predict the test score** based on these features [1].

The dataset used consists of **1000 rows and 8 columns**, including both categorical and numerical features [3].

## Project Structure

The project follows a **modular and component-based architecture** to ensure code reusability and maintainability [2, 4]. The main source code resides in the `src` folder [5].

The key directories and files within `src` are:

*   **`components`**: Contains distinct modules for each stage of the ML lifecycle [2, 6].
    *   `data_ingestion.py`: Handles reading the data from a source (initially CSV, planned for databases), splitting it into train and test sets, and saving the data [2, 7, 8].
    *   `data_transformation.py`: Contains code for data cleaning, preprocessing, and feature engineering, including handling missing values, categorical encoding, and scaling numerical features [2, 7, 9, 10].
    *   `model_trainer.py`: Responsible for training various machine learning models, evaluating them, selecting the best model, and saving it [2, 9, 11].
*   **`pipeline`**: Contains scripts to orchestrate the workflow by calling the components [2, 4].
    *   `train_pipeline.py`: Script for running the entire training process from data ingestion to model training [4].
    *   `predict_pipeline.py`: Script for handling new data inputs, preprocessing them using the saved objects, and generating predictions [2, 4, 12].
*   **`exception.py`**: Implements a **custom exception handling mechanism** to provide detailed error messages including script name and line number [2, 4, 13].
*   **`logger.py`**: Sets up a **logging configuration** to record execution flow and errors into a log file [2, 4, 14].
*   **`utils.py`**: Contains **reusable utility functions** for common tasks like saving and loading Python objects (models, preprocessors) [2, 4].
*   **`__init__.py`**: Present in package directories (like `src`, `components`, `pipeline`) to allow Python to recognize them as packages and enable imports [4-6, 15].

Other important files at the root level:

*   **`setup.py`**: Used to build the entire application as a **Python package**, allowing easy installation and distribution [15-18].
*   **`requirements.txt`**: Lists all the necessary Python libraries and dependencies required for the project [15, 16, 19].
*   **`.gitignore`**: Specifies files and directories (like `venv`, `artifacts`) that should be ignored by Git and not committed to the repository [15, 20, 21].
*   **`application.py`**: The entry point file for the web application deployment on AWS Elastic Beanstalk (copy of `app.py` renamed for deployment) [15, 23].
*   **`app.py`**: Contains the Flask application code for the web interface [12].
*   **`notebook/`**: Contains Jupyter notebooks used for initial Data Exploration (EDA) and model experimentation [24, 25].
    *   `data/student.csv`: The raw dataset file [24].
    *   `EDA.ipynb`: Notebook demonstrating the initial Exploratory Data Analysis [24].
    *   `Model Training.ipynb`: Notebook demonstrating initial model training and evaluation [24].
*   **`artifacts/`**: Directory created during the data ingestion and transformation process to store processed data and trained model/preprocessor objects [8, 15, 21].

## Technologies Used

*   Python [26]
*   Flask (for web application) [12]
*   scikit-learn (for machine learning algorithms and preprocessing) [12]
*   Pandas (for data manipulation) [5]
*   NumPy (for numerical operations) [5]
*   Seaborn and Matplotlib (for data visualization, used in notebooks) [3, 5]
*   Various ML Algorithms (Linear Regression, Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, CatBoosting, XGBoost) [15, 27]
*   Grid Search CV (for Hyperparameter Tuning) [15, 28]
*   Git and GitHub (for Version Control) [15, 29]
*   Conda (for Environment Management) [15, 30]

## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd ml_projects # or your project folder name
    ```
    *(Note: The source used 'ml_projects' initially, but the GitHub repo shown later was 'ml_project' [19, 35]. Use the actual repo name).*

2.  **Set up a Virtual Environment**:
    Using Conda is recommended [30]. Open your Anaconda Prompt or terminal.
    Navigate to the project directory [30].
    Create a new Conda environment with Python 3.8 [15]:
    ```bash
    conda create -p venv python=3.8 -y
    ```
    Activate the environment [15]:
    ```bash
    conda activate venv
    ```
    *(The `venv` directory will be created inside your project folder [30]).*

3.  **Install Dependencies**:
    Install the required libraries using the `requirements.txt` file [16]:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you have `pip install -e .` or `-e src` in your `requirements.txt`, it's recommended to **comment or remove that line** for initial dependency installation to avoid building the package prematurely [3, 18]).*

4.  **Install the Project as a Package**:
    The project is structured to be installable as a Python package using `setup.py` [16]. From the project root directory, run:
    ```bash
    pip install -e .
    ```
    *(The `-e` flag installs the package in "editable" mode, meaning changes to the source code are reflected without reinstallation).*

The utils.py file contains helpful functions, such as save_object and load_object, which are used across different components (e.g., in data transformation and model training/prediction) to save and load Python objects (like the preprocessor and trained model) using pickle [4, 15, 37].
Contributing
Feel free to fork the repository, explore the code, suggest improvements, and contribute!
