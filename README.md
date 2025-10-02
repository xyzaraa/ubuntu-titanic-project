# 🚢 Titanic Survival Prediction
## ⚙️ Setup & Run

1. Clone repo
   ```bash
   git clone https://github.com/xyzaraa/ubuntu-titanic-project.git
   cd ubuntu-titanic-project

2. Make a venv
    ```bash 
    python3 -m venv dsciencenv
    source dsciencenv/bin/activate

3. Install Dependencies
    ```bash
    pip install -r requirements.txt

4. Run the pipeline
    ```bash
    python3 main.py

## Data Analysis

The analysis was conducted using **Exploratory Data Analysis (EDA)** with `pandas`, `matplotlib`, and `seaborn`.  
The Titanic dataset consists of 891 rows of training data and 418 rows of test data, with features such as:

- Pclass  
- Name  
- Sex  
- Age  
- SibSp  
- Parch  
- Ticket  
- Fare  
- Cabin  
- Embarked  

### Key Insights

- **Gender**  
  The most significant factor. Female passengers had a much higher survival rate compared to male passengers. Visualizations show a clear dominance of survival among females.

- **Pclass (Ticket Class)**  
  First-class passengers were more likely to survive compared to third-class passengers. This is likely due to better facilities and easier access to lifeboats.

- **Age**  
  The age distribution showed that children (<10 years) had a higher survival rate compared to adults.

- **Fare**  
  Higher ticket fares were positively correlated with survival, since they are linked to better cabin classes.

- **Family Size**  
  Passengers traveling alone were more vulnerable, while those with small family sizes (2–4 members) had better chances of survival.

- **Embarked**  
  Most passengers boarded at Southampton (S). There were differences in survival rates among embarkation points, but not as strong as gender and ticket class.


## Data Preprocessing

Preprocessing steps included:

- Filling missing values:  
  - `Age` → median from the training set  
  - `Fare` → median from the training set  
  - `Embarked` → mode from the training set  

- Encoding categorical features:  
  - `Sex` → male = 0, female = 1  
  - `Embarked` → C = 0, Q = 1, S = 2  

- Dropping irrelevant columns: `PassengerId`, `Name`, `Ticket`, `Cabin`


## Modeling

The baseline model used **Logistic Regression** from `scikit-learn`.

- The dataset was split into 80% training and 20% validation.  
- Logistic Regression was trained with `max_iter=200`.  
- Validation accuracy achieved: **~81%**.  

This baseline model is simple yet competitive, and provides a strong foundation for future improvements with feature engineering and advanced models.


## Next Steps

- Perform feature engineering (e.g., `FamilySize`, `IsAlone`, `Title` extracted from `Name`)  
- Try different models such as RandomForest, XGBoost, and LightGBM  
- Apply hyperparameter tuning (`GridSearchCV`, `RandomizedSearchCV`)  
- Deploy the model using Flask or FastAPI for serving predictions via API  


## Tech Stack

- Python 3.12  
- Pandas, Numpy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  
- WSL2 (Ubuntu 22.04)  



