import pandas as pd

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def clean_data(df_train, df_test):
    # ---- numeric ----
    age_median = df_train["Age"].median()
    fare_median = df_train["Fare"].median()

    df_train["Age"] = df_train["Age"].fillna(age_median)
    df_test["Age"] = df_test["Age"].fillna(age_median)
    df_test["Fare"] = df_test["Fare"].fillna(fare_median)

    # ---- categorical ----
    embarked_mode = df_train["Embarked"].mode()[0]
    df_train["Embarked"] = df_train["Embarked"].fillna(embarked_mode)
    df_test["Embarked"] = df_test["Embarked"].fillna(embarked_mode)

    # ---- encoding ----
    df_train["Sex"] = df_train["Sex"].map({"male": 0, "female": 1})
    df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})

    df_train["Embarked"] = df_train["Embarked"].map({"C":0, "Q":1, "S":2})
    df_test["Embarked"] = df_test["Embarked"].map({"C":0, "Q":1, "S":2})

    # ---- drop columns not used ----
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df_train = df_train.drop(columns=drop_cols)
    df_test = df_test.drop(columns=drop_cols)

    return df_train, df_test