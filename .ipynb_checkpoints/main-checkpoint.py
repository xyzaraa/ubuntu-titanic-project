import pandas as pd
from src.preprocessing import load_data, clean_data
from src.model import train_model
from src.predict import make_submission


if __name__ == "__main__":
    df_train, df_test = load_data("data/train.csv", "data/test.csv")
    passenger_ids = pd.read_csv("data/test.csv")['PassengerId']
    df_train, df_test = clean_data(df_train, df_test)
    model = train_model(df_train)
    make_submission(model, df_test, passenger_ids)