import pandas as pd

def make_submission(model, df_test, passenger_ids, output_path='data/submission.csv'):
    predictions = model.predict(df_test)
    submission = pd.DataFrame({
        "PassengerId":passenger_ids,
        "Survived":predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")