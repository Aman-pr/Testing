hello this is testing repo
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=download




# I will now export the training pipeline that I built into a standalone Python script file.

# This script will include:

# - Loading the dataset

# - Preprocessing (label conversion, feature selection, imputation, optional time features)

# - Train/test split

# - RandomForest training and evaluation

# - Saving model + feature importances



script_content = """\

import os

import pickle

import pandas as pd

import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix



def to_binary(series):

    s = series.astype(str).str.strip().str.lower()

    return s.map(lambda x: 1 if x in ['yes','y','1','true','t'] else 0 if x in ['no','n','0','false','f'] else np.nan)



def main():

    # Path to dataset

    data_path = "cloudpredictionsystemproject.csv"  # Change if dataset is elsewhere

    df = pd.read_csv(data_path)

    

    # Label column

    label_col = None

    for candidate in ['CloudBurst Today', 'CloudBurstToday', 'CloudBurstTomorrow', 'CloudBurst Tomorrow']:

        if candidate in df.columns:

            label_col = candidate

            break

    if label_col is None:

        raise ValueError("No explicit CloudBurst label column found.")



    # Convert Yes/No to binary

    df['cloudburst_target'] = to_binary(df[label_col])

    mask = df['cloudburst_target'].notna()

    df = df.loc[mask].copy()



    # Parse Date if present

    date_col = None

    for c in df.columns:

        if 'date' in c.lower():

            date_col = c

            break

    if date_col:

        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')



    # Features: numeric only

    drop_cols = [label_col, 'cloudburst_target']

    if date_col: drop_cols.append(date_col)

    X = df.drop(columns=drop_cols, errors='ignore')

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_num = X[numeric_cols]

    

    # Impute missing

    imputer = SimpleImputer(strategy='median')

    X_imputed = pd.DataFrame(imputer.fit_transform(X_num), columns=X_num.columns)

    

    # Add simple time features if date exists

    if date_col and df[date_col].notna().sum() > 0:

        X_imputed['hour'] = df[date_col].dt.hour.fillna(0).astype(int)

        X_imputed['month'] = df[date_col].dt.month.fillna(0).astype(int)

    

    y = df['cloudburst_target'].astype(int).reset_index(drop=True)

    X_final = X_imputed.reset_index(drop=True)



    # Train/test split

    if date_col and df[date_col].notna().sum() > 0:

        df_sorted = df.sort_values(by=date_col)

        cutoff = int(len(df_sorted) * 0.8)

        train_idx = df_sorted.index[:cutoff]

        test_idx = df_sorted.index[cutoff:]

        X_train = X_final.loc[train_idx]

        X_test = X_final.loc[test_idx]

        y_train = y.loc[train_idx]

        y_test = y.loc[test_idx]

    else:

        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)



    # Train model

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

    rf.fit(X_train, y_train)



    # Evaluate

    y_pred = rf.predict(X_test)

    y_proba = rf.predict_proba(X_test)[:,1]

    print("Classification report:")

    print(classification_report(y_test, y_pred, digits=4))

    try:

        auc = roc_auc_score(y_test, y_proba)

        print(f"ROC AUC: {auc:.4f}")

    except:

        pass

    print("Confusion matrix:\\n", confusion_matrix(y_test, y_pred))



    # Save model + feature importances

    out_dir = Path("model_outputs")

    out_dir.mkdir(exist_ok=True)

    model_path = out_dir / "cloudburst_rf_model.pkl"

    with open(model_path, "wb") as f:

        pickle.dump(rf, f)

    feat_imp = pd.DataFrame({'feature': X_train.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

    feat_imp.to_csv(out_dir / "feature_importances.csv", index=False)

    print(f"Saved model to {model_path}, feature importances to {out_dir/'feature_importances.csv'}")



if __name__ == "__main__":

    main()

"""



# Save to a .py file

script_path = "/mnt/data/cloudburst_model_pipeline.py"

with open(script_path, "w") as f:

    f.write(script_content)



script_path
