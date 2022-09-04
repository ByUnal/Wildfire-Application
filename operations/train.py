import argparse
import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from xgboost import XGBClassifier


##############################################################
def parse_args():
    """
    Takes parameter from user for training
    :return: Inputs
    """
    # print("Inside: parse_args()")
    parser = argparse.ArgumentParser(
        description="Train XGBoost Model"
    )
    parser.add_argument(
        "--learning_rate",
        default=0.7,
        help="Learning Rate parameter for XGBoostClassifier",
    )
    parser.add_argument(
        "--model_name",
        default="models",
        help="Name of the saved models. Please enter with its extension(ex.: .pt, .pkl)",
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="The output directory of the saved models during training",
    )
    parser.add_argument(
        "--cv",
        default=5,
        help="Cross Validation count in evaluation",
    )
    parser.add_argument(
        "--train_size",
        default=0.7,
        help="Proportion of data which will be used in training",
    )
    parser.add_argument(
        "--tree_method",
        default="hist",
        choices=["hist", "gpu_hist", "approx"],
        help="Estimator number for XGBoost",
    )
    parser.add_argument(
        "--verbosity",
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity of printing messages. Valid values of 0 (silent), 1 (warning), 2 (info), and 3 (debug)."
    )

    # arg parsing debug
    # print(vars(parser.parse_args()))
    return parser.parse_args()


###############################################################
def main():
    """
    Main function to start training
    :return: None
    """
    args = parse_args()

    # Load Data
    df = pd.read_csv("../data/wildfire_cleansed.csv", low_memory=False)
    del df["Unnamed: 0"]

    # Initialize labelEncoder
    le = LabelEncoder()

    # Transform String values to numeric values
    df['STATE'] = le.fit_transform(df['STATE'])
    df['DAY_OF_WEEK'] = le.fit_transform(df['DAY_OF_WEEK'])

    # This one also be  label column, so we need to create label mapping for cause prediction
    df['STAT_CAUSE_DESCR'] = le.fit_transform(df['STAT_CAUSE_DESCR'])

    # Save Label encoder int pkl file for using in restful api
    joblib.dump(le, '../models/labelEncoder.joblib', compress=9)

    # Create dataset for training
    X = df.drop(['STAT_CAUSE_DESCR'], axis=1)
    y = df['STAT_CAUSE_DESCR']

    # Data is imbalanced. Handle by resampling via SMOTE
    smote = SMOTE(random_state=26)
    X_res, y_res = smote.fit_sample(X, y)

    scaler = MinMaxScaler()
    # transform data
    X_res = scaler.fit_transform(X_res)

    print("Shape of features: ", X_res.shape)
    print("Shape of labels: ", y_res.shape)

    # 20% for testing, 80% for training
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size=float(args.train_size),
                                                        random_state=0, stratify=y_res)

    # fit models
    print("\nTraining the models...")
    model = XGBClassifier(objective='multi:softmax', tree_method=args.tree_method,
                          learning_rate=float(args.learning_rate), verbosity=int(args.verbosity))
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Classification Report: \n", classification_report(y_test, y_pred))

    # # Use Cross Validation to measure the success of the models
    cv = cross_validate(models, X_train, y_train, cv=int(args.cv))
    print("Test score after Cross Validation: ", cv["test_score"].mean())

    # Save models in pkl format
    model.save_model(f"../{args.model_dir}/{args.model_name}")


if __name__ == '__main__':
    main()
