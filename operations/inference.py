from astropy import time
import argparse
import dateutil.parser
import joblib
import numpy as np
from xgboost import XGBClassifier


##############################################################
def parse_args():
    """
    Takes parameter from user for prediction
    :return: Inputs
    """
    parser = argparse.ArgumentParser(
        description="Train XGBoost Model"
    )
    parser.add_argument(
        "--state",
        choices=['CA', 'NM', 'OR', 'NC', 'WY', 'CO', 'WA', 'MT', 'UT', 'AZ', 'SD',
                 'AR', 'NV', 'ID', 'MN', 'TX', 'FL', 'SC', 'LA', 'OK', 'KS', 'MO',
                 'NE', 'MI', 'KY', 'OH', 'IN', 'VA', 'IL', 'TN', 'GA', 'AK', 'ND',
                 'WV', 'WI', 'AL', 'NH', 'PA', 'MS', 'ME', 'VT', 'NY', 'IA', 'DC',
                 'MD', 'CT', 'MA', 'NJ', 'HI', 'DE', 'PR', 'RI'],
        help="State", required=True
    )
    parser.add_argument(
        "--date",
        help="Date of the estimated Wildfire",
        required=True
    )
    parser.add_argument(
        "--latitude",
        help="Latitude the place where Wildfire (possibly) occurred",
        required=True
    )
    parser.add_argument(
        "--longitude",
        help="Longitude the place where Wildfire (possibly) occurred",
        required=True
    )
    parser.add_argument(
        "--fire_size",
        help="Size of the Wildfire",
        required=True
    )

    # arg parsing debug
    # print(vars(parser.parse_args()))
    return parser.parse_args()


##############################################################
def get_features(le):
    """
    Transforms the given into numpy array for prediction
    :param le: Trained LabelEncoder
    :return: numpy array of features
    """
    args = parse_args()

    date = args.date
    dt = dateutil.parser.parse(date)
    fire_year = dt.year

    discovery_year = float(time.Time(dt).jd)
    fire_size = float(args.fire_size)
    latitude = float(args.latitude)
    longitude = float(args.longitude)

    state = args.state
    # convert to numeric form
    state_enc = le.fit_transform([state])[0]

    return np.array([[fire_year, discovery_year, fire_size, latitude, longitude, state_enc]])


##############################################################
def main():
    """
    Main function to execute trained model
    :return: None
    """
    # load trained labelEncoder
    le = joblib.load('../models/labelEncoder.joblib')
    le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))

    # Load models to see if it works properly
    model = XGBClassifier()
    model.load_model("../models/model_xgb.pkl")

    features = get_features(le)
    prediction = model.predict(features)
    print("Cause of the Wildfire: ", le_name_mapping[prediction[0]])


if __name__ == "__main__":
    main()
