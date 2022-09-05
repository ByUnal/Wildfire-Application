from astropy import time
import dateutil.parser
import logging
import numpy as np
import joblib
import sqlite3

from flask import Flask, request, render_template, redirect
from flask_paginate import Pagination, get_page_args
from xgboost import XGBClassifier

app = Flask(__name__)

# Set the configs for logging(errors only)
logging.basicConfig(filename='logs/error.log', level=logging.ERROR,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

STATE_LIST = ['CA', 'NM', 'OR', 'NC', 'WY', 'CO', 'WA', 'MT', 'UT', 'AZ', 'SD',
              'AR', 'NV', 'ID', 'MN', 'TX', 'FL', 'SC', 'LA', 'OK', 'KS', 'MO',
              'NE', 'MI', 'KY', 'OH', 'IN', 'VA', 'IL', 'TN', 'GA', 'AK', 'ND',
              'WV', 'WI', 'AL', 'NH', 'PA', 'MS', 'ME', 'VT', 'NY', 'IA', 'DC',
              'MD', 'CT', 'MA', 'NJ', 'HI', 'DE', 'PR', 'RI']

DB_PATH = 'data/wildfires.sqlite'
DATA_PER_PAGE = 15

# Will include query results
db_data = []

# Will include column names after query
headings = []


def get_information_per(query_results, offset=0, per_page=DATA_PER_PAGE):
    """

    :param query_results: Includes all query results
    :param offset: index to start
    :param per_page: number of data will be shown in specific page
    :return: Specific part of results to show in UI
    """
    return query_results[offset: offset + per_page]


def fetch_data(sql_query):
    """
    Makes SQL query
    :param sql_query: user's SQL query
    :return: data and column names
    """
    global db_data
    global headings
    try:
        with sqlite3.connect('data/wildfires.sqlite') as con:
            # Initialize cursor for sql queries
            cursor = con.cursor()

            # Fetch the results into list
            db_data = cursor.execute(sql_query).fetchall()

            # Brings Columns which are queried
            headings = [i[0] for i in cursor.description]

        return db_data, headings
    except sqlite3.OperationalError as s_error:
        app.logger.error("SQL Operation Error: %s", s_error)


@app.route("/fetchDatabase")
def fetch_database():
    """
    Fetches the results from database by using user input
    :return: Passes the operation to '/sqlResponse'
    """
    sql_query = request.args.get('sqlquery', '')

    fetch_data(sql_query)

    return redirect("sqlResponse")


@app.route('/sqlResponse')
def return_table():
    """
    Includes functions for SQL page of UI
    :return: Loads variable into (sql_response.html) html file
    """
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')

    total = len(db_data)
    information_per_page = get_information_per(db_data, offset=offset, per_page=DATA_PER_PAGE)
    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap5')

    return render_template('sql_response.html', information=information_per_page,
                           headings=headings, pagination=pagination)


@app.route('/')
def home():
    """
    Main endpoint. Initializes the UI
    :return: Loads variable into main(index.html) html file
    """
    if request.args:
        try:
            date = request.args.get('date', '')
            dt = dateutil.parser.parse(date)
            fire_year = dt.year
            month = dt.month
            day = dt.day

            discovery_year = float(time.Time(dt).jd)
            fire_size = float(request.args.get('fsize', ''))
            latitude = float(request.args.get('lat', ''))
            longitude = float(request.args.get('lon', ''))

            state = request.args.get('states', '')
            # convert to numeric form
            state_enc = le.fit_transform([state])[0]

            features = np.array([[fire_year, discovery_year, fire_size, latitude, longitude, state_enc, month, day]])
            prediction = model.predict(features)

            return render_template("index.html", predicton=le_name_mapping[prediction[0]],
                                   fsize=fire_size, state_list=STATE_LIST)
        except Exception as e:
            app.logger.error("Error while entering features: %s", e)
    else:
        return render_template('index.html', state_list=STATE_LIST)


if __name__ == "__main__":
    try:
        # load trained labelEncoder
        le = joblib.load('models/labelEncoder.joblib')
        le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))

        # Load models for prediction
        model = XGBClassifier()
        model.load_model("models/model.pkl")
    except FileNotFoundError as file_error:
        app.logger.critical("Files under the 'models' cannot be found Error: %s", file_error)

    # Launch the Flask dev server
    app.run(host="0.0.0.0")
