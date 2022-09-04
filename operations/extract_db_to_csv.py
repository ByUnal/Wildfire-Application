import sqlite3
import pandas as pd

if __name__ == '__main__':
    DB_PATH = '../data/FPA_FOD_20170508.sqlite'

    # Create connection.
    cnx = sqlite3.connect(DB_PATH)

    # We need 'Fires' table among others for our purpose
    # Load table into Pandas dataframe by taking specific columns
    df = pd.read_sql_query("SELECT FIRE_YEAR, DISCOVERY_DATE, STAT_CAUSE_DESCR, STAT_CAUSE_CODE, FIRE_SIZE, "
                           "FIRE_SIZE_CLASS, LATITUDE, LONGITUDE, STATE, COUNTY, FIPS_NAME FROM Fires", cnx)

    # Extract DataFrame to CSV file
    df.to_csv("../data/1.88_Million_US_Wildfires.csv")
