import pandas as pd
import sqlite3

# Load Data
df = pd.read_csv("../data/1.88_Million_US_Wildfires.csv", low_memory=False)
del df["Unnamed: 0"]

# Write cleansed DataFrame to SQLite
# It will be needed for the customer to do analysis in the Database by using the Web Application
conn = sqlite3.connect('../data/wildfires.sqlite')
df.to_sql('Fires', conn, if_exists='replace', index=False)
conn.close()

# The DISCOVERY_DATE field is in Julian Date format. I can convert this field to format we use everyday
df['DATE'] = pd.to_datetime(df['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')

# Separate Month and Day_of_week into different columns
df['MONTH'] = pd.DatetimeIndex(df['DATE']).month
df['DAY_OF_WEEK'] = df['DATE'].dt.day_name()

# Not all columns necessary, so let's drop unnecessary columns**
# I don't think that we need to have "FIRE_SIZE_CLASS" and "FIPS_NAME" anymore. They were kept for only analyze purpose.
df = df.drop(["DATE", "FIRE_SIZE_CLASS", "COUNTY", "FIPS_NAME", "STAT_CAUSE_CODE"], axis=1)

# Delete duplicated values
# Keep first duplicate row
df_clean = df.drop_duplicates(keep='first')

df.to_csv("../data/wildfire_cleansed.csv")

print("Data has been processed and saved in 'data' folder. ")
