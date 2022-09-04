![Wildfire-cover](https://user-images.githubusercontent.com/43930582/188334716-0cd6666a-0fbc-47cb-83bb-59fd15262b6c.jpg)<h2 align="center">Wildfires</h2>
<p align="center">
  Developed by <a href="https://github.com/ByUnal"> M.Cihat Unal </a> 
</p>

## Overview

The API provides User Interface for the SQL aggregations and XGBoost model to predict cause of the Wildfires 
according to given inputs. [1.88 Million US Wildfires](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires/code) is used for the training.
This dataset includes several tables, but I used only "Fires" table for both training the model and SQL aggregations.

## Clone the repository
I used Git LFS while uploading the large size file. So, you can use it ```git clone``` to clone the repo.
However, you can also use ```git lfs clone ``` instead to better performance.


## Model Training Details

| identifier                                                     | learning rate | tree method | database                                                                                                          | 
|----------------------------------------------------------------|:-------------:|:-----------:|:------------------------------------------------------------------------------------------------------------------|
| [XGBoostClassifier](https://xgboost.readthedocs.io/en/stable/) |      0.5      |    hist     | [1.88 Million US Wildfires](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires/code) (Preprocessed) |

## Data Installation and Preparation
Firstly, create ```data``` and ```logs``` folder in the folder. Then, you need to download the [dataset](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires/code). Next, put it under the "data" folder.
You can see the steps I followed while preparing the data for the training below. Open the terminal in the project's directory first.
Then go inside "operation" folder.
- As I mentioned above, 1.88 Million US Wildfires in SQL format and includes lots of tables. We're going to extract Fires table only.
Then, we will convert SQL table to CSV file and save it. For this:
```
python extract_db_to_csv.py
```
It will save the DataFrame as "1.88_Million_US_Wildfires.csv" by taking specific columns into account. You can
examine the extracted CSV file.

Before training the model, we should extract useful information from the dataset and get rid of the unnecessary things to make our model successful.
- To prepare the data for training we need to do;
  - Convert columns into numerical format (if they are not already.)
  - Drop unnecessary columns
  - Drop duplicates
  - "DISCOVERY_SIZE" is in Julian Date format. So, convert it to date type(the type we used in every day). Then save in "DATE" column.
  - Divide "DATE" column as "MONTH" and "DAY_OF_WEEK" to increase feature number.
To do aforementioned steps:
```
python data_preprocessing.py
```

Lastly, the final DataFrame is ready for the training, and it will be extracted to "wildfire_cleansed.csv".
Also, final DataFrame saved at "wildfires.sqlite" to use in aggregations in UI.
Datasets can be found in **data** folder.

## Running the API

### via Docker
Build the image inside the Dockerfile's directory
```commandline
docker build -t canonical .
```
Then for running the image in local network
```commandline
docker run --network host --name canonical-cont canonical
```
Finally, you can reach the API from your browser by entering:
```bash
http://localhost:5000/
```

### via Python in Terminal

Open the terminal in the project's directory.
Install the requirements first.
```commandline
pip install -r requirements.txt
```
Then, run the main.py file
```commandline
python main.py
```

## User Interface
You will encounter with this page when you run the API successfully.

![image](https://user-images.githubusercontent.com/43930582/188329472-2514e603-1417-4512-8a7d-b72b2089f8d9.png)

## Train Model
Training can be done by using different parameters by using environment variable.
```commandline
python train.py --learnin_rate 0.3 --train_size 0.7 --tree_method hist --model_name canonical.pkl
```

## Inference
You can also use model for inference by giving inputs (all of them are required)
```
python inference.py --state NM --date 22.07.2008 --latitude 40.8213 --longitude -121.5397 --fire_size 9.0
```

## Examine my work further
You can glance my works with Jupyter Notebook in *notebooks* folder. Notebooks cover:
- Examining data in detail
- EDA (Exploratory Data Analysis)
- Data Cleaning
- Correlation Matrix
- RandomForrest and Decision Tree training
- Hyperparameter optimization.


## Improvement Suggestions
- SQL database is slow in loading. Therefore, MongoDB can be one of the efficient in terms of latency.
- Current model has %56.42 accuracy. Total label count 12. It may be too much to make correct prediction. Also, data is
imbalanced in terms of labels. Hence, label count can be lowered by defining new labels and distributing existed labels
into these labels. For example,
  - natural = ['Lightning']
  - accidental = ['Structure','Fireworks','Powerline','Railroad','Smoking','Children','Campfire','Equipment Use','Debris Burning']
  - malicious = ['Arson']
  - other = ['Missing/Undefined','Miscellaneous']
- Mlflow can be used to track ML operations.

## Citation
- Short, Karen C. 2017. Spatial wildfire occurrence data for the United States, 1992-2015 [FPAFOD20170508]. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4