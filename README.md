# project-uol
CSM500 - University of London Project

## Pipeline

### Get Data
Download data from Kaggle. Change the config.py KAGGLE_DATASET_NAME with the data to be analysed.

Then in data/kaggle_data.py run the following function

```commandline
download_kaggle_upload_to_sqlite()
```
or command in a terminal: 

```commandline
python -m data.kaggle_data
```

## Get Data Profile per table

In data_calibration/calibration_1st_layer.py run the following function:


```commandline
dataset_calibration()
```

or the following command:

```commandline

```