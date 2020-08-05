# Databricks notebook source
# MAGIC %md # W261 Final Project - Airline Delays and Weather
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Summer 2020`__
# MAGIC ### Team 1:
# MAGIC * Sayan Das
# MAGIC * Kevin Hartman
# MAGIC * Hersh Solanki
# MAGIC * Nick Sylva

# COMMAND ----------

# MAGIC %md # Table of Contents
# MAGIC ### 1. Introduction
# MAGIC ### 2. Data Sources
# MAGIC ### 3. Data Lake Prep
# MAGIC ### 4. Exploratory Data Analysis
# MAGIC ### 5. Data Wrangling, Cleanup and Prep
# MAGIC ### 6. Model Exploration (Pipeline)
# MAGIC #### a. Feature Selection
# MAGIC #### b. Feature Engineering
# MAGIC #### c. Transformations (Encoding & Scaling)
# MAGIC #### d. Evaluation

# COMMAND ----------

# MAGIC %md # 1. Introduction
# MAGIC 
# MAGIC ## Objective
# MAGIC Our goal is to understand flight delays given information about the flight, and surrounding weather information. Flight delays are common, however, then can be a result of various amount of factors. We want to identify what these features may be, and create a model that can accurately predict whether the flight will be delayed by atleast 15 minutes, or not. We attempt to use multiple models, as well as hyperparameter turning to get our final result.
# MAGIC 
# MAGIC ## Testing Approach 
# MAGIC We decided to partition our dataset by years, where our training dataset consisted of data from 2015-2017, our validation data is from 2018, and our testing data is from 2019. The reason we split the data like this is because we don't want to to use future data to predict past data.  Furthermore, it is essential that all of the testing data is not sampled from any data that is in the future, for otherwise the model would not be practically useful.  
# MAGIC 
# MAGIC Note for the evaluation metric that is used, it appears that other literature on this topic have used the accuracy metric as the ideal metric to gauge model performance (Ye, 2020, Abdulwahab, 2020).  Therefore, our team has also decided to adopt accuracy as the de facto metric of comparison to determine which model is performing the best.  In addition, while it is important to minimize both the number of false positives and false negatives, our group has decided to prioritize minimizing the number of false positives, as we do not want to tell an individual that there is a delay when there actually is not a delay, as that could cause the individual to miss the flight which is the worst possible scenario.  Indeed, even if there a large number of false negatives, which implies that we tell the individual that there is a delay even though there is not a delay, then this outcome does not have as a negative impact on the user compared to them potentially missing their flight.  
# MAGIC 
# MAGIC ## Baseline Model
# MAGIC For our baseline model, we decided to use a logistic regression model which just predicts on the binary variable of a delay greater than 15 minutes (DEP_DEL15).  This raw model was able to produce an accuracy score of 0.8154 on the validation data, but it unfortunately predicted a fairly large number of false positives, namely 4053.  This is large with respect to other baseline model of random forest, which produced a similar accuracy score of 0.8156.  However, it produced a much smaller number of false positives, namely 426.
# MAGIC 
# MAGIC Overall, in order to be practically useful, our model should have an accuracy score that exceeds 0.8 with a miniscule number of false positives.  If there are a large number of false positives, then the veracity of our model is incredibly questionable.  
# MAGIC 
# MAGIC ## Limitations
# MAGIC Some of the limitations of our model include our model not predicting on different severities of delay.  From a user perspective, it is more beneficial have different levels of delay based on how long their will be a delay.  By only predicting whether there is a delay or not, it is difficult for the individual to truly manage their schedule to accomodate for the flight delay.  This distinction between different magnitudes of delay will especially have prominent impacts on airports that have a lot of traffic.

# COMMAND ----------

# MAGIC %md # 2. Data Sources
# MAGIC 
# MAGIC ## Airline delays 
# MAGIC ### Bureau of Transportation Statistics
# MAGIC https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236   
# MAGIC https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC 
# MAGIC Dates covered in dataset: 2015 - 2019
# MAGIC 
# MAGIC ### Additional sources
# MAGIC This might be useful in matching station codes to airports:
# MAGIC 1. http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html
# MAGIC 2. https://www.world-airport-codes.com/

# COMMAND ----------

# MAGIC %md # 3. Data Lake Prep
# MAGIC Note that our group worked with the Delta Lake.  The Delta Lake brings reliability, performance, and lifecycle management to data lakes. The pitch with Delta Lake is that there is no more malformed data ingestion, difficulty deleting data for compliance, or issues modifying data for change data capture.  The Delta Lake allowed us to establish different versions of our data - bronze, silver, and gold - while allowing us to read and write data concurrently with ACID Transactions.  Bronze data is the raw data in the native format upon ingestion, silver data is sanitized and cleaned data, and finally, our gold data is the data that is pushed to our Delta Lake for modeling use.  Moreover, the Delta Lake transaction log records details about any changes made to our data, providing us with a full audit history.  
# MAGIC 
# MAGIC **The setup for the Delta Lake and the initial imports are shown below:**

# COMMAND ----------

# MAGIC %md ## Notebook Imports

# COMMAND ----------

!pip install networkx
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
from pyspark.sql.window import Window

from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import time
import numpy as np 
import pandas as pd
import seaborn as sns
from pytz import timezone 
from datetime import  datetime, timedelta 
import os
import networkx as nx
from delta.tables import DeltaTable
from copy import deepcopy

# Model Imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

%matplotlib inline
sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md ## Configure access to staging areas - bronze & silver

# COMMAND ----------

username = "kevin2"
dbutils.widgets.text("username", username)
spark.sql(f"CREATE DATABASE IF NOT EXISTS airline_delays_{username}")
spark.sql(f"USE airline_delays_{username}")

flights_loc = f"/airline_delays/{username}/DLRS/flights/"
flights_3m_loc = f"/airline_delays/{username}/DLRS/flights_3m/"
flights_6m_loc = f"/airline_delays/{username}/DLRS/flights_6m/"
airports_loc = f"/airline_delays/{username}/DLRS/airports/"
weather_loc = f"/airline_delays/{username}/DLRS/weather/"
stations_loc = f"/airline_delays/{username}/DLRS/stations/"

spark.conf.set("spark.sql.shuffle.partitions", 8)

# COMMAND ----------

# MAGIC %md # WARNING: DO NOT RUN CODE IN THE NEXT SECTION UNLESS YOU NEED TO RECONSTRUCT THE BRONZE AND SILVER  DATA LAKES
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Please start execution from Section 4. EDA to load from already processed (silver) data

# COMMAND ----------

# MAGIC %md ## Download and store data locally (Bronze)

# COMMAND ----------

# MAGIC %md #### Clear raw (bronze) landing zone

# COMMAND ----------

dbutils.fs.rm(flights_loc + "raw", recurse=True)
dbutils.fs.rm(flights_3m_loc + "raw", recurse=True)
dbutils.fs.rm(flights_6m_loc + "raw", recurse=True)
dbutils.fs.rm(airports_loc + "raw", recurse=True)
dbutils.fs.rm(airports_loc + "land", recurse=True)
dbutils.fs.rm(airports_loc + "stage", recurse=True)
dbutils.fs.rm(weather_loc + "raw", recurse=True)
dbutils.fs.rm(stations_loc + "raw", recurse=True)

# COMMAND ----------

# MAGIC %md #### Ingest Data from Flights, Weather and Stations
# MAGIC 
# MAGIC Data from mids-w261 project folder

# COMMAND ----------

source_path = "dbfs:/mnt/mids-w261/data/datasets_final_project/"
flights_source_path = source_path + "parquet_airlines_data/"
flights_3m_source_path = source_path + "parquet_airlines_data_3m/"
flights_6m_source_path = source_path + "parquet_airlines_data_6m/"
weather_source_path = source_path + "weather_data/"

demo8_source_path = "dbfs:/mnt/mids-w261/data/DEMO8/"
stations_source_path = demo8_source_path + "gsod/"

# COMMAND ----------

# MAGIC %md #### Airports
# MAGIC Data from OpenFlights: https://openflights.org/data.html

# COMMAND ----------

airports_source_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

# COMMAND ----------

# MAGIC %python
# MAGIC os.environ['airports_source_url'] = airports_source_url

# COMMAND ----------

# MAGIC %sh 
# MAGIC wget $airports_source_url
# MAGIC ls

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/airports.dat", 
              airports_loc + "land/airports.dat")

# COMMAND ----------

airports_land_df = spark.read.option("header", "false").csv(airports_loc + "land/airports.dat", sep = ",")

# COMMAND ----------

# MAGIC %md #### Let's rename some of the Airport columns 

# COMMAND ----------

def add_airport_data_headers(df):
  return (df.select(f.col("_c0").alias("AIRPORT_ID"),
                    f.col("_c1").alias("AIRPORT_NAME"),
                    f.col("_c2").alias("AIRPORT_CITY"),
                    f.col("_c3").alias("AIRPORT_COUNTRY"),
                    f.col("_c4").alias("IATA"),
                    f.col("_c5").alias("ICAO"),
                    f.col("_c6").alias("AIRPORT_LAT"),
                    f.col("_c7").alias("AIRPORT_LONG"),
                    f.col("_c8").alias("AIRPORT_ALT"),
                    f.col("_c9").alias("AIRPORT_TZ_OFFSET"),
                    f.col("_c10").alias("AIRPORT_DST"),
                    f.col("_c11").alias("AIRPORT_TZ_NAME"),
                    f.col("_c12").alias("AIRPORT_TYPE"),
                    f.col("_c13").alias("AIRPORT_SOURCE")
                   )
         )
  
airports_stage_df = add_airport_data_headers(airports_land_df)

# COMMAND ----------

airports_stage_path = airports_loc + "stage/"

# COMMAND ----------

(airports_stage_df.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("AIRPORT_TZ_NAME")
 .save(airports_stage_path))

# COMMAND ----------

# MAGIC %md #### Ingest data from staging or source zone (wherever data currently resides) and place into the bronze zone (in raw format)

# COMMAND ----------

flights_source_df = spark.read.option("header", "true").parquet(flights_source_path + "*.parquet")
flights_source_3m_df = spark.read.option("header", "true").parquet(flights_3m_source_path + "*.parquet")
flights_source_6m_df = spark.read.option("header", "true").parquet(flights_6m_source_path + "*.parquet")
weather_source_df = spark.read.option("header", "true").parquet(weather_source_path + "*.parquet")
stations_source_df = spark.read.option("header", "true").csv(stations_source_path + "stations.csv.gz")
airports_source_df = spark.read.option("header", "true").parquet(airports_stage_path)

# COMMAND ----------

flights_source_df.write.format("delta").mode("overwrite").save(flights_loc + "raw")
flights_source_3m_df.write.format("delta").mode("overwrite").save(flights_3m_loc + "raw")
flights_source_6m_df.write.format("delta").mode("overwrite").save(flights_6m_loc + "raw")
weather_source_df.write.format("delta").mode("overwrite").save(weather_loc + "raw")
stations_source_df.write.format("delta").mode("overwrite").save(stations_loc + "raw")
airports_source_df.write.format("delta").mode("overwrite").save(airports_loc + "raw")

# COMMAND ----------

# MAGIC %md #### Re-read raw files as delta lake

# COMMAND ----------

flights_raw_df = spark.read.format("delta").load(flights_loc + "raw")
flights_3m_raw_df = spark.read.format("delta").load(flights_3m_loc + "raw")
flights_6m_raw_df = spark.read.format("delta").load(flights_6m_loc + "raw")
weather_raw_df = spark.read.format("delta").load(weather_loc + "raw")
stations_raw_df = spark.read.format("delta").load(stations_loc + "raw")
airports_raw_df = spark.read.format("delta").load(airports_loc + "raw")

# COMMAND ----------

# MAGIC %md ## Perform data processing for analysis (Silver)

# COMMAND ----------

# MAGIC %md #### Clear processed staging folders

# COMMAND ----------

dbutils.fs.rm(flights_loc + "processed", recurse=True)
dbutils.fs.rm(flights_3m_loc + "processed", recurse=True)
dbutils.fs.rm(flights_6m_loc + "processed", recurse=True)
dbutils.fs.rm(airports_loc + "processed", recurse=True)
dbutils.fs.rm(weather_loc + "processed", recurse=True)
dbutils.fs.rm(stations_loc + "processed", recurse=True)

# COMMAND ----------

# We'll distribute flight data across 31 partitions
spark.conf.set("spark.sql.shuffle.partitions", 31)

# COMMAND ----------

# MAGIC %md #### Flight data pre-processing

# COMMAND ----------

def process_flight_data(df):
  cols = df.columns
  cols.append('IN_FLIGHT_AIR_DELAY')
  return (
    df
    .withColumn('IN_FLIGHT_AIR_DELAY', f.lit(df['ARR_DELAY'] - df['DEP_DELAY'] )) # this column is the time difference between arrival and departure and does not include total flight delay
    .select(cols)
    )

flights_processed_df = process_flight_data(flights_raw_df)
flights_3m_processed_df = process_flight_data(flights_3m_raw_df)
flights_6m_processed_df = process_flight_data(flights_6m_raw_df)

# COMMAND ----------

(flights_3m_processed_df.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("DAY_OF_MONTH")
 .save(flights_3m_loc + "processed"))

parquet_table = f"parquet.`{flights_3m_loc}processed`"
partitioning_scheme = "DAY_OF_MONTH string"

DeltaTable.convertToDelta(spark, parquet_table, partitioning_scheme)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_3m_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_3m_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_3m/processed"

# COMMAND ----------

(flights_6m_processed_df.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("DAY_OF_MONTH")
 .save(flights_6m_loc + "processed"))

parquet_table = f"parquet.`{flights_6m_loc}processed`"
partitioning_scheme = "DAY_OF_MONTH string"

DeltaTable.convertToDelta(spark, parquet_table, partitioning_scheme)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_6m_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_6m_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_6m/processed"

# COMMAND ----------

(flights_processed_df.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("DAY_OF_MONTH")
 .save(flights_loc + "processed"))

parquet_table = f"parquet.`{flights_loc}processed`"
partitioning_scheme = "DAY_OF_MONTH string"

DeltaTable.convertToDelta(spark, parquet_table, partitioning_scheme)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights/processed"

# COMMAND ----------

# MAGIC %md
# MAGIC  #### Weather data pre-processing

# COMMAND ----------

# We'll distribute weather data across 15195 partitions
spark.conf.set("spark.sql.shuffle.partitions", 15195)

# COMMAND ----------

def process_weather_data(df):
  df = (df
        .withColumn("STATION", f.lpad(df.STATION, 11, '0'))
        .withColumnRenamed("DATE", "WEATHER_DATE")
        .withColumnRenamed("SOURCE", "WEATHER_SOURCE")
        .withColumnRenamed("STATION", "WEATHER_STATION")
        .withColumnRenamed("LATITUDE", "WEATHER_LAT")
        .withColumnRenamed("LONGITUDE", "WEATHER_LON")
        .withColumnRenamed("ELEVATION", "WEATHER_ELEV")
        .withColumnRenamed("NAME", "WEATHER_NAME")
        .withColumnRenamed("REPORT_TYPE", "WEATHER_REPORT_TYPE")
        .withColumnRenamed("CALL_SIGN", "WEATHER_CALL_SIGN")
       )

  return df

weather_processed_df = process_weather_data(weather_raw_df)



# COMMAND ----------

(weather_processed_df.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("WEATHER_STATION")
 .save(weather_loc + "processed"))

parquet_table = f"parquet.`{weather_loc}processed`"
partitioning_scheme = "WEATHER_STATION string"

DeltaTable.convertToDelta(spark, parquet_table, partitioning_scheme)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS weather_processed;
# MAGIC 
# MAGIC CREATE TABLE weather_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/weather/processed"

# COMMAND ----------

# MAGIC %md #### Station data pre-processing

# COMMAND ----------

# We'll distribute station data across the 74 'states'
spark.conf.set("spark.sql.shuffle.partitions", 74)

# COMMAND ----------

def process_station_data(df):
  return (df.withColumn("STATION_USAF_WBAN", f.concat(f.col("usaf"), f.col("wban")))
         .select(f.col("STATION_USAF_WBAN"),
                    f.col("name").alias("STATION_NAME"),
                    f.col("country").alias("STATION_COUNTRY"),
                    f.col("state").alias("STATION_STATE"),
                    f.col("call").alias("STATION_CALL"),
                    f.col("lat").alias("STATION_LAT"),
                    f.col("lon").alias("STATION_LONG"),
                    f.col("elev").alias("AIRPORT_ELEV"),
                    f.col("begin").alias("STATION_BEGIN"),
                    f.col("end").alias("STATION_END")
                   )
         )
  
stations_processed_df = process_station_data(stations_raw_df)

# COMMAND ----------

(stations_processed_df.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("STATION_STATE")
 .save(stations_loc + "processed"))

parquet_table = f"parquet.`{stations_loc}processed`"
partitioning_scheme = "STATION_STATE string"

DeltaTable.convertToDelta(spark, parquet_table, partitioning_scheme)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS stations_processed;
# MAGIC 
# MAGIC CREATE TABLE stations_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/stations/processed"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airport data pre-processing

# COMMAND ----------

# We'll distribute airport data across the 15 timezone names
spark.conf.set("spark.sql.shuffle.partitions", 15)

# COMMAND ----------

def process_airport_data(df):
  cols = df.columns
  return (df.select(cols)
          .where('AIRPORT_COUNTRY = "United States" OR AIRPORT_COUNTRY = "Puerto Rico" OR AIRPORT_COUNTRY = "Virgin Islands"')
         )

airports_processed_df = process_airport_data(airports_raw_df)

# COMMAND ----------

(airports_raw_df.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("AIRPORT_TZ_NAME")
 .save(airports_loc + "processed"))

parquet_table = f"parquet.`{airports_loc}processed`"
partitioning_scheme = "AIRPORT_TZ_NAME string"

DeltaTable.convertToDelta(spark, parquet_table, partitioning_scheme)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS airports_processed;
# MAGIC 
# MAGIC CREATE TABLE airports_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/airports/processed"

# COMMAND ----------

# Back to the defaults
spark.conf.set("spark.sql.shuffle.partitions", 8)

# COMMAND ----------

# MAGIC %md # 4. EDA
# MAGIC 
# MAGIC We did EDA on two main datasets.  We did it on the flights dataset, and we also did it on the weather dataset.  Note that we created a custom EDA function that provides counts on key metrics such as unique and missing values along with associated mean, minimums, and maximums for each column.  The function also provided bar graphs for each component of each feature (e.g counts for the number of records in each month).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load data from processed (silver) staging area

# COMMAND ----------

flights_processed = spark.sql("select * from flights_processed")
weather_processed = spark.sql("select * from weather_processed")
stations_processed = spark.sql("select * from stations_processed")
airports_processed = spark.sql("select * from airports_processed")

# COMMAND ----------

# MAGIC %md ## Flights EDA
# MAGIC 
# MAGIC Schema for flights: https://annettegreiner.com/vizcomm/OnTime_readme.html

# COMMAND ----------

flights_processed.printSchema()

# COMMAND ----------

f'{flights_processed.count():,}'

# COMMAND ----------

# MAGIC %md **We will perform EDA on a sample distribution from the first three  years**
# MAGIC 
# MAGIC Note, we initially ran these routines on the first three months of data for the project exploration. After this early look, and the decision to use the first three years of data for training, we moved to perform our EDA review again on a random sample from this distribution.

# COMMAND ----------

# we will perform EDA on a random sample taken from the first three years
#flights_sample = flights_processed.where('(ORIGIN = "ORD" OR ORIGIN = "ATL") AND QUARTER = 1 and YEAR = 2015').sample(False, .10, seed = 42)
flights_sample = flights_processed.where('YEAR IN (2015, 2016, 2017)').sample(False, .01, seed = 42)

# COMMAND ----------

f'{flights_sample.count()}'

# COMMAND ----------

display(flights_sample)

# COMMAND ----------

# MAGIC %md **Review our sample for missing info**

# COMMAND ----------

# review missing info
def info_missing_table(df_pd):
    """Input pandas dataframe and Return columns with missing value and percentage"""
    mis_val = df_pd.isnull().sum() #count total of null in each columns in dataframe
#count percentage of null in each columns
    mis_val_percent = 100 * df_pd.isnull().sum() / len(df_pd) 
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) 
 #join to left (as column) between mis_val and mis_val_percent
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'}) 
#rename columns in table
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1) 
        
    print ("DataFrame has " + str(df_pd.shape[1]) + " columns.\n"    #.shape[1] : just view total columns in dataframe  
    "There are " + str(mis_val_table_ren_columns.shape[0]) +              
    " columns that have missing values.") #.shape[0] : just view total rows in dataframe
    return mis_val_table_ren_columns

# COMMAND ----------

flights_sample_panda_df = flights_sample.toPandas()

missing_recs = info_missing_table(flights_sample_panda_df)
missing_recs

# COMMAND ----------

missing_recs[missing_recs['% of Total Values'] > 95]

# COMMAND ----------

# MAGIC %md **As a first pass will drop all columns that have more than 99% missing information as these records will be impossible to impute**

# COMMAND ----------

cols_to_drop = missing_recs[missing_recs['% of Total Values'] > 99].index.values
cols_to_drop

# COMMAND ----------

cols_to_drop = ['DIV4_AIRPORT', 'DIV4_TOTAL_GTIME', 'DIV3_TOTAL_GTIME',
       'DIV3_WHEELS_ON', 'DIV3_AIRPORT_SEQ_ID', 'DIV3_AIRPORT_ID',
       'DIV3_AIRPORT', 'DIV3_TAIL_NUM', 'DIV3_WHEELS_OFF',
       'DIV4_AIRPORT_ID', 'DIV4_AIRPORT_SEQ_ID', 'DIV4_WHEELS_ON',
       'DIV3_LONGEST_GTIME', 'DIV4_LONGEST_GTIME', 'DIV5_TAIL_NUM',
       'DIV4_TAIL_NUM', 'DIV5_AIRPORT', 'DIV5_WHEELS_OFF',
       'DIV5_LONGEST_GTIME', 'DIV5_TOTAL_GTIME', 'DIV5_WHEELS_ON',
       'DIV5_AIRPORT_SEQ_ID', 'DIV5_AIRPORT_ID', 'DIV4_WHEELS_OFF',
       'DIV2_TAIL_NUM', 'DIV2_WHEELS_OFF', 'DIV2_WHEELS_ON',
       'DIV2_AIRPORT', 'DIV2_TOTAL_GTIME', 'DIV2_LONGEST_GTIME',
       'DIV2_AIRPORT_SEQ_ID', 'DIV2_AIRPORT_ID', 'DIV_ARR_DELAY',
       'DIV_ACTUAL_ELAPSED_TIME', 'DIV1_TAIL_NUM', 'DIV1_WHEELS_OFF',
       'DIV_DISTANCE', 'DIV_REACHED_DEST', 'DIV1_AIRPORT_SEQ_ID',
       'DIV1_TOTAL_GTIME', 'DIV1_WHEELS_ON', 'DIV1_AIRPORT_ID',
       'DIV1_AIRPORT', 'DIV1_LONGEST_GTIME', 'LONGEST_ADD_GTIME',
       'TOTAL_ADD_GTIME', 'FIRST_DEP_TIME']

# COMMAND ----------

cols = flights_sample_panda_df.columns
cols_to_keep = []
remove_cols = set(cols_to_drop)
for col in cols:
  if not col in remove_cols:
    cols_to_keep.append(col)

# COMMAND ----------

# MAGIC %md **Custom EDA function**

# COMMAND ----------

# Custom-made class to assist with EDA on this dataset
# The code is generalizable. However, specific decisions on plot types were made because
# most features are categorical
class Analyze:
    def __init__(self, df):
        self.df = df
    
    def remove_df():
        self.df = None
        gc.collect()
        
    def print_eda_summary(self):
        #sns.set(rc={'figure.figsize':(10*2,16*8)})
        sns.set()
        i=0
        fig, ax = plt.subplots(nrows=round(len(self.df.columns)), ncols=2, figsize=(16,6*round(len(self.df.columns))))
        all_cols=[]
        for col in self.df.columns:
            if self.df[col].dtype.name == 'object'  or self.df[col].dtype.name == 'category': 
                self.df[col] = self.df[col].astype('str')
            all_cols.append(col)
            max_len = self.df[col].nunique()
            if max_len > 10:
                max_len = 10
            g=sns.countplot(y=self.df[col].fillna(-1), hue=self.df['DEP_DEL15'], order=self.df[col].fillna(-1).value_counts(dropna=False).iloc[:max_len].index, ax=ax[i][0])
            g.set_xlim(0,self.df.shape[0])
            plt.tight_layout()
            ax[i][0].title.set_text(col)
            ax[i][0].xaxis.label.set_visible(False)
            xlabels = ['{:,.0f}'.format(x) + 'K' for x in g.get_xticks()/1000]
            g.set_xticklabels(xlabels)
            ax[i][1].axis("off")
            # Basic info
            desc = self.df[col].describe()
            summary = "DESCRIPTION\n   Name: {:}\n   Type: {:}\n  Count: {:}\n Unique: {:}\nMissing: {:}\nPercent: {:2.3f}".format(
                desc.name.ljust(50), str(desc.dtype).ljust(10), self.df[col].count(), self.df[col].nunique(),
                ('yes' if self.df[col].hasnans else 'no'), (1-self.df[col].count()/self.df.shape[0])*100)
            ax[i][1].text(0, 1, summary, verticalalignment="top", family='monospace', fontsize=12)
            analysis=[]
            if self.df[col].dtype.name == 'object': 
                # additional analysis for categorical variables
                if len(self.df[col].str.lower().unique()) != len(self.df[col].unique()):
                    analysis.append("- duplicates from case\n")
                # look for HTML escape characters (&#x..;)
                # and unicode characters (searching for: anything not printable)
                self.df_bad = self.df[col][self.df[col].str.contains(r'[\x00-\x1f]|&#x\d\d;', regex=True, na=True)]
                if len(self.df_bad) - self.df.shape[0] - self.df[col].count()>0:
                    analysis.append("- illegal chars: {:}\n".format(len(self.df_bad) - self.df.shape[0] - self.df[col].count()))
                # find different capitalizations of "unknown"
                # if more than one present, need to read as string, turn to lowercase, then make categorical
                self.df_unknown = self.df[col].str.lower() == 'unknown'
                unknowns = self.df[col][self.df_unknown].unique()
                if len(unknowns) > 1:
                    analysis.append("- unknowns\n  {:}\n".format(unknowns))
                if len(''.join(analysis)) > 0:
                    ax[i][1].text(.5, .85, 'FINDINGS\n'+''.join(analysis), verticalalignment="top", family='monospace', fontsize=12)
            else:
                # Stats for numeric variables
                statistics = "STATS\n   Mean: {:5.4g}\n    Std: {:5.4g}\n    Min: {:5.4g}\n    25%: {:5.4g}\n    50%: {:5.4g}\n    75%: {:5.4g}\n    Max: {:5.4g}".format(
                    desc.mean(), desc.std(), desc.min(), desc.quantile(.25), desc.quantile(.5), desc.quantile(.75), desc.max())
                ax[i][1].text(.5, .85, statistics, verticalalignment="top", family='monospace', fontsize=12)

            # Top 5 and bottom 5 unique values or all unique values if < 10
            if self.df[col].nunique() <= 10:
                values = pd.DataFrame(list(zip(self.df[col].value_counts(dropna=False).keys().tolist(),
                                         self.df[col].value_counts(dropna=False).tolist())),
                                columns=['VALUES', 'COUNTS'])
                values = values.to_string(index=False)
                ax[i][1].text(0, .6, values, verticalalignment="top", family='monospace', fontsize=12)
            else:
                values = pd.DataFrame(list(zip(self.df[col].value_counts(dropna=False).iloc[:5].keys().tolist(),
                                         self.df[col].value_counts(dropna=False).iloc[:5].tolist())),
                                columns=['VALUES', 'COUNTS'])
                mid_row = pd.DataFrame({'VALUES':[":"],
                                        'COUNTS':[":"]})
                bot_values = pd.DataFrame(list(zip(self.df[col].value_counts(dropna=False).iloc[-5:].keys().tolist(),
                                         self.df[col].value_counts(dropna=False).iloc[-5:].tolist())),
                                columns=['VALUES', 'COUNTS'])
                values = values.append(mid_row)
                values = values.append(bot_values)
                values = values.to_string(index=False)
                ax[i][1].text(0, .6, values, verticalalignment="top", family='monospace', fontsize=12)
            i=i+1
        fig.show()

# COMMAND ----------

# MAGIC %md **Now we will analyze the remaining columns for their distribution of values, descriptive statistics and potential for explanatory power**

# COMMAND ----------

df_to_analyze = deepcopy(flights_sample_panda_df[cols_to_keep])
analyzer = Analyze(df_to_analyze)
analyzer.print_eda_summary()

# COMMAND ----------

# MAGIC %md Based on the output from above, there does not appear to much predictive value from the columns of above.  However, we are able to see the distribution of values for each of the features.  It will be interesting to see the output of this function on the weather data.
# MAGIC 
# MAGIC Let's take a look at a heatmap:

# COMMAND ----------

sns.set(rc={'figure.figsize':(100,100)})
sns.heatmap(flights_sample_panda_df[cols_to_keep].corr(), cmap='RdBu_r', annot=True, center=0.0)
sns.set(rc={'figure.figsize':(10,10)})

# COMMAND ----------

# MAGIC %md There does not appear to be any noteworthy correlations between columns on the flights data based on the heatmap above.

# COMMAND ----------

# MAGIC %md ### Analysis on departures that are on time or early

# COMMAND ----------

flights_sample.where('DEP_DELAY < 0').count() / flights_sample.count() # This statistic explains that 58% of flights depart earlier

# COMMAND ----------

flights_sample.where('DEP_DELAY == 0').count() / flights_sample.count()  # This statistic explains that 5.4% of flights depart EXACTLY on time

# COMMAND ----------

bins, counts = flights_sample.select('DEP_DELAY').where('DEP_DELAY <= 0').rdd.flatMap(lambda x: x).histogram(100)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = flights_sample.select('DEP_DELAY').where('DEP_DELAY <= 0 AND DEP_DELAY > -25').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md ### Analysis on departures that are delayed

# COMMAND ----------

bins, counts = flights_sample.select('DEP_DELAY').where('DEP_DELAY > 0').rdd.flatMap(lambda x: x).histogram(100)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = flights_sample.select('DEP_DELAY').where('DEP_DELAY > 0 AND DEP_DELAY < 300').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = flights_sample.select('DEP_DELAY').where('DEP_DELAY > -25 AND DEP_DELAY < 50').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md Analyzing the plot above, it is apparent that the distribution is right-skewed, implying that there is a heavy amount of data that is delayed and shifting the distribution towards the right, so therefore the median departure delay is higher than the mean.  Intuitively, this makes sense, for it is more likely that a flight will depart a day later compared to a flight departing a day earlier.  Moreover, we can see that much of the data revolves around flights that depart early or on time, and it is possible that the data is from airports that are smaller with less load; this would explain how the flights would be more likely to depart at an earlier time.  Further analysis of the locations of the actual airports and the distribution of these airports is necessary.

# COMMAND ----------

# MAGIC %md ### Arrival Delay

# COMMAND ----------

# MAGIC %md Next, we will look into visualizing arrival delay.  However, we should note that arrival delay also encompasses any delay from the departure delay.  Therefore, we engineered a new column in our pre-processor (IN_FLIGHT_AIR_DELAY) that accounts for this discrepancy.

# COMMAND ----------

bins, counts = flights_sample.select('IN_FLIGHT_AIR_DELAY').where('IN_FLIGHT_AIR_DELAY > -50 AND IN_FLIGHT_AIR_DELAY < 50').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md We can see that there is a normal distribution that is centered around -5; this indicates that the flight makes up 5 minutes of time after departing from the airport.  In general, this is implying that flights are making up time in the air time.  Further analysis should look into analyzing the amount of time made up in the air based on distance to see if flights make up more delay time with longer flight distances.

# COMMAND ----------

# MAGIC %md ## Weather EDA
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532
# MAGIC 
# MAGIC 
# MAGIC Schema for weather: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf

# COMMAND ----------

f'{weather_processed.count():,}'

# COMMAND ----------

# MAGIC %md **Run EDA on Weather Sample**

# COMMAND ----------

# sample from the first three years
weather_sample = weather_processed.where('WEATHER_DATE >= TO_DATE("01/01/2015", "MM/dd/yyyy") AND WEATHER_DATE <= TO_DATE("12/31/2017", "MM/dd/yyyy")').sample(False, .001, seed = 42)
# subset to 1Q2015
#weather_subset = weather_processed.where('WEATHER_DATE >= TO_DATE("01/01/2015", "MM/dd/yyyy") AND WEATHER_DATE <= TO_DATE("03/31/2015", "MM/dd/yyyy")') 

# COMMAND ----------

f'{weather_sample.count():,}'

# COMMAND ----------

weather_sample.printSchema()

# COMMAND ----------

display(weather_sample)

# COMMAND ----------

# MAGIC %md **Review weather columns with missing data**

# COMMAND ----------

weather_sample_panda_df = weather_sample.toPandas()

missing_recs = info_missing_table(weather_sample_panda_df)
missing_recs

# COMMAND ----------

# MAGIC %md The weather data has encoded fields that we must parse out before we can perform our EDA. We will decode a few of these fields and flatten them out so we may review.

# COMMAND ----------

def decode_weather_fields(df):
  WND_col = f.split(df['WND'], ',')
  CIG_col = f.split(df['CIG'], ',')
  VIS_col = f.split(df['VIS'], ',')
  TMP_col = f.split(df['TMP'], ',')
  DEW_col = f.split(df['DEW'], ',')
  SLP_col = f.split(df['SLP'], ',')
  df = (df
    # WND Fields [direction angle, quality code, type code, speed rate, speed quality code]
    .withColumn('WND_DIRECTION_ANGLE', WND_col.getItem(0).cast('int')) # continuous
    .withColumn('WND_QUALITY_CODE', WND_col.getItem(1).cast('int')) # categorical
    .withColumn('WND_TYPE_CODE', WND_col.getItem(2).cast('string')) # categorical
    .withColumn('WND_SPEED_RATE', WND_col.getItem(3).cast('int')) # categorical
    .withColumn('WND_SPEED_QUALITY_CODE', WND_col.getItem(4).cast('int')) # categorical
    # CIG Fields
    .withColumn('CIG_CEILING_HEIGHT_DIMENSION', CIG_col.getItem(0).cast('int')) # continuous 
    .withColumn('CIG_CEILING_QUALITY_CODE', CIG_col.getItem(1).cast('int')) # categorical
    .withColumn('CIG_CEILING_DETERMINATION_CODE', CIG_col.getItem(2).cast('string')) # categorical 
    .withColumn('CIG_CAVOK_CODE', CIG_col.getItem(3).cast('string')) # categorical/binary
    # VIS Fields
    .withColumn('VIS_DISTANCE_DIMENSION', VIS_col.getItem(0).cast('int')) # continuous
    .withColumn('VIS_DISTANCE_QUALITY_CODE', VIS_col.getItem(1).cast('int')) # categorical
    .withColumn('VIS_VARIABILITY_CODE', VIS_col.getItem(2).cast('string')) # categorical/binary
    .withColumn('VIS_QUALITY_VARIABILITY_CODE', VIS_col.getItem(3).cast('int')) # categorical
    # TMP Fields
    .withColumn('TMP_AIR_TEMP', TMP_col.getItem(0).cast('int')) # continuous
    .withColumn('TMP_AIR_TEMP_QUALITY_CODE', TMP_col.getItem(1).cast('string')) # categorical
    # DEW Fields
    .withColumn('DEW_POINT_TEMP', DEW_col.getItem(0).cast('int')) # continuous
    .withColumn('DEW_POINT_QUALITY_CODE', DEW_col.getItem(1).cast('string')) # categorical
    # SLP Fields
    .withColumn('SLP_SEA_LEVEL_PRES', SLP_col.getItem(0).cast('int')) # continuous
    .withColumn('SLP_SEA_LEVEL_PRES_QUALITY_CODE', SLP_col.getItem(1).cast('int')) # categorical
    # SNOW Fields
    
       )

  return df

# COMMAND ----------

decoded_weather_sample = decode_weather_fields(weather_sample)

# COMMAND ----------

display(decoded_weather_sample)

# COMMAND ----------

# MAGIC %md **Now we will plot our data to get a visual representation of these flattened points**

# COMMAND ----------

# create histogram of wind speed. Filtered to remove nulls and values higher than the world record of 253 mph (113 m/s)
bins, counts = decoded_weather_sample.where('WND_Speed_Rate <= 113').select('WND_Speed_Rate').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# create histogram of VIS distance code
bins, counts = decoded_weather_sample.where('VIS_Distance_Dimension < 999999').select('VIS_Distance_Dimension').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# create histogram of SLP level code
bins, counts = decoded_weather_sample.where('SLP_Sea_Level_Pres < 99999').select('SLP_Sea_Level_Pres').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# create histogram of DEW field code
bins, counts = decoded_weather_sample.where('DEW_Point_Temp < 9999').select('DEW_Point_Temp').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# create histogram of Air Temp code
bins, counts = decoded_weather_sample.where('TMP_Air_Temp < 9999').select('TMP_Air_Temp').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md # Joining Weather to Station Data
# MAGIC The most efficient way to do this will be to first identify the station associated with each airport, add a column for that to the flights data, and then join directly flights to weather. Note this will require a composite key because we care about both **time** and **location**.  
# MAGIC   
# MAGIC ### Composite Key Strategy
# MAGIC Our goal for joining the weather data to the flights is to capture the most accurate weather at inference time (two hours prior to a flight) to enable flight delay prediction as a function of weather. We plan to create a composite key that is comprised of the ID of the closest weather station to each airport, the date of the flight, and the hour of the flight minus two. Example: *01002099999-2019010105*. The number prior to the hyphen is the weather station id. After the hyphen we have  a datetime in the format of YYYYMMDDHH.  
# MAGIC   
# MAGIC **Note on real-world validity:** After performing our joins and modeling, we had the realization that this technique does not take into account the possibility of a station only having weather readings for a given hour that occur AFTER our desired inference time. If our data engineering and model were deployed in production we would be unable to make predictions for some number of flights because there would undoubtly be cases where no weather readings have been collected yet. We do not fall back on data from the previous hour.  
# MAGIC   
# MAGIC **Note on practical validity:** Because we discovered this issue at the 11th hour, we opted not to update our data pipeline or model. We believe our model still holds practical validity under the assumption that the weather conditions associated with our error are unlikely to be significantly different those that would exist had we accounted for our edge case. The maximum forward-error using our technique is 59 minutes, 59 seconds if inference time occurs exactly on the hour and the weather reading for that hour occurs 59 minutes and 59 seconds after the hour. If more time were available to us to work on this project we would have either corrected the issue and performed a statistical analysis of the error introduced by our technique.

# COMMAND ----------

# MAGIC %md Before we join the weather and station data, it is important to make sure each airport in our flight data is represented in our airports file.

# COMMAND ----------

flights_processed = spark.sql("select * from flights_processed")
weather_processed = spark.sql("select * from weather_processed")
stations_processed = spark.sql("select * from stations_processed")
airports_processed = spark.sql("select * from airports_processed")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(ORIGIN)
# MAGIC FROM flights_processed
# MAGIC WHERE ORIGIN NOT IN (SELECT IATA FROM airports_processed);

# COMMAND ----------

# MAGIC %md These airports were missing from the airport file. Let's add them to our silver repository. 
# MAGIC 
# MAGIC Tokeen, Kearney, Bullhead City and Williston

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO airports_processed SELECT t.* FROM (SELECT 9999,"Tokeen Seaplane Base", "Tokeen", "United States", "TKI", "57A", "55.937222", "-133.326667", 0, -8, "A", "airport", "Internet", "America/Metlakatla" ) t;
# MAGIC INSERT INTO airports_processed SELECT t.* FROM (SELECT 9998,"Kearney Regional Airport", "Kearney", "United States", "EAR", "KEAR", "40.7270012", "-99.0067978", 2133, -5, "A", "airport", "Internet", "America/Chicago" ) t;
# MAGIC INSERT INTO airports_processed SELECT t.* FROM (SELECT 9997,"Laughlin Bullhead International Airport", "Bullhead City", "United States", "IFP", "KIFP", "35.1573982", "-114.5599976", 695, -7, "A", "airport", "Internet", "America/Phoenix" ) t;
# MAGIC INSERT INTO airports_processed SELECT t.* FROM (SELECT 9996,"Williston Basin International Airport", "Williston", "United States", "XWA", "KXWA", "48.1778984", "-103.6419983", 1982, -5, "A", "airport", "Internet", "America/Chicago" ) t;

# COMMAND ----------

# MAGIC %md Now that we have added the missing airports, let's check out the schema.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM airports_processed
# MAGIC LIMIT 1;

# COMMAND ----------

#re-retrieve airports_processed dataframe
airports_processed = spark.sql("select * from airports_processed")

# COMMAND ----------

# MAGIC %md ### Get Distinct Stations From Weather Data
# MAGIC This ensures that we only use stations that are valid for our analysis period.

# COMMAND ----------

#create set of distinct ids from weather data
weather_distinct_ids = weather_processed.select('WEATHER_STATION').distinct()

#join distinct ids to stations tables and subset for matches
valid_stations = weather_distinct_ids.join(stations_processed,\
                                           weather_distinct_ids.WEATHER_STATION == stations_processed.STATION_USAF_WBAN,\
                                           'left').where('STATION_USAF_WBAN IS NOT NULL')


# COMMAND ----------

# MAGIC %md * Query the flights to get the minimum and maximum date. 
# MAGIC   * Per next two cells, flight data covers 1/1/2015-12/31/2019
# MAGIC * Query the stations to look at the start and end date to verify they are all active for the reporting period we are concerned with.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT FL_DATE, COUNT(FL_DATE) 
# MAGIC FROM flights_processed
# MAGIC GROUP BY FL_DATE
# MAGIC ORDER BY FL_DATE DESC
# MAGIC LIMIT 3;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT FL_DATE, COUNT(FL_DATE) 
# MAGIC FROM flights_processed
# MAGIC GROUP BY FL_DATE
# MAGIC ORDER BY FL_DATE ASC
# MAGIC LIMIT 3;

# COMMAND ----------

# get max station end date
display(valid_stations.select('STATION_END').sort(f.col("STATION_END").desc()).limit(3))

# COMMAND ----------

# MAGIC %md The stations table itself only has stations that show as active through March 2019, but is this reflected in the weather data?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT WEATHER_DATE
# MAGIC FROM weather_processed
# MAGIC ORDER BY WEATHER_DATE DESC
# MAGIC LIMIT 1;

# COMMAND ----------

# MAGIC %md Weather data are collected through the end of 2019, so the ending date in the stations table is not reliable to use. Let's take a look at our valid stations.

# COMMAND ----------

display(valid_stations)

# COMMAND ----------

# MAGIC %md ## Find Nearest Station to Each Airport
# MAGIC Now that we have only the stations that are valid for our analysis, we can find the nearest one to each airport.

# COMMAND ----------

# MAGIC %md Define a function for the Haversine distance. This is not perfect because it does not consider the projection used when determining the latitude and longitude in the stations and airport data. However, that information is unknown, so the Haversine distance should be a reasonable approximation.

# COMMAND ----------

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# COMMAND ----------

# MAGIC %md Spark job for finding the closest stations. There is probably a better way to do this without converting stuff to RDDs, but this will work.

# COMMAND ----------

def find_closest_station(airports,stations):
    '''
    airports: rdd
    stations: rdd
    '''
    
    stations = sc.broadcast(stations.collect())
    
    def calc_distances(airport):
        airport_lon = float(airport['AIRPORT_LONG'])
        airport_lat = float(airport['AIRPORT_LAT'])

        for station in stations.value:
            if not station['STATION_LONG'] or not station['STATION_LAT']:
                continue
            station_lon = float(station['STATION_LONG'])
            station_lat = float(station['STATION_LAT'])
            station_id = station['STATION_USAF_WBAN']
            yield (airport['IATA'], (station_id, haversine(airport_lon, airport_lat, station_lon, station_lat)))
  
  
    def take_min(x,y):
      '''
      x and y are tuples of (airport, (station.id, distance))
      returns (airport, (argmin(station.id), min(distance)))
      '''
      minimum_index = np.argmin([x[1], y[1]])
      if minimum_index == 0:
          return x
      else:
          return y
      
    
    
    #output = airports.mapPartitions(calc_distances)
    output = airports.flatMap(calc_distances)
    #print(set(output.keys().collect()))           
    output = output.reduceByKey(lambda x, y: take_min(x,y))
    
    return output
  

# COMMAND ----------

# build aiport and station rdds
airports_rdd = airports_processed.rdd
stations_rdd = valid_stations.rdd

# COMMAND ----------

airports_rdd.take(1)[0]

# COMMAND ----------

stations_rdd.take(1)[0]

# COMMAND ----------

closest_stations = find_closest_station(airports_rdd,stations_rdd).cache()

# COMMAND ----------

airport_stations_loc = f"/airline_delays/{username}/DLRS/airport_stations/"

dbutils.fs.rm(airport_stations_loc + 'processed', recurse=True)


# COMMAND ----------

airports_stations = sqlContext.createDataFrame(closest_stations)

# COMMAND ----------

airports_stations.write.option('mergeSchema', True).mode('overwrite').format('delta').save(airport_stations_loc + 'processed')


# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS airport_stations_processed;
# MAGIC 
# MAGIC CREATE TABLE airport_stations_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/airport_stations/processed"

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from airport_stations_processed

# COMMAND ----------

# MAGIC %md ### Need to look at Honolulu

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM stations_processed
# MAGIC WHERE STATION_STATE = "HI"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT WEATHER_STATION, COUNT(*) AS NUM_RECORDS
# MAGIC FROM weather_processed
# MAGIC WHERE WEATHER_STATION IN (SELECT STATION_USAF_WBAN FROM stations_processed WHERE STATION_STATE = "HI")
# MAGIC GROUP BY WEATHER_STATION
# MAGIC ORDER BY NUM_RECORDS DESC

# COMMAND ----------

airports_stations = airports_stations.withColumn("NEAREST_STATION_ID",f.col("_2")["_1"]).withColumn("NEAREST_STATION_DIST",f.col("_2")["_2"])
airports_stations =airports_stations.drop("_2")
airports_stations_origin = airports_stations.withColumnRenamed("_1", "IATA")
airports_stations_dest = airports_stations_origin

airports_stations_origin = airports_stations_origin.withColumnRenamed("IATA", "IATA_ORIGIN")
airports_stations_origin = airports_stations_origin.withColumnRenamed("NEAREST_STATION_ID", "NEAREST_STATION_ID_ORIGIN")
airports_stations_origin = airports_stations_origin.withColumnRenamed("NEAREST_STATION_DIST", "NEAREST_STATION_DIST_ORIGIN")

airports_stations_dest = airports_stations_dest.withColumnRenamed("IATA", "IATA_DEST")
airports_stations_dest = airports_stations_dest.withColumnRenamed("NEAREST_STATION_ID", "NEAREST_STATION_ID_DEST")
airports_stations_dest = airports_stations_dest.withColumnRenamed("NEAREST_STATION_DIST", "NEAREST_STATION_DIST_DEST")

# COMMAND ----------

display(airports_stations_origin)

# COMMAND ----------

display(airports_stations_dest)

# COMMAND ----------

# MAGIC %md ## Join Nearest Stations to Flights

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM flights_processed
# MAGIC LIMIT 1;

# COMMAND ----------

#flights_processed = spark.read.format('delta').load(f'{flights_loc}processed')
flights_processed = spark.sql("select * from flights_processed")

# COMMAND ----------

joined_flights_stations = flights_processed.join(f.broadcast(airports_stations_origin), flights_processed.ORIGIN == airports_stations_origin.IATA_ORIGIN, 'left')
joined_flights_stations = joined_flights_stations.join(f.broadcast(airports_stations_dest), joined_flights_stations.DEST == airports_stations_dest.IATA_DEST, 'left')

# COMMAND ----------

# MAGIC %md Now that we have our nearest weather stations for each flight, we need to create composite keys based on the time of the flight. That will require flight time adjustment to UTC because flight times are in the local time zone. To do that, we join a subset of the airport table to our joined flights and stations table.

# COMMAND ----------

#subset airports
airports_tz = airports_processed.select(['IATA', 'AIRPORT_TZ_NAME'])

#join flights with stations to airport_tz subset on the origin airport because only the departure time needs the UTC adjustment
joined_flights_stations_tz = joined_flights_stations.join(airports_tz, joined_flights_stations.ORIGIN == airports_tz.IATA, 'left')

# COMMAND ----------

# MAGIC %md Before continuing, we will store this data into our Silver Delta Lake.

# COMMAND ----------

joined_flights_stations_tz.count()

# COMMAND ----------

joined_flights_stations_tz.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{flights_loc}processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM flights_processed
# MAGIC LIMIT 1;

# COMMAND ----------

# MAGIC %md ### Create Composite Keys in Flight Data

# COMMAND ----------

# Ensure we are always using the most recent version of the data in the delta lake
#flights_processed = spark.read.format('delta').load(f'{flights_loc}processed')
flights_processed = spark.sql("select * from flights_processed")

# COMMAND ----------

flights_processed.count()

# COMMAND ----------

# MAGIC %md Define UDFs for creating composite keys in the flights data

# COMMAND ----------

def get_utc_datetime(year, month, day, hour, tz):
  tz = timezone(tz)
  utc = timezone('UTC')
  
  local_dt = datetime(int(year), int(month), int(day), hour=int(hour), tzinfo=tz)
  utc_dt = local_dt.astimezone(utc)
  
  return utc_dt

get_utc_datetime = udf(get_utc_datetime)

def get_flight_hour(flight_time):
    flight_time = str(flight_time)
    hour = ''
    if len(flight_time) in [1,2]:
        hour = '00'
    elif len(flight_time) == 3:
      hour = flight_time[0]
    elif len(flight_time) == 4:
      hour = flight_time[:2]
    return hour
  
get_flight_hour = udf(get_flight_hour)


def get_two_hour_adjusted_datetime(current_datetime):
  return (current_datetime - timedelta(hours=2))

get_two_hour_adjusted_datetime = udf(get_two_hour_adjusted_datetime)

def get_datetime_string(d):
  return d.strftime("%Y%m%d%H")

get_datetime_string = udf(get_datetime_string)

# COMMAND ----------

# MAGIC %md Create composite keys on flights data

# COMMAND ----------

# extract the hour from the departure time
flights_processed = flights_processed.withColumn("CRS_DEP_TIME_HOUR", get_flight_hour("CRS_DEP_TIME"))

# COMMAND ----------

#convert the flight time to UTC
flights_processed = flights_processed.withColumn("FLIGHT_TIME_UTC", get_utc_datetime("YEAR",\
                                                                                     "MONTH",\
                                                                                     "DAY_OF_MONTH",\
                                                                                      get_flight_hour("CRS_DEP_TIME"),\
                                                                                      "AIRPORT_TZ_NAME"))

# COMMAND ----------

# Get the weather prediction time T-2hours from departure
flights_processed = flights_processed.withColumn("WEATHER_PREDICTION_TIME_UTC",
                                                                             get_two_hour_adjusted_datetime("FLIGHT_TIME_UTC"))

# COMMAND ----------

# Convert the datetime objects to strings
flights_processed = flights_processed.withColumn("FLIGHT_TIME_UTC",
                                                 get_datetime_string("FLIGHT_TIME_UTC"))\
                                                 .withColumn("WEATHER_PREDICTION_TIME_UTC",
                                                 get_datetime_string("WEATHER_PREDICTION_TIME_UTC"))

# COMMAND ----------

#Finally, generate the composite keys for each station
flights_processed = flights_processed.withColumn("ORIGIN_WEATHER_KEY",
                                                 f.concat_ws("-", "nearest_station_id_ORIGIN", "WEATHER_PREDICTION_TIME_UTC"))\
                                     .withColumn("DEST_WEATHER_KEY",
                                                 f.concat_ws("-", "nearest_station_id_DEST", "WEATHER_PREDICTION_TIME_UTC"))

# COMMAND ----------

display(flights_processed) 

# COMMAND ----------

flights_processed.where('ORIGIN_WEATHER_KEY IS NULL OR DEST_WEATHER_KEY IS NULL').count()

# COMMAND ----------

# MAGIC %md Now we have composite keys to join the weather to these data. But first, lets land this in the delta lake

# COMMAND ----------

flights_processed.count()

# COMMAND ----------

flights_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{flights_loc}processed')

# COMMAND ----------

# Ensure we are always using the most recent version of the data in the delta lake
#flights_processed = spark.read.format('delta').load(f'{flights_loc}processed')
flights_processed = spark.sql("select * from flights_processed")

# COMMAND ----------

# MAGIC %md Honolulu is missing a lot of weather data so we will replace its weather key with that of Hilo
# MAGIC Station ID 99999921515

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM flights_processed
# MAGIC WHERE ORIGIN = "HNL"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM flights_processed
# MAGIC WHERE DEST = "HNL"

# COMMAND ----------

#create udfs for fixing the HNL nearest stations and weather keys
def replace_HNL_weather_station_id(id):
    return "99999921515"
def fix_HNL_weather_key(weather_key):
    station, time = weather_key.split('-')
    new_weather_key = f'99999921515-{time}'
    return new_weather_key
  
# register UDFs
spark.udf.register("replace_HNL_weather_station_id", replace_HNL_weather_station_id)
spark.udf.register("fix_HNL_weather_key", fix_HNL_weather_key)



# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE flights_processed
# MAGIC SET nearest_station_id_ORIGIN = replace_HNL_weather_station_id(nearest_station_id_ORIGIN), ORIGIN_WEATHER_KEY = fix_HNL_weather_key(ORIGIN_WEATHER_KEY)
# MAGIC WHERE ORIGIN = "HNL"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM flights_processed
# MAGIC WHERE ORIGIN = "HNL"

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE flights_processed
# MAGIC SET nearest_station_id_DEST = replace_HNL_weather_station_id(nearest_station_id_DEST), DEST_WEATHER_KEY = fix_HNL_weather_key(DEST_WEATHER_KEY)
# MAGIC WHERE DEST = "HNL"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM flights_processed
# MAGIC WHERE DEST = "HNL"

# COMMAND ----------

# Ensure we are always using the most recent version of the data in the delta lake
#flights_processed = spark.read.format('delta').load(f'{flights_loc}processed')
flights_processed = spark.sql("select * from flights_processed")

# COMMAND ----------

# MAGIC %md ### Create Composite Keys in Weather Data

# COMMAND ----------

# Ensure we are always using the most recent version of the data in the delta lake
#weather_processed = spark.read.format('delta').load(f'{weather_loc}processed')
weather_processed = spark.sql("select * from weather_processed")

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT COUNT(*)
# MAGIC FROM weather_processed
# MAGIC WHERE WEATHER_STATION IS NULL

# COMMAND ----------

# MAGIC %md Remove weather data with no station ID and store it back onto the delta lake (we won't use these records and saving it back now will reduce our time for future queries)

# COMMAND ----------

weather_processed.where("WEATHER_STATION IS NOT NULL").write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{weather_loc}processed')

# COMMAND ----------

# Ensure we are always using the most recent version of the data in the delta lake
#weather_processed = spark.read.format('delta').load(f'{weather_loc}processed')
weather_processed = spark.sql("select * from weather_processed")

# COMMAND ----------

# MAGIC %md Now we will define a UDF to create our composite weather key

# COMMAND ----------

weather_processed.printSchema()

# COMMAND ----------

def create_composite_weather_key(d, k):
    datestr = d.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    date, time = datestr.split("T")
    date_components = date.split("-")
    id_date_portion = "".join(date_components)
    id_hour_portion = time[:2]
    id_datetime_portion = id_date_portion+id_hour_portion
    the_id = "-".join([k,id_datetime_portion])
    return the_id
#create_composite_weather_key = udf(create_composite_weather_key)
spark.udf.register("create_composite_weather_key", create_composite_weather_key)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT create_composite_weather_key(WEATHER_DATE, WEATHER_STATION)
# MAGIC FROM weather_processed;

# COMMAND ----------

weather_processed = spark.sql("SELECT *, create_composite_weather_key(WEATHER_DATE, WEATHER_STATION) AS WEATHER_KEY FROM weather_processed;")

# COMMAND ----------

# MAGIC %md
# MAGIC before

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM weather_processed
# MAGIC LIMIT 1

# COMMAND ----------

# MAGIC %md
# MAGIC after

# COMMAND ----------

display(weather_processed.limit(1))

# COMMAND ----------

weather_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{weather_loc}processed')

# COMMAND ----------

# MAGIC %md
# MAGIC confirming

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM weather_processed
# MAGIC LIMIT 1

# COMMAND ----------

# MAGIC %md ## Join Weather to Flights

# COMMAND ----------

# MAGIC %md Weather data has multiple readings within the hour. We want just the first reading. So we'll apply a window join on a sub-select to retrieve it (as illustrated below).

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC      FROM flights_processed fp
# MAGIC      JOIN ( 
# MAGIC             SELECT * FROM (
# MAGIC                   SELECT *, ROW_NUMBER() OVER ( 
# MAGIC                         partition by WEATHER_KEY 
# MAGIC                         ORDER BY WEATHER_DATE ASC 
# MAGIC                   ) as w1_row_num 
# MAGIC                   FROM weather_processed 
# MAGIC               ) as ordered_weather 
# MAGIC               WHERE ordered_weather.w1_row_num = 1 
# MAGIC           ) as w1 
# MAGIC           ON fp.ORIGIN_WEATHER_KEY = w1.WEATHER_KEY
# MAGIC LIMIT 10

# COMMAND ----------

flights_and_weather_origin_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_origin/"
flights_and_weather_origin_ren_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_origin_renamed/"
flights_and_weather_dest_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_dest/"
flights_and_weather_combined_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_combined/"


dbutils.fs.rm(flights_and_weather_origin_loc + "processed", recurse=True)
dbutils.fs.rm(flights_and_weather_origin_ren_loc + "processed", recurse=True)
dbutils.fs.rm(flights_and_weather_dest_loc + "processed", recurse=True)
dbutils.fs.rm(flights_and_weather_combined_loc + "processed", recurse=True)

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE flights_and_weather_origin 
# MAGIC USING DELTA 
# MAGIC LOCATION '/airline_delays/$username/DLRS/flights_and_weather_origin/processed' 
# MAGIC AS SELECT * FROM flights_processed fp LEFT
# MAGIC JOIN (SELECT * 
# MAGIC       FROM (SELECT *, ROW_NUMBER() 
# MAGIC             OVER (PARTITION BY wp.WEATHER_KEY
# MAGIC                   ORDER BY wp.WEATHER_DATE ASC) AS ORIGIN_ROW_NUM
# MAGIC             FROM weather_processed wp) AS w
# MAGIC       WHERE w.ORIGIN_ROW_NUM = 1) AS w1 
# MAGIC ON fp.ORIGIN_WEATHER_KEY = w1.WEATHER_KEY

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from flights_and_weather_origin

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE flights_and_weather_origin_renamed USING DELTA LOCATION '/airline_delays/$username/DLRS/flights_and_weather_origin_renamed/processed' AS
# MAGIC SELECT fp.YEAR,fp.QUARTER,fp.MONTH,fp.DAY_OF_MONTH,fp.DAY_OF_WEEK, fp.OP_UNIQUE_CARRIER,
# MAGIC        fp.OP_CARRIER_AIRLINE_ID,fp.OP_CARRIER,fp.TAIL_NUM,fp.OP_CARRIER_FL_NUM,
# MAGIC        fp.ORIGIN_AIRPORT_ID,fp.ORIGIN_AIRPORT_SEQ_ID,fp.ORIGIN_CITY_MARKET_ID,
# MAGIC        fp.ORIGIN,fp.ORIGIN_CITY_NAME,fp.ORIGIN_STATE_ABR,fp.ORIGIN_STATE_FIPS,
# MAGIC        fp.ORIGIN_STATE_NM,fp.ORIGIN_WAC,fp.DEST_AIRPORT_ID, fp.DEST_AIRPORT_SEQ_ID,
# MAGIC        fp.DEST_CITY_MARKET_ID,fp.DEST,fp.DEST_CITY_NAME,fp.DEST_STATE_ABR, fp.DEST_STATE_FIPS,
# MAGIC        fp.DEST_STATE_NM,fp.DEST_WAC,fp.CRS_DEP_TIME,fp.DEP_TIME,fp.DEP_DELAY, fp.DEP_DELAY_NEW,
# MAGIC        fp.DEP_DEL15,fp.DEP_DELAY_GROUP,fp.DEP_TIME_BLK,fp.TAXI_OUT,
# MAGIC        fp.WHEELS_OFF,fp.WHEELS_ON,fp.TAXI_IN,fp.CRS_ARR_TIME,fp.ARR_TIME, fp.ARR_DELAY,
# MAGIC        fp.ARR_DELAY_NEW,fp.ARR_DEL15,fp.ARR_DELAY_GROUP,fp.ARR_TIME_BLK, fp.CANCELLED,
# MAGIC        fp.CANCELLATION_CODE,fp.DIVERTED,fp.CRS_ELAPSED_TIME, fp.ACTUAL_ELAPSED_TIME,fp.AIR_TIME,
# MAGIC        fp.FLIGHTS,fp.DISTANCE,fp.DISTANCE_GROUP,fp.CARRIER_DELAY, fp.WEATHER_DELAY,
# MAGIC        fp.NAS_DELAY,fp.SECURITY_DELAY,fp.LATE_AIRCRAFT_DELAY,fp.FIRST_DEP_TIME,
# MAGIC        fp.TOTAL_ADD_GTIME,fp.LONGEST_ADD_GTIME,fp.DIV_AIRPORT_LANDINGS, fp.DIV_REACHED_DEST,
# MAGIC        fp.DIV_ACTUAL_ELAPSED_TIME,fp.DIV_ARR_DELAY,fp.DIV_DISTANCE, fp.DIV1_AIRPORT,fp.DIV1_AIRPORT_ID,
# MAGIC        fp.DIV1_AIRPORT_SEQ_ID,fp.DIV1_WHEELS_ON,fp.DIV1_TOTAL_GTIME, fp.DIV1_LONGEST_GTIME,
# MAGIC        fp.DIV1_WHEELS_OFF,fp.DIV1_TAIL_NUM,fp.DIV2_AIRPORT,fp.DIV2_AIRPORT_ID,
# MAGIC        fp.DIV2_AIRPORT_SEQ_ID,fp.DIV2_WHEELS_ON,fp.DIV2_TOTAL_GTIME, fp.DIV2_LONGEST_GTIME,
# MAGIC        fp.DIV2_WHEELS_OFF,fp.DIV2_TAIL_NUM,fp.DIV3_AIRPORT,fp.DIV3_AIRPORT_ID,
# MAGIC        fp.DIV3_AIRPORT_SEQ_ID,fp.DIV3_WHEELS_ON,fp.DIV3_TOTAL_GTIME, fp.DIV3_LONGEST_GTIME,
# MAGIC        fp.DIV3_WHEELS_OFF,fp.DIV3_TAIL_NUM,fp.DIV4_AIRPORT,fp.DIV4_AIRPORT_ID,
# MAGIC        fp.DIV4_AIRPORT_SEQ_ID,fp.DIV4_WHEELS_ON,fp.DIV4_TOTAL_GTIME, fp.DIV4_LONGEST_GTIME,
# MAGIC        fp.DIV4_WHEELS_OFF,fp.DIV4_TAIL_NUM,fp.DIV5_AIRPORT,fp.DIV5_AIRPORT_ID,
# MAGIC        fp.DIV5_AIRPORT_SEQ_ID,fp.DIV5_WHEELS_ON,fp.DIV5_TOTAL_GTIME, fp.DIV5_LONGEST_GTIME,
# MAGIC        fp.DIV5_WHEELS_OFF,fp.DIV5_TAIL_NUM,fp.IN_FLIGHT_AIR_DELAY,fp.FL_DATE, fp.IATA_ORIGIN,
# MAGIC        fp.NEAREST_STATION_ID_ORIGIN,fp.NEAREST_STATION_DIST_ORIGIN,fp.IATA_DEST,
# MAGIC        fp.NEAREST_STATION_ID_DEST,fp.NEAREST_STATION_DIST_DEST,fp.IATA, fp.AIRPORT_TZ_NAME,
# MAGIC        fp.CRS_DEP_TIME_HOUR,fp.FLIGHT_TIME_UTC,fp.WEATHER_PREDICTION_TIME_UTC,
# MAGIC        fp.ORIGIN_WEATHER_KEY,fp.WEATHER_STATION as ORIGIN_WEATHER_STATION,
# MAGIC        fp.WEATHER_SOURCE as ORIGIN_WEATHER_SOURCE,
# MAGIC        fp.WEATHER_LAT as ORIGIN_WEATHER_LAT,fp.WEATHER_LON as ORIGIN_WEATHER_LON,fp.WEATHER_ELEV as ORIGIN_WEATHER_ELEV,
# MAGIC        fp.WEATHER_NAME as ORIGIN_WEATHER_NAME, fp.WEATHER_REPORT_TYPE as ORIGIN_WEATHER_REPORT_TYPE,
# MAGIC        fp.WEATHER_CALL_SIGN as ORIGIN_WEATHER_CALL_SIGN, fp.QUALITY_CONTROL as ORIGIN_QUALITY_CONTROL,fp.WND as ORIGIN_WND,
# MAGIC        fp.CIG as ORIGIN_CIG,fp.VIS as ORIGIN_VIS,fp.TMP as ORIGIN_TMP,
# MAGIC        fp.DEW as ORIGIN_DEW,fp.SLP as ORIGIN_SLP,fp.AW1 as ORIGIN_AW1,
# MAGIC        fp.GA1 as ORIGIN_GA1,fp.GA2 as ORIGIN_GA2,fp.GA3 as ORIGIN_GA3,
# MAGIC        fp.GA4 as ORIGIN_GA4,fp.GE1 as ORIGIN_GE1,fp.GF1 as ORIGIN_GF1,
# MAGIC        fp.KA1 as ORIGIN_KA1,fp.KA2 as ORIGIN_KA2,fp.MA1 as ORIGIN_MA1,
# MAGIC        fp.MD1 as ORIGIN_MD1,fp.MW1 as ORIGIN_MW1,fp.MW2 as ORIGIN_MW2,
# MAGIC        fp.OC1 as ORIGIN_OC1,fp.OD1 as ORIGIN_OD1,fp.OD2 as ORIGIN_OD2,
# MAGIC        fp.REM as ORIGIN_REM,fp.EQD as ORIGIN_EQD,fp.AW2 as ORIGIN_AW2,
# MAGIC        fp.AX4 as ORIGIN_AX4,fp.GD1 as ORIGIN_GD1,fp.AW5 as ORIGIN_AW5,
# MAGIC        fp.GN1 as ORIGIN_GN1,fp.AJ1 as ORIGIN_AJ1,fp.AW3 as ORIGIN_AW3,
# MAGIC        fp.MK1 as ORIGIN_MK1,fp.KA4 as ORIGIN_KA4,fp.GG3 as ORIGIN_GG3,
# MAGIC        fp.AN1 as ORIGIN_AN1,fp.RH1 as ORIGIN_RH1,fp.AU5 as ORIGIN_AU5,
# MAGIC        fp.HL1 as ORIGIN_HL1,fp.OB1 as ORIGIN_OB1,fp.AT8 as ORIGIN_AT8,
# MAGIC        fp.AW7 as ORIGIN_AW7,fp.AZ1 as ORIGIN_AZ1,fp.CH1 as ORIGIN_CH1,
# MAGIC        fp.RH3 as ORIGIN_RH3,fp.GK1 as ORIGIN_GK1,fp.IB1 as ORIGIN_IB1,
# MAGIC        fp.AX1 as ORIGIN_AX1,fp.CT1 as ORIGIN_CT1,fp.AK1 as ORIGIN_AK1,
# MAGIC        fp.CN2 as ORIGIN_CN2,fp.OE1 as ORIGIN_OE1,fp.MW5 as ORIGIN_MW5,
# MAGIC        fp.AO1 as ORIGIN_AO1,fp.KA3 as ORIGIN_KA3,fp.AA3 as ORIGIN_AA3,
# MAGIC        fp.CR1 as ORIGIN_CR1,fp.CF2 as ORIGIN_CF2,fp.KB2 as ORIGIN_KB2,
# MAGIC        fp.GM1 as ORIGIN_GM1,fp.AT5 as ORIGIN_AT5,fp.AY2 as ORIGIN_AY2,
# MAGIC        fp.MW6 as ORIGIN_MW6,fp.MG1 as ORIGIN_MG1,fp.AH6 as ORIGIN_AH6,
# MAGIC        fp.AU2 as ORIGIN_AU2,fp.GD2 as ORIGIN_GD2,fp.AW4 as ORIGIN_AW4,
# MAGIC        fp.MF1 as ORIGIN_MF1,fp.AA1 as ORIGIN_AA1,fp.AH2 as ORIGIN_AH2,
# MAGIC        fp.AH3 as ORIGIN_AH3,fp.OE3 as ORIGIN_OE3,fp.AT6 as ORIGIN_AT6,
# MAGIC        fp.AL2 as ORIGIN_AL2,fp.AL3 as ORIGIN_AL3,fp.AX5 as ORIGIN_AX5,
# MAGIC        fp.IB2 as ORIGIN_IB2,fp.AI3 as ORIGIN_AI3,fp.CV3 as ORIGIN_CV3,
# MAGIC        fp.WA1 as ORIGIN_WA1,fp.GH1 as ORIGIN_GH1,fp.KF1 as ORIGIN_KF1,
# MAGIC        fp.CU2 as ORIGIN_CU2,fp.CT3 as ORIGIN_CT3,fp.SA1 as ORIGIN_SA1,
# MAGIC        fp.AU1 as ORIGIN_AU1,fp.KD2 as ORIGIN_KD2,fp.AI5 as ORIGIN_AI5,
# MAGIC        fp.GO1 as ORIGIN_GO1,fp.GD3 as ORIGIN_GD3,fp.CG3 as ORIGIN_CG3,
# MAGIC        fp.AI1 as ORIGIN_AI1,fp.AL1 as ORIGIN_AL1,fp.AW6 as ORIGIN_AW6,
# MAGIC        fp.MW4 as ORIGIN_MW4,fp.AX6 as ORIGIN_AX6,fp.CV1 as ORIGIN_CV1,
# MAGIC        fp.ME1 as ORIGIN_ME1,fp.KC2 as ORIGIN_KC2,fp.CN1 as ORIGIN_CN1,
# MAGIC        fp.UA1 as ORIGIN_UA1,fp.GD5 as ORIGIN_GD5,fp.UG2 as ORIGIN_UG2,
# MAGIC        fp.AT3 as ORIGIN_AT3,fp.AT4 as ORIGIN_AT4,fp.GJ1 as ORIGIN_GJ1,
# MAGIC        fp.MV1 as ORIGIN_MV1,fp.GA5 as ORIGIN_GA5,fp.CT2 as ORIGIN_CT2,
# MAGIC        fp.CG2 as ORIGIN_CG2,fp.ED1 as ORIGIN_ED1,fp.AE1 as ORIGIN_AE1,
# MAGIC        fp.CO1 as ORIGIN_CO1,fp.KE1 as ORIGIN_KE1,fp.KB1 as ORIGIN_KB1,
# MAGIC        fp.AI4 as ORIGIN_AI4,fp.MW3 as ORIGIN_MW3,fp.KG2 as ORIGIN_KG2,
# MAGIC        fp.AA2 as ORIGIN_AA2,fp.AX2 as ORIGIN_AX2,fp.AY1 as ORIGIN_AY1,
# MAGIC        fp.RH2 as ORIGIN_RH2,fp.OE2 as ORIGIN_OE2,fp.CU3 as ORIGIN_CU3,
# MAGIC        fp.MH1 as ORIGIN_MH1,fp.AM1 as ORIGIN_AM1,fp.AU4 as ORIGIN_AU4,
# MAGIC        fp.GA6 as ORIGIN_GA6,fp.KG1 as ORIGIN_KG1,fp.AU3 as ORIGIN_AU3,
# MAGIC        fp.AT7 as ORIGIN_AT7,fp.KD1 as ORIGIN_KD1,fp.GL1 as ORIGIN_GL1,
# MAGIC        fp.IA1 as ORIGIN_IA1,fp.GG2 as ORIGIN_GG2,fp.OD3 as ORIGIN_OD3,
# MAGIC        fp.UG1 as ORIGIN_UG1,fp.CB1 as ORIGIN_CB1,fp.AI6 as ORIGIN_AI6,
# MAGIC        fp.CI1 as ORIGIN_CI1,fp.CV2 as ORIGIN_CV2,fp.AZ2 as ORIGIN_AZ2,
# MAGIC        fp.AD1 as ORIGIN_AD1,fp.AH1 as ORIGIN_AH1,fp.WD1 as ORIGIN_WD1,
# MAGIC        fp.AA4 as ORIGIN_AA4,fp.KC1 as ORIGIN_KC1,fp.IA2 as ORIGIN_IA2,
# MAGIC        fp.CF3 as ORIGIN_CF3,fp.AI2 as ORIGIN_AI2,fp.AT1 as ORIGIN_AT1,
# MAGIC        fp.GD4 as ORIGIN_GD4,fp.AX3 as ORIGIN_AX3,fp.AH4 as ORIGIN_AH4,
# MAGIC        fp.KB3 as ORIGIN_KB3,fp.CU1 as ORIGIN_CU1,fp.CN4 as ORIGIN_CN4,
# MAGIC        fp.AT2 as ORIGIN_AT2,fp.CG1 as ORIGIN_CG1,fp.CF1 as ORIGIN_CF1,
# MAGIC        fp.GG1 as ORIGIN_GG1,fp.MV2 as ORIGIN_MV2,fp.CW1 as ORIGIN_CW1,
# MAGIC        fp.GG4 as ORIGIN_GG4,fp.AB1 as ORIGIN_AB1,fp.AH5 as ORIGIN_AH5,
# MAGIC        fp.CN3 as ORIGIN_CN3,fp.WEATHER_DATE as ORIGIN_WEATHER_DATE,
# MAGIC        fp.DEST_WEATHER_KEY
# MAGIC FROM   flights_and_weather_origin fp 

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE flights_and_weather_dest USING DELTA LOCATION '/airline_delays/$username/DLRS/flights_and_weather_dest/processed' AS
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   flights_and_weather_origin_renamed fp LEFT
# MAGIC   JOIN (
# MAGIC     SELECT
# MAGIC       *
# MAGIC     FROM
# MAGIC       (
# MAGIC         SELECT
# MAGIC           *,
# MAGIC           ROW_NUMBER() OVER (
# MAGIC             partition by wp.WEATHER_KEY
# MAGIC             ORDER BY
# MAGIC               wp.WEATHER_DATE ASC
# MAGIC           ) as DEST_ROW_NUM
# MAGIC         FROM
# MAGIC           weather_processed wp
# MAGIC       ) as w
# MAGIC     WHERE
# MAGIC       w.DEST_ROW_NUM = 1
# MAGIC   ) as w1 ON fp.DEST_WEATHER_KEY = w1.WEATHER_KEY

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE flights_and_weather_combined USING DELTA PARTITIONED BY (fL_DATE) LOCATION '/airline_delays/$username/DLRS/flights_and_weather_combined/processed' AS
# MAGIC SELECT fp.YEAR,fp.QUARTER,fp.MONTH,fp.DAY_OF_MONTH,fp.DAY_OF_WEEK, fp.OP_UNIQUE_CARRIER, 
# MAGIC        fp.OP_CARRIER_AIRLINE_ID,fp.OP_CARRIER,fp.TAIL_NUM,fp.OP_CARRIER_FL_NUM, 
# MAGIC        fp.ORIGIN_AIRPORT_ID,fp.ORIGIN_AIRPORT_SEQ_ID,fp.ORIGIN_CITY_MARKET_ID, 
# MAGIC        fp.ORIGIN,fp.ORIGIN_CITY_NAME,fp.ORIGIN_STATE_ABR,fp.ORIGIN_STATE_FIPS, 
# MAGIC        fp.ORIGIN_STATE_NM,fp.ORIGIN_WAC,fp.DEST_AIRPORT_ID, fp.DEST_AIRPORT_SEQ_ID, 
# MAGIC        fp.DEST_CITY_MARKET_ID,fp.DEST,fp.DEST_CITY_NAME,fp.DEST_STATE_ABR, fp.DEST_STATE_FIPS, 
# MAGIC        fp.DEST_STATE_NM,fp.DEST_WAC,fp.CRS_DEP_TIME,fp.DEP_TIME,fp.DEP_DELAY, fp.DEP_DELAY_NEW, 
# MAGIC        fp.DEP_DEL15,fp.DEP_DELAY_GROUP,fp.DEP_TIME_BLK,fp.TAXI_OUT, 
# MAGIC        fp.WHEELS_OFF,fp.WHEELS_ON,fp.TAXI_IN,fp.CRS_ARR_TIME,fp.ARR_TIME, fp.ARR_DELAY, 
# MAGIC        fp.ARR_DELAY_NEW,fp.ARR_DEL15,fp.ARR_DELAY_GROUP,fp.ARR_TIME_BLK, fp.CANCELLED, 
# MAGIC        fp.CANCELLATION_CODE,fp.DIVERTED,fp.CRS_ELAPSED_TIME, fp.ACTUAL_ELAPSED_TIME,fp.AIR_TIME, 
# MAGIC        fp.FLIGHTS,fp.DISTANCE,fp.DISTANCE_GROUP,fp.CARRIER_DELAY, fp.WEATHER_DELAY, 
# MAGIC        fp.NAS_DELAY,fp.SECURITY_DELAY,fp.LATE_AIRCRAFT_DELAY,fp.FIRST_DEP_TIME, 
# MAGIC        fp.TOTAL_ADD_GTIME,fp.LONGEST_ADD_GTIME,fp.DIV_AIRPORT_LANDINGS, fp.DIV_REACHED_DEST, 
# MAGIC        fp.DIV_ACTUAL_ELAPSED_TIME,fp.DIV_ARR_DELAY,fp.DIV_DISTANCE, fp.DIV1_AIRPORT,fp.DIV1_AIRPORT_ID, 
# MAGIC        fp.DIV1_AIRPORT_SEQ_ID,fp.DIV1_WHEELS_ON,fp.DIV1_TOTAL_GTIME, fp.DIV1_LONGEST_GTIME, 
# MAGIC        fp.DIV1_WHEELS_OFF,fp.DIV1_TAIL_NUM,fp.DIV2_AIRPORT,fp.DIV2_AIRPORT_ID, 
# MAGIC        fp.DIV2_AIRPORT_SEQ_ID,fp.DIV2_WHEELS_ON,fp.DIV2_TOTAL_GTIME, fp.DIV2_LONGEST_GTIME, 
# MAGIC        fp.DIV2_WHEELS_OFF,fp.DIV2_TAIL_NUM,fp.DIV3_AIRPORT,fp.DIV3_AIRPORT_ID, 
# MAGIC        fp.DIV3_AIRPORT_SEQ_ID,fp.DIV3_WHEELS_ON,fp.DIV3_TOTAL_GTIME, fp.DIV3_LONGEST_GTIME, 
# MAGIC        fp.DIV3_WHEELS_OFF,fp.DIV3_TAIL_NUM,fp.DIV4_AIRPORT,fp.DIV4_AIRPORT_ID, 
# MAGIC        fp.DIV4_AIRPORT_SEQ_ID,fp.DIV4_WHEELS_ON,fp.DIV4_TOTAL_GTIME, fp.DIV4_LONGEST_GTIME, 
# MAGIC        fp.DIV4_WHEELS_OFF,fp.DIV4_TAIL_NUM,fp.DIV5_AIRPORT,fp.DIV5_AIRPORT_ID, 
# MAGIC        fp.DIV5_AIRPORT_SEQ_ID,fp.DIV5_WHEELS_ON,fp.DIV5_TOTAL_GTIME, fp.DIV5_LONGEST_GTIME, 
# MAGIC        fp.DIV5_WHEELS_OFF,fp.DIV5_TAIL_NUM,fp.IN_FLIGHT_AIR_DELAY,fp.FL_DATE, fp.IATA_ORIGIN, 
# MAGIC        fp.NEAREST_STATION_ID_ORIGIN,fp.NEAREST_STATION_DIST_ORIGIN,fp.IATA_DEST, 
# MAGIC        fp.NEAREST_STATION_ID_DEST,fp.NEAREST_STATION_DIST_DEST,fp.IATA, fp.AIRPORT_TZ_NAME, 
# MAGIC        fp.CRS_DEP_TIME_HOUR,fp.FLIGHT_TIME_UTC,fp.WEATHER_PREDICTION_TIME_UTC, 
# MAGIC        fp.ORIGIN_WEATHER_KEY,fp.ORIGIN_WEATHER_STATION, 
# MAGIC        fp.ORIGIN_WEATHER_SOURCE,fp.ORIGIN_WEATHER_LAT,fp.ORIGIN_WEATHER_LON, fp.ORIGIN_WEATHER_ELEV, 
# MAGIC        fp.ORIGIN_WEATHER_NAME,fp.ORIGIN_WEATHER_REPORT_TYPE, fp.ORIGIN_WEATHER_CALL_SIGN, 
# MAGIC        fp.ORIGIN_QUALITY_CONTROL,fp.ORIGIN_WND,fp.ORIGIN_CIG,fp.ORIGIN_VIS, fp.ORIGIN_TMP,fp.ORIGIN_DEW, 
# MAGIC        fp.ORIGIN_SLP,fp.ORIGIN_AW1,fp.ORIGIN_GA1,fp.ORIGIN_GA2,fp.ORIGIN_GA3, 
# MAGIC        fp.ORIGIN_GA4,fp.ORIGIN_GE1,fp.ORIGIN_GF1,fp.ORIGIN_KA1,fp.ORIGIN_KA2, fp.ORIGIN_MA1, 
# MAGIC        fp.ORIGIN_MD1,fp.ORIGIN_MW1,fp.ORIGIN_MW2,fp.ORIGIN_OC1,fp.ORIGIN_OD1, 
# MAGIC        fp.ORIGIN_OD2,fp.ORIGIN_REM,fp.ORIGIN_EQD,fp.ORIGIN_AW2,fp.ORIGIN_AX4, fp.ORIGIN_GD1, 
# MAGIC        fp.ORIGIN_AW5,fp.ORIGIN_GN1,fp.ORIGIN_AJ1,fp.ORIGIN_AW3,fp.ORIGIN_MK1, 
# MAGIC        fp.ORIGIN_KA4,fp.ORIGIN_GG3,fp.ORIGIN_AN1,fp.ORIGIN_RH1,fp.ORIGIN_AU5, fp.ORIGIN_HL1, 
# MAGIC        fp.ORIGIN_OB1,fp.ORIGIN_AT8,fp.ORIGIN_AW7,fp.ORIGIN_AZ1,fp.ORIGIN_CH1, 
# MAGIC        fp.ORIGIN_RH3,fp.ORIGIN_GK1,fp.ORIGIN_IB1,fp.ORIGIN_AX1,fp.ORIGIN_CT1, fp.ORIGIN_AK1, 
# MAGIC        fp.ORIGIN_CN2,fp.ORIGIN_OE1,fp.ORIGIN_MW5,fp.ORIGIN_AO1,fp.ORIGIN_KA3, 
# MAGIC        fp.ORIGIN_AA3,fp.ORIGIN_CR1,fp.ORIGIN_CF2,fp.ORIGIN_KB2,fp.ORIGIN_GM1, fp.ORIGIN_AT5, 
# MAGIC        fp.ORIGIN_AY2,fp.ORIGIN_MW6,fp.ORIGIN_MG1,fp.ORIGIN_AH6,fp.ORIGIN_AU2, 
# MAGIC        fp.ORIGIN_GD2,fp.ORIGIN_AW4,fp.ORIGIN_MF1,fp.ORIGIN_AA1,fp.ORIGIN_AH2, fp.ORIGIN_AH3, 
# MAGIC        fp.ORIGIN_OE3,fp.ORIGIN_AT6,fp.ORIGIN_AL2,fp.ORIGIN_AL3,fp.ORIGIN_AX5, 
# MAGIC        fp.ORIGIN_IB2,fp.ORIGIN_AI3,fp.ORIGIN_CV3,fp.ORIGIN_WA1,fp.ORIGIN_GH1, fp.ORIGIN_KF1, 
# MAGIC        fp.ORIGIN_CU2,fp.ORIGIN_CT3,fp.ORIGIN_SA1,fp.ORIGIN_AU1,fp.ORIGIN_KD2, 
# MAGIC        fp.ORIGIN_AI5,fp.ORIGIN_GO1,fp.ORIGIN_GD3,fp.ORIGIN_CG3,fp.ORIGIN_AI1, fp.ORIGIN_AL1, 
# MAGIC        fp.ORIGIN_AW6,fp.ORIGIN_MW4,fp.ORIGIN_AX6,fp.ORIGIN_CV1,fp.ORIGIN_ME1, 
# MAGIC        fp.ORIGIN_KC2,fp.ORIGIN_CN1,fp.ORIGIN_UA1,fp.ORIGIN_GD5,fp.ORIGIN_UG2, fp.ORIGIN_AT3, 
# MAGIC        fp.ORIGIN_AT4,fp.ORIGIN_GJ1,fp.ORIGIN_MV1,fp.ORIGIN_GA5,fp.ORIGIN_CT2, 
# MAGIC        fp.ORIGIN_CG2,fp.ORIGIN_ED1,fp.ORIGIN_AE1,fp.ORIGIN_CO1,fp.ORIGIN_KE1, fp.ORIGIN_KB1, 
# MAGIC        fp.ORIGIN_AI4,fp.ORIGIN_MW3,fp.ORIGIN_KG2,fp.ORIGIN_AA2,fp.ORIGIN_AX2, 
# MAGIC        fp.ORIGIN_AY1,fp.ORIGIN_RH2,fp.ORIGIN_OE2,fp.ORIGIN_CU3,fp.ORIGIN_MH1, fp.ORIGIN_AM1, 
# MAGIC        fp.ORIGIN_AU4,fp.ORIGIN_GA6,fp.ORIGIN_KG1,fp.ORIGIN_AU3,fp.ORIGIN_AT7, 
# MAGIC        fp.ORIGIN_KD1,fp.ORIGIN_GL1,fp.ORIGIN_IA1,fp.ORIGIN_GG2,fp.ORIGIN_OD3, fp.ORIGIN_UG1, 
# MAGIC        fp.ORIGIN_CB1,fp.ORIGIN_AI6,fp.ORIGIN_CI1,fp.ORIGIN_CV2,fp.ORIGIN_AZ2, 
# MAGIC        fp.ORIGIN_AD1,fp.ORIGIN_AH1,fp.ORIGIN_WD1,fp.ORIGIN_AA4,fp.ORIGIN_KC1, fp.ORIGIN_IA2, 
# MAGIC        fp.ORIGIN_CF3,fp.ORIGIN_AI2,fp.ORIGIN_AT1,fp.ORIGIN_GD4,fp.ORIGIN_AX3, 
# MAGIC        fp.ORIGIN_AH4,fp.ORIGIN_KB3,fp.ORIGIN_CU1,fp.ORIGIN_CN4,fp.ORIGIN_AT2, fp.ORIGIN_CG1, 
# MAGIC        fp.ORIGIN_CF1,fp.ORIGIN_GG1,fp.ORIGIN_MV2,fp.ORIGIN_CW1,fp.ORIGIN_GG4, 
# MAGIC        fp.ORIGIN_AB1,fp.ORIGIN_AH5,fp.ORIGIN_CN3,fp.ORIGIN_WEATHER_DATE, 
# MAGIC        fp.DEST_WEATHER_KEY,fp.WEATHER_STATION as DEST_WEATHER_STATION, 
# MAGIC        fp.WEATHER_SOURCE as DEST_WEATHER_SOURCE, fp.WEATHER_LAT as DEST_WEATHER_LAT,fp.WEATHER_LON as DEST_WEATHER_LON, 
# MAGIC        fp.WEATHER_ELEV as DEST_WEATHER_ELEV,fp.WEATHER_NAME as DEST_WEATHER_NAME, fp.WEATHER_REPORT_TYPE as DEST_WEATHER_REPORT_TYPE, 
# MAGIC        fp.WEATHER_CALL_SIGN as DEST_WEATHER_CALL_SIGN, fp.QUALITY_CONTROL as DEST_QUALITY_CONTROL,fp.WND as DEST_WND, 
# MAGIC        fp.CIG as DEST_CIG,fp.VIS as DEST_VIS,fp.TMP as DEST_TMP, 
# MAGIC        fp.DEW as DEST_DEW,fp.SLP as DEST_SLP,fp.AW1 as DEST_AW1, 
# MAGIC        fp.GA1 as DEST_GA1,fp.GA2 as DEST_GA2,fp.GA3 as DEST_GA3, 
# MAGIC        fp.GA4 as DEST_GA4,fp.GE1 as DEST_GE1,fp.GF1 as DEST_GF1, 
# MAGIC        fp.KA1 as DEST_KA1,fp.KA2 as DEST_KA2,fp.MA1 as DEST_MA1, 
# MAGIC        fp.MD1 as DEST_MD1,fp.MW1 as DEST_MW1,fp.MW2 as DEST_MW2, 
# MAGIC        fp.OC1 as DEST_OC1,fp.OD1 as DEST_OD1,fp.OD2 as DEST_OD2, 
# MAGIC        fp.REM as DEST_REM,fp.EQD as DEST_EQD,fp.AW2 as DEST_AW2, 
# MAGIC        fp.AX4 as DEST_AX4,fp.GD1 as DEST_GD1,fp.AW5 as DEST_AW5, 
# MAGIC        fp.GN1 as DEST_GN1,fp.AJ1 as DEST_AJ1,fp.AW3 as DEST_AW3, 
# MAGIC        fp.MK1 as DEST_MK1,fp.KA4 as DEST_KA4,fp.GG3 as DEST_GG3, 
# MAGIC        fp.AN1 as DEST_AN1,fp.RH1 as DEST_RH1,fp.AU5 as DEST_AU5, 
# MAGIC        fp.HL1 as DEST_HL1,fp.OB1 as DEST_OB1,fp.AT8 as DEST_AT8, 
# MAGIC        fp.AW7 as DEST_AW7,fp.AZ1 as DEST_AZ1,fp.CH1 as DEST_CH1, 
# MAGIC        fp.RH3 as DEST_RH3,fp.GK1 as DEST_GK1,fp.IB1 as DEST_IB1, 
# MAGIC        fp.AX1 as DEST_AX1,fp.CT1 as DEST_CT1,fp.AK1 as DEST_AK1, 
# MAGIC        fp.CN2 as DEST_CN2,fp.OE1 as DEST_OE1,fp.MW5 as DEST_MW5, 
# MAGIC        fp.AO1 as DEST_AO1,fp.KA3 as DEST_KA3,fp.AA3 as DEST_AA3, 
# MAGIC        fp.CR1 as DEST_CR1,fp.CF2 as DEST_CF2,fp.KB2 as DEST_KB2, 
# MAGIC        fp.GM1 as DEST_GM1,fp.AT5 as DEST_AT5,fp.AY2 as DEST_AY2, 
# MAGIC        fp.MW6 as DEST_MW6,fp.MG1 as DEST_MG1,fp.AH6 as DEST_AH6, 
# MAGIC        fp.AU2 as DEST_AU2,fp.GD2 as DEST_GD2,fp.AW4 as DEST_AW4, 
# MAGIC        fp.MF1 as DEST_MF1,fp.AA1 as DEST_AA1,fp.AH2 as DEST_AH2, 
# MAGIC        fp.AH3 as DEST_AH3,fp.OE3 as DEST_OE3,fp.AT6 as DEST_AT6, 
# MAGIC        fp.AL2 as DEST_AL2,fp.AL3 as DEST_AL3,fp.AX5 as DEST_AX5, 
# MAGIC        fp.IB2 as DEST_IB2,fp.AI3 as DEST_AI3,fp.CV3 as DEST_CV3, 
# MAGIC        fp.WA1 as DEST_WA1,fp.GH1 as DEST_GH1,fp.KF1 as DEST_KF1, 
# MAGIC        fp.CU2 as DEST_CU2,fp.CT3 as DEST_CT3,fp.SA1 as DEST_SA1, 
# MAGIC        fp.AU1 as DEST_AU1,fp.KD2 as DEST_KD2,fp.AI5 as DEST_AI5, 
# MAGIC        fp.GO1 as DEST_GO1,fp.GD3 as DEST_GD3,fp.CG3 as DEST_CG3, 
# MAGIC        fp.AI1 as DEST_AI1,fp.AL1 as DEST_AL1,fp.AW6 as DEST_AW6, 
# MAGIC        fp.MW4 as DEST_MW4,fp.AX6 as DEST_AX6,fp.CV1 as DEST_CV1, 
# MAGIC        fp.ME1 as DEST_ME1,fp.KC2 as DEST_KC2,fp.CN1 as DEST_CN1, 
# MAGIC        fp.UA1 as DEST_UA1,fp.GD5 as DEST_GD5,fp.UG2 as DEST_UG2, 
# MAGIC        fp.AT3 as DEST_AT3,fp.AT4 as DEST_AT4,fp.GJ1 as DEST_GJ1, 
# MAGIC        fp.MV1 as DEST_MV1,fp.GA5 as DEST_GA5,fp.CT2 as DEST_CT2, 
# MAGIC        fp.CG2 as DEST_CG2,fp.ED1 as DEST_ED1,fp.AE1 as DEST_AE1, 
# MAGIC        fp.CO1 as DEST_CO1,fp.KE1 as DEST_KE1,fp.KB1 as DEST_KB1, 
# MAGIC        fp.AI4 as DEST_AI4,fp.MW3 as DEST_MW3,fp.KG2 as DEST_KG2, 
# MAGIC        fp.AA2 as DEST_AA2,fp.AX2 as DEST_AX2,fp.AY1 as DEST_AY1, 
# MAGIC        fp.RH2 as DEST_RH2,fp.OE2 as DEST_OE2,fp.CU3 as DEST_CU3, 
# MAGIC        fp.MH1 as DEST_MH1,fp.AM1 as DEST_AM1,fp.AU4 as DEST_AU4, 
# MAGIC        fp.GA6 as DEST_GA6,fp.KG1 as DEST_KG1,fp.AU3 as DEST_AU3, 
# MAGIC        fp.AT7 as DEST_AT7,fp.KD1 as DEST_KD1,fp.GL1 as DEST_GL1, 
# MAGIC        fp.IA1 as DEST_IA1,fp.GG2 as DEST_GG2,fp.OD3 as DEST_OD3, 
# MAGIC        fp.UG1 as DEST_UG1,fp.CB1 as DEST_CB1,fp.AI6 as DEST_AI6, 
# MAGIC        fp.CI1 as DEST_CI1,fp.CV2 as DEST_CV2,fp.AZ2 as DEST_AZ2, 
# MAGIC        fp.AD1 as DEST_AD1,fp.AH1 as DEST_AH1,fp.WD1 as DEST_WD1, 
# MAGIC        fp.AA4 as DEST_AA4,fp.KC1 as DEST_KC1,fp.IA2 as DEST_IA2, 
# MAGIC        fp.CF3 as DEST_CF3,fp.AI2 as DEST_AI2,fp.AT1 as DEST_AT1, 
# MAGIC        fp.GD4 as DEST_GD4,fp.AX3 as DEST_AX3,fp.AH4 as DEST_AH4, 
# MAGIC        fp.KB3 as DEST_KB3,fp.CU1 as DEST_CU1,fp.CN4 as DEST_CN4, 
# MAGIC        fp.AT2 as DEST_AT2,fp.CG1 as DEST_CG1,fp.CF1 as DEST_CF1, 
# MAGIC        fp.GG1 as DEST_GG1,fp.MV2 as DEST_MV2,fp.CW1 as DEST_CW1, 
# MAGIC        fp.GG4 as DEST_GG4,fp.AB1 as DEST_AB1,fp.AH5 as DEST_AH5, 
# MAGIC        fp.CN3 as DEST_CN3,fp.WEATHER_DATE as DEST_WEATHER_DATE 
# MAGIC FROM   flights_and_weather_dest fp 

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from flights_and_weather_combined

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from flights_processed

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from flights_and_weather_combined
# MAGIC limit 1

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT a.ORIGIN, ((CASE WHEN b.CNT_MISSING IS NOT NULL THEN b.CNT_MISSING
# MAGIC                         ELSE 0 END)/a.TOTAL) AS PROP_MISSING, a.TOTAL, b.CNT_MISSING
# MAGIC FROM (SELECT ORIGIN, COUNT(*) AS TOTAL 
# MAGIC       FROM flights_and_weather_combined
# MAGIC       GROUP BY ORIGIN) a
# MAGIC JOIN (SELECT ORIGIN, count(*) AS CNT_MISSING
# MAGIC       FROM flights_and_weather_combined
# MAGIC       WHERE ORIGIN_WEATHER_STATION IS NULL
# MAGIC       GROUP BY ORIGIN) b
# MAGIC ON a.ORIGIN=b.ORIGIN
# MAGIC ORDER BY PROP_MISSING DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT a.DEST, ((CASE WHEN b.CNT_MISSING IS NOT NULL THEN b.CNT_MISSING
# MAGIC                         ELSE 0 END)/a.TOTAL) AS PROP_MISSING, a.TOTAL, b.CNT_MISSING
# MAGIC FROM (SELECT DEST, COUNT(*) AS TOTAL 
# MAGIC       FROM flights_and_weather_combined
# MAGIC       GROUP BY DEST) a
# MAGIC JOIN (SELECT DEST, count(*) AS CNT_MISSING
# MAGIC       FROM flights_and_weather_combined
# MAGIC       WHERE DEST_WEATHER_STATION IS NULL
# MAGIC       GROUP BY DEST) b
# MAGIC ON a.DEST=b.DEST
# MAGIC ORDER BY PROP_MISSING DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC There are a number of airports that have a majority of flights with missing weather data. One major one, Honolulu was fixed. The others, we are not touching for now. We would like to get back to these at some point.

# COMMAND ----------

# MAGIC %md # Data Wrangling and Cleaning

# COMMAND ----------

# Ensure we are always using the most recent version of the data in the delta lake
flights_and_weather_combined = spark.sql("select * from flights_and_weather_combined")

# COMMAND ----------

def cast_flight_data_types(df):

  df = df.select(df.columns)
  # cast fields to correct data type
  # StringType(), DoubleType(), IntegerType()
  df = df.withColumn('YEAR', df['YEAR'].cast(StringType()))
  df = df.withColumn('QUARTER', df['QUARTER'].cast(StringType()))
  df = df.withColumn('MONTH', df['MONTH'].cast(StringType()))
  df = df.withColumn('DAY_OF_WEEK', df['DAY_OF_WEEK'].cast(StringType()))
  df = df.withColumn('DEP_DELAY', df['DEP_DELAY'].cast(IntegerType()))
  df = df.withColumn('DEP_DELAY_NEW', df['DEP_DELAY_NEW'].cast(IntegerType()))
  df = df.withColumn('DEP_DEL15', df['DEP_DEL15'].cast(IntegerType()))
  df = df.withColumn('DISTANCE', df['DISTANCE'].cast(IntegerType()))

  return df


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Set up final stage of cleaned and pre-processed data for the modeling pipeline

# COMMAND ----------

# MAGIC %md Prepare directories

# COMMAND ----------

flights_and_weather_pipeline_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_pipeline/"

dbutils.fs.rm(flights_and_weather_pipeline_loc + 'processed', recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Apply our cleaning, pruning and imputation routines to the dataset

# COMMAND ----------

flights_and_weather_combined = cast_flight_data_types(flights_and_weather_combined) # clean and cast our data to appropriate data types

cols_to_drop = ['DIV4_AIRPORT', 'DIV4_TOTAL_GTIME', 'DIV3_TOTAL_GTIME',
     'DIV3_WHEELS_ON', 'DIV3_AIRPORT_SEQ_ID', 'DIV3_AIRPORT_ID',
     'DIV3_AIRPORT', 'DIV3_TAIL_NUM', 'DIV3_WHEELS_OFF',
     'DIV4_AIRPORT_ID', 'DIV4_AIRPORT_SEQ_ID', 'DIV4_WHEELS_ON',
     'DIV3_LONGEST_GTIME', 'DIV4_LONGEST_GTIME', 'DIV5_TAIL_NUM',
     'DIV4_TAIL_NUM', 'DIV5_AIRPORT', 'DIV5_WHEELS_OFF',
     'DIV5_LONGEST_GTIME', 'DIV5_TOTAL_GTIME', 'DIV5_WHEELS_ON',
     'DIV5_AIRPORT_SEQ_ID', 'DIV5_AIRPORT_ID', 'DIV4_WHEELS_OFF',
     'DIV2_TAIL_NUM', 'DIV2_WHEELS_OFF', 'DIV2_WHEELS_ON',
     'DIV2_AIRPORT', 'DIV2_TOTAL_GTIME', 'DIV2_LONGEST_GTIME',
     'DIV2_AIRPORT_SEQ_ID', 'DIV2_AIRPORT_ID', 'DIV_ARR_DELAY',
     'DIV_ACTUAL_ELAPSED_TIME', 'DIV1_TAIL_NUM', 'DIV1_WHEELS_OFF',
     'DIV_DISTANCE', 'DIV_REACHED_DEST', 'DIV1_AIRPORT_SEQ_ID',
     'DIV1_TOTAL_GTIME', 'DIV1_WHEELS_ON', 'DIV1_AIRPORT_ID',
     'DIV1_AIRPORT', 'DIV1_LONGEST_GTIME', 'LONGEST_ADD_GTIME',
     'TOTAL_ADD_GTIME', 'FIRST_DEP_TIME']

cols_to_drop = cols_to_drop + ['ACTUAL_ELAPSED_TIME', 
     'AIR_TIME','CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY',
     'LATE_AIRCRAFT_DELAY','IN_FLIGHT_AIR_DELAY']
    
flights_and_weather_combined = flights_and_weather_combined.drop(*cols_to_drop)


# COMMAND ----------

# MAGIC %md Store in Delta Lake format and create a table for access

# COMMAND ----------

flights_and_weather_combined.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_pipeline_loc + 'processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_and_weather_pipeline_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_and_weather_pipeline_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_and_weather_pipeline/processed"

# COMMAND ----------

# MAGIC %md # Feature Engineering
# MAGIC This section includes the modification of our features for use in our models. This process includes:
# MAGIC * Fix the target column, **DEP_DEL15**
# MAGIC * Cleaning the comma-delimited columns in the weather data
# MAGIC   * Select weather columns we want and indicate what columns they should be split into
# MAGIC   * Split the columns
# MAGIC * Remove columns we cannot use.
# MAGIC * Creation of novel columns from the flight and weather data
# MAGIC * PageRank for flights Feature

# COMMAND ----------

# MAGIC %md Clean up the DEP_DEL15  
# MAGIC Assumptions for NULL **DEP_DEL15** values:
# MAGIC * Most are cancelled - drop those rows for now. Will revisit cancellations later.
# MAGIC * If DEP_DEL_15 is NULL and the difference between schedule departure and actual departure is 0 , set DEP_DEL15 -> 0
# MAGIC * If DEP_DEL_15 is NULL and the difference between schedule departure and actual departure is not 0 (5 records), -> drop records

# COMMAND ----------

flights_and_weather_pipeline_processed = spark.sql("SELECT * FROM flights_and_weather_pipeline_processed WHERE CANCELLED = 0")

# COMMAND ----------

def fix_missing_dep_del15(dep_del_15, scheduled_dep_time, actual_dep_time):
    '''Fixes missing DEP_DEL15 value, else returns existing value'''
    diff = scheduled_dep_time - actual_dep_time
    if  diff < 15:
        return 0
    elif diff >= 15 and diff <100:
        return 1
    else:
        return None
fix_missing_dep_del15 = udf(fix_missing_dep_del15)

# COMMAND ----------

flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.withColumn("DEP_DEL15", f.when(flights_and_weather_pipeline_processed["DEP_DEL15"].isNull(), fix_missing_dep_del15("DEP_DEL15","CRS_DEP_TIME","DEP_TIME")).otherwise(flights_and_weather_pipeline_processed["DEP_DEL15"]))

# COMMAND ----------

flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.where('DEP_DEL15 IS NOT NULL')

# COMMAND ----------

flights_and_weather_pipeline_processed.write.option('overwriteSchema', True).mode('overwrite').format('delta').save(flights_and_weather_pipeline_loc + 'processed')

# COMMAND ----------

flights_and_weather_pipeline_processed = spark.sql("SELECT * FROM flights_and_weather_pipeline_processed")

# COMMAND ----------

# MAGIC %md ## Clean up comma delimited columns

# COMMAND ----------

cols_to_split = {'WND' : {
                          'WND_DIRECTION_ANGLE': {'data_type': 'int', 'missing_value': 99, 'include': True},
                          'WND_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'WND_TYPE_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                          'WND_SPEED_RATE': {'data_type': 'int', 'missing_value': 9999, 'include': True},
                          'WND_SPEED_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                         },
                'CIG' : {
                          'CIG_CEILING_HEIGHT_DIMENSION': {'data_type': 'int', 'missing_value': 99999, 'include': True},
                          'CIG_CEILING_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'CIG_CEILING_DETERMINATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                          'CIG_CAVOK_CODE': {'data_type':'string', 'missing_value': '9', 'include': True}
                        },
                'VIS' : {
                         'VIS_DISTANCE_DIMENSION': {'data_type': 'int', 'missing_value': 999999, 'include': True},
                         'VIS_DISTANCE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                         'VIS_VARIABILITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                         'VIS_QUALITY_VARIABILITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                         },
                'TMP' : {
                         'TMP_AIR_TEMP': {'data_type': 'int', 'missing_value': 9999, 'include': True},
                         'TMP_AIR_TEMP_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                         },
                'DEW' : {
                         'DEW_POINT_TEMP': {'data_type': 'int', 'missing_value': 9999, 'include': True},
                         'DEW_POINT_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                         },
                'SLP' : {
                         'SLP_SEA_LEVEL_PRES': {'data_type': 'int', 'missing_value': 99999, 'include': True},
                         'SLP_SEA_LEVEL_PRES_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                         },
                'AA1' : {
                         'AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY': {'data_type': 'int', 'missing_value': 99, 'include': True},
                         'AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION': {'data_type': 'int', 'missing_value': 9999, 'include': True},
                         'AA1_LIQUID_PRECIPITATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                         'AA1_LIQUID_PRECIPITATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AA2' : {
                         'AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY': {'data_type': 'int', 'missing_value': 99, 'include': True},
                         'AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION': {'data_type': 'int', 'missing_value': 9999, 'include': True},
                         'AA2_LIQUID_PRECIPITATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                         'AA2_LIQUID_PRECIPITATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AA3' : {
                         'AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY': {'data_type': 'int', 'missing_value': 99, 'include': True},
                         'AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION': {'data_type': 'int', 'missing_value': 9999, 'include': True},
                         'AA3_LIQUID_PRECIPITATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                         'AA3_LIQUID_PRECIPITATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AA4' : {
                         'AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY': {'data_type': 'int', 'missing_value': 99, 'include': True},
                         'AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION': {'data_type': 'int', 'missing_value': 9999, 'include': True},
                         'AA4_LIQUID_PRECIPITATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                         'AA4_LIQUID_PRECIPITATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AJ1' : {
                         'SNOW_DEPTH_DIMENSION': {'data_type': 'int', 'missing_value': 9999, 'include': True},
                         'SNOW_DEPTH_CONDITION_CODE' : {'data_type':'string', 'missing_value': '9', 'include': True},
                         'SNOW_DEPTH_QUALITY_CODE' : {'data_type':'string', 'missing_value': '9', 'include': False},
                         'SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 999999, 'include': True},
                         'SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_CODE' : {'data_type':'string', 'missing_value': '9', 'include': False},
                         'SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_QUALITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': False}
                        },
                'AL1' : {
                         'AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY': {'data_type':'int', 'missing_value': 99, 'include': True},
                         'AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 999, 'include': True},
                         'AL1_SNOW_ACCUMULATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                         'AL1_SNOW_ACCUMULATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        },
                'AL2' : {
                         'AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY': {'data_type':'int', 'missing_value': 99, 'include': True},
                         'AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 999, 'include': True},
                         'AL2_SNOW_ACCUMULATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                         'AL2_SNOW_ACCUMULATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        },
                'AL3' : {
                         'AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY': {'data_type':'int', 'missing_value': 99, 'include': True},
                         'AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 999, 'include': True},
                         'AL3_SNOW_ACCUMULATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                         'AL3_SNOW_ACCUMULATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        },
                'AO1' : {
                        'AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY': {'data_type':'int', 'missing_value': 99, 'include': True},
                        'AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 9999, 'include': True},
                        'AO1_LIQUID_PRECIPITATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                        'AO1_LIQUID_PRECIPITATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': False} 
                        },
                'AT1' : {
                        'AT1_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE': {'data_type':'string', 'missing_value': '', 'include': True},
                        'AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT1_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False} 
                        },
                'AT2' : {
                        'AT2_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE': {'data_type':'string', 'missing_value': '', 'include': True},
                        'AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT2_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False} 
                        },
                'AT3' : {
                        'AT3_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE': {'data_type':'string', 'missing_value': '', 'include': True},
                        'AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT3_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False} 
                        },
                'AT4' : {
                        'AT4_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE': {'data_type':'string', 'missing_value': '', 'include': True},
                        'AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT4_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False} 
                        },
                'AT5' : {
                        'AT5_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE': {'data_type':'string', 'missing_value': '', 'include': True},
                        'AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT5_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False} 
                        },
                'AT6' : {
                        'AT6_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE': {'data_type':'string', 'missing_value': '', 'include': True},
                        'AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT6_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False} 
                        },
                'AT7' : {
                        'AT7_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE': {'data_type':'string', 'missing_value': '', 'include': True},
                        'AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT7_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False} 
                        },
                'AT8' : {
                        'AT8_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE': {'data_type':'string', 'missing_value': '', 'include': True},
                        'AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR': {'data_type':'string', 'missing_value': '', 'include': False},
                        'AT8_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False} 
                        },
                'AU1' : {
                        'AU1_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU1_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU1_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU1_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU1_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU1_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU1_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AU2' : {
                        'AU2_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU2_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU2_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU2_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU2_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU2_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU2_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AU3' : {
                        'AU3_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU3_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU3_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU3_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU3_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU3_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU3_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AU4' : {
                        'AU4_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU4_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU4_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU4_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU4_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU4_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU4_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AU5' : {
                        'AU5_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU5_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU5_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU5_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU5_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU5_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU5_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AW1' : {
                        'AW1_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'AW1_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AW2' : {
                        'AW2_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'AW2_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AW3' : {
                        'AW3_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'AW3_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'AW4' : {
                        'AW4_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'AW4_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'ED1' : {
                        'RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE': {'data_type':'int', 'missing_value': 99, 'include': False},
                        'RUNWAY_VISUAL_RANGE_OBSERVATION_RUNWAY_DESIGNATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION': {'data_type':'int', 'missing_value': 9999, 'include': False},
                        'RUNWAY_VISUAL_RANGE_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GA1' : {
                        'GA1_SKY_COVER_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA1_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GA1_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA1_SKY_COVER_LAYER_CLOUD_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA1_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GA2' : {
                        'GA2_SKY_COVER_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA2_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GA2_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA2_SKY_COVER_LAYER_CLOUD_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA2_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GA3' : {
                        'GA3_SKY_COVER_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA3_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GA3_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA3_SKY_COVER_LAYER_CLOUD_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA3_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GA4' : {
                        'GA4_SKY_COVER_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA4_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GA4_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA4_SKY_COVER_LAYER_CLOUD_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA4_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GA5' : {
                        'GA5_SKY_COVER_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA5_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GA5_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA5_SKY_COVER_LAYER_CLOUD_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA5_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GA6' : {
                        'GA6_SKY_COVER_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA6_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GA6_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GA6_SKY_COVER_LAYER_CLOUD_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GA6_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GD1' : {
                        'GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD1_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE': {'data_type':'string', 'missing_value': '9', 'include': True}
                        },
                'GD2' : {
                        'GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD2_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE': {'data_type':'string', 'missing_value': '9', 'include': True}
                        },
                'GD3' : {
                        'GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD3_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE': {'data_type':'string', 'missing_value': '9', 'include': True}
                        },
                'GD4' : {
                        'GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD4_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE': {'data_type':'string', 'missing_value': '9', 'include': True}
                        },
                'GD5' : {
                        'GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GD5_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE': {'data_type':'string', 'missing_value': '9', 'include': True}
                        },
                'GE1' : {
                        'SKY_CONDITION_OBSERVATION_CONVECTIVE_CLOUD_ATTRIBUTE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'SKY_CONDITION_OBSERVATION_VERTICAL_DATUM_ATTRIBUTE': {'data_type':'string', 'missing_value': '999999', 'include': False},
                        'SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE': {'data_type':'int', 'missing_value': 99999, 'include': True}
                        },
                'GF1' : {
                        'SKY_CONDITION_OBSERVATION_TOTAL_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'SKY_CONDITION_OBSERVATION_TOTAL_OPAQUE_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_COVERAGE_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'SKY_CONDITION_OBSERVATION_TOTAL_LOWEST_CLOUD_COVER_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_LOWEST_CLOUD_COVER_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'SKY_CONDITION_OBSERVATION_LOW_CLOUD_GENUS_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'SKY_CONDITION_OBSERVATION_QUALITY_LOW_CLOUD_GENUS_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'SKY_CONDITION_OBSERVATION_MID_CLOUD_GENUS_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'SKY_CONDITION_OBSERVATION_QUALITY_MID_CLOUD_GENUS_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'SKY_CONDITION_OBSERVATION_HIGH_CLOUD_GENUS_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'SKY_CONDITION_OBSERVATION_QUALITY_HIGH_CLOUD_GENUS_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        } ,
                'GG1' : {
                        'GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG1_BELOW_STATION_CLOUD_LAYER_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG1_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG1_BELOW_STATION_CLOUD_LAYER_TOP_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG1_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GG2' : {
                        'GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG2_BELOW_STATION_CLOUD_LAYER_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG2_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG2_BELOW_STATION_CLOUD_LAYER_TOP_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG2_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GG3' : {
                        'GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG3_BELOW_STATION_CLOUD_LAYER_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG3_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG3_BELOW_STATION_CLOUD_LAYER_TOP_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG3_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'GG4' : {
                        'GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                        'GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG4_BELOW_STATION_CLOUD_LAYER_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG4_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        'GG4_BELOW_STATION_CLOUD_LAYER_TOP_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GG4_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'IA1' : {
                        'GROUND_SURFACE_OBSERVATION_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                        'GROUND_SURFACE_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                'OC1' : {
                        'WIND_GUST_OBSERVATION_SPEED_RATE': {'data_type':'int', 'missing_value': 9999, 'include': True},
                        'WIND_GUST_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        }
                }



# COMMAND ----------

# MAGIC %md Need to correct for missing codes
# MAGIC 
# MAGIC 
# MAGIC e.g.
# MAGIC data_with_missing_values_corrected = correct_missing_codes(dataset_with_weather_columns_added)

# COMMAND ----------

def process_weather_data(df, col_prefix, cols_to_split = {}):
    for col, sub_cols in cols_to_split.items():
        if col_prefix == '':
            col_split = f.split(df[col], ',')
        else:
            col_split = f.split(df[f'{col_prefix}_{col}'], ',')
        c = 0
        for sub_col, sub_col_params in sub_cols.items():
            if col_prefix == '':
                df = df.withColumn(sub_col, col_split.getItem(c).cast(sub_col_params['data_type']))
            else:
                df = df.withColumn(f'{col_prefix}_{sub_col}', col_split.getItem(c).cast(sub_col_params['data_type']))
            c += 1
    
    return df

# COMMAND ----------

flights_and_weather_pipeline_processed.printSchema()

# COMMAND ----------

# this data structure simply denotes the coluns that caused errors when splitting columns because they were not there in our weather data
# we keep them here for posterity.
removed_from_dict = {'AG1' : {
                         'PRECIPITATION_ESTIMATED_OBSERVATION_DISCREPANCY_CODE' : {'data_type': 'string', 'missing_value': '9', 'include': True},
                         'PRECIPITATION_ESTIMATED_OBSERVATION_ESTIMATE_WATER_DEPTH_DIMENSION': {'data_type': 'int', 'missing_value': 999, 'include': True}
                        },
                     'AL4' : {
                         'AL4_SNOW_ACCUMULATION_PERIOD_QUANTITY': {'data_type':'int', 'missing_value': 99, 'include': True},
                         'AL4_SNOW_ACCUMULATION_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 999, 'include': True},
                         'AL4_SNOW_ACCUMULATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                         'AL4_SNOW_ACCUMULATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                        },
                     'AO2' : {
                        'AO2_LIQUID_PRECIPITATION_PERIOD_QUANTITY': {'data_type':'int', 'missing_value': 99, 'include': True},
                        'AO2_LIQUID_PRECIPITATION_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 9999, 'include': True},
                        'AO2_LIQUID_PRECIPITATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                        'AO2_LIQUID_PRECIPITATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': False} 
                        },
                     'AO3' : {
                        'AO3_LIQUID_PRECIPITATION_PERIOD_QUANTITY': {'data_type':'int', 'missing_value': 99, 'include': True},
                        'AO3_LIQUID_PRECIPITATION_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 9999, 'include': True},
                        'AO3_LIQUID_PRECIPITATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                        'AO3_LIQUID_PRECIPITATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': False} 
                        },
                     'AO4' : {
                        'AO4_LIQUID_PRECIPITATION_PERIOD_QUANTITY': {'data_type':'int', 'missing_value': 99, 'include': True},
                        'AO4_LIQUID_PRECIPITATION_DEPTH_DIMENSION': {'data_type':'int', 'missing_value': 9999, 'include': True},
                        'AO4_LIQUID_PRECIPITATION_CONDITION_CODE': {'data_type':'string', 'missing_value': '9', 'include': False},
                        'AO4_LIQUID_PRECIPITATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': False} 
                        },
                     'AU6' : {
                        'AU6_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU6_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU6_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU6_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU6_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU6_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                        'AU6_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                        },
                      'AU7' : {
                              'AU7_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU7_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU7_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU7_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU7_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU7_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU7_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                              },
                      'AU8' : {
                              'AU8_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU8_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU8_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU8_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU8_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU8_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU8_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                              },
                      'AU9' : {
                              'AU9_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU9_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU9_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU9_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU9_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU9_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                              'AU9_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                              },
                         'GD6' : {
                          'GD6_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE': {'data_type':'string', 'missing_value': '9', 'include': True},
                          'GD6_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2': {'data_type':'string', 'missing_value': '99', 'include': True},
                          'GD6_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'GD6_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                          'GD6_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'GD6_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE': {'data_type':'string', 'missing_value': '9', 'include': True}
                          },
                       'GG5' : {
                          'GG5_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                          'GG5_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'GG5_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                          'GG5_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'GG5_BELOW_STATION_CLOUD_LAYER_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                          'GG5_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'GG5_BELOW_STATION_CLOUD_LAYER_TOP_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                          'GG5_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                          },
                  'GG6' : {
                          'GG6_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                          'GG6_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'GG6_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION': {'data_type':'int', 'missing_value': 99999, 'include': True},
                          'GG6_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'GG6_BELOW_STATION_CLOUD_LAYER_TYPE_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                          'GG6_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False},
                          'GG6_BELOW_STATION_CLOUD_LAYER_TOP_CODE': {'data_type':'string', 'missing_value': '99', 'include': True},
                          'GG6_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE': {'data_type':'string', 'missing_value': '', 'include': False}
                          },
                    }


# COMMAND ----------

#split out comma-delimited fields
flights_and_weather_pipeline_processed = process_weather_data(flights_and_weather_pipeline_processed, 'ORIGIN', cols_to_split)
flights_and_weather_pipeline_processed = process_weather_data(flights_and_weather_pipeline_processed, 'DEST', cols_to_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop columns that won't be useful

# COMMAND ----------

### --- ALL COLUMNS TO BE DROPPED IN THIS CELL ARE FROM THE WEATHER DATA ---###
#individual fields from weather data that will not be processed and simply dropped
drop = ['AB1', 'AC1', 'AD1', 'AE1', 'AH1', 'AH2',\
        'AH3', 'AH4', 'AH5', 'AH6','AI1', 'AI2', 'AI3', 'AI4', 'AI5',\
        'AI6','AK1', 'AM1','AN1','AP1','AP3',\
        'AP3','AP4','AX1','AX2','AX3','AX4','AX5','AX6','AY1','AY2',\
        'AZ1','AZ2','CB1','CB2','CF1','CF2','CF3','CG1', 'CG2', 'CG3'\
        'CH1', 'CH2', 'CI1', 'CN1', 'CN2', 'CN3', 'CN4', 'CO1', 'CO2',\
        'CO1', 'CO2', 'CO3', 'CO4','CO5', 'CO6', 'CO7', 'CO8', 'CO9',\
        'CR1', 'CT1', 'CT2', 'CT3','CU1', 'CU2', 'CU3','CV1', 'CV2', 'CV3',\
        'CW1','CX1', 'CX2', 'CX3', 'GH1', 'GJ1', 'GK1', 'GL1', 'GM1', 'GN1',\
        'GO1', 'GP1', 'GQ1', 'GR1', 'IA2','IB1','IB2','IC1','KA1','KA2','KA3',\
        'KA4', 'KB1','KB2','KB3','KC1','KC2','KD1','KD2','KE1','KF1','KG1','KG2',\
        'MA1','MD1','ME1','MF1','MG1','MH1','MK1','MV1','MV2','MV3','MV4','MV5','MV6',\
        'MV7','MW1','MW2','MW3','MW4','MW5','MW6','MW7','OA1','OA2', 'OA3', 'OB1','OB2',\
        'OD1','OD2','OD3','SA1','ST1', 'UA1', 'UG1', 'UG2', 'WA1','WD1','WG1','WJ1','REM','EQD',\
        'QNN', 'RH1', 'RH2', 'RH3']

# remove groups of cols
group_letters = ['Q','P','R', 'C', 'D', 'N']
group_cols_to_drop = []
for gl in group_letters:
    for i in range(1,100):
        #single digits are pre-padded with 0
        if i < 10:
            group_cols_to_drop.append(f'{gl}0{i}')
        else:
            group_cols_to_drop.append(f'{gl}{i}')

# drop columns we don't care about
cols_to_drop = drop + group_cols_to_drop
cols_to_drop_prefixed_origin = [f'ORIGIN_{col}' for col in cols_to_drop]
cols_to_drop_prefixed_dest = [f'DEST_{col}' for col in cols_to_drop]
cols_to_drop_prefixed = cols_to_drop_prefixed_origin + cols_to_drop_prefixed_dest
flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.drop(*cols_to_drop_prefixed)

#prefix and drop the comma-delimited columns the splits were derived from
cd_cols_prefixed_origin = [f'ORIGIN_{col}' for col in cols_to_split.keys()]
cd_cols_prefixed_dest = [f'DEST_{col}' for col in cols_to_split.keys()]
cd_cols_prefixed = cd_cols_prefixed_origin + cd_cols_prefixed_dest
flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.drop(*cd_cols_prefixed)

# prefix and drop other weather columns
ctrl_cols_to_drop = ['SOURCE','STATION','LAT', 'LON', 'ELEV', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CODE']
ctrl_cols_to_drop_prefixed_origin = [f'ORIGIN_WEATHER_{col}' for col in ctrl_cols_to_drop]
ctrl_cols_to_drop_prefixed_dest = [f'DEST_WEATHER_{col}' for col in ctrl_cols_to_drop]
ctrl_cols_to_drop_prefixed = ctrl_cols_to_drop_prefixed_origin + ctrl_cols_to_drop_prefixed_dest
flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.drop(*ctrl_cols_to_drop_prefixed)

#misc cols to drop not caught before
misc_ctd = ['QUALITY_CONTROL', 'AW5', 'HL1', 'AW7', 'CH1', 'OE1', 'OE3', 'CG3', 'AW6', 'OE2', 'WEATHER_DATE']
misc_ctd_prefixed_origin = [f'ORIGIN_{col}' for col in misc_ctd]
misc_ctd_prefixed_dest = [f'DEST_{col}' for col in misc_ctd]
misc_ctd_prefixed = misc_ctd_prefixed_origin + misc_ctd_prefixed_dest
flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.drop(*misc_ctd_prefixed)

# COMMAND ----------

### --- ALL COLUMNS TO BE DROPPED IN THIS CELL ARE FROM THE FLIGHTS DATA ---###
flights_cols_to_drop = ['DIV5_TAIL_NUM', 'DIV4_WHEELS_ON', 'DIV3_AIRPORT',
       'DIV3_AIRPORT_ID', 'DIV3_AIRPORT_SEQ_ID', 'DIV3_WHEELS_ON',
       'DIV3_TOTAL_GTIME', 'DIV3_LONGEST_GTIME', 'DIV3_WHEELS_OFF',
       'DIV3_TAIL_NUM', 'DIV4_AIRPORT', 'DIV4_AIRPORT_SEQ_ID',
       'DIV4_AIRPORT_ID', 'DIV4_TOTAL_GTIME', 'DIV5_AIRPORT_SEQ_ID',
       'DIV5_WHEELS_OFF', 'DIV5_LONGEST_GTIME', 'DIV4_LONGEST_GTIME',
       'DIV5_WHEELS_ON', 'DIV5_TOTAL_GTIME', 'DIV5_AIRPORT_ID',
       'DIV5_AIRPORT', 'DIV4_TAIL_NUM', 'DIV4_WHEELS_OFF',
       'DIV2_TAIL_NUM', 'DIV2_WHEELS_OFF', 'DIV2_AIRPORT_ID',
       'DIV2_LONGEST_GTIME', 'DIV2_TOTAL_GTIME', 'DIV2_WHEELS_ON',
       'DIV2_AIRPORT_SEQ_ID', 'DIV2_AIRPORT', 'DIV_ACTUAL_ELAPSED_TIME',
       'DIV_ARR_DELAY', 'DIV1_WHEELS_OFF', 'DIV1_TAIL_NUM',
       'DIV_DISTANCE', 'DIV_REACHED_DEST', 'DIV1_AIRPORT_SEQ_ID',
       'DIV1_WHEELS_ON', 'DIV1_LONGEST_GTIME', 'DIV1_AIRPORT',
       'DIV1_AIRPORT_ID', 'DIV1_TOTAL_GTIME', 'FIRST_DEP_TIME',
       'LONGEST_ADD_GTIME', 'TOTAL_ADD_GTIME', 'CANCELLATION_CODE',
       'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 
       'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 
       'ORIGIN_CITY_MARKET_ID', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS',
       'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID',
       'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'DEST_STATE_ABR',
       'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC',
       'CRS_DEP_TIME', 'DEP_TIME', 'DEP_TIME_BLK',
       'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON',
       'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME',
       'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15',
       'ARR_DELAY_GROUP', 'ARR_TIME_BLK',  'CANCELLED',
       'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME',
       'AIR_TIME', 'FLIGHTS', 'DISTANCE_GROUP', 
       'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY',
       'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DIV_AIRPORT_LANDINGS',
       'IN_FLIGHT_AIR_DELAY', 'DAY_OF_MONTH', 'CRS_DEP_TIME_HOUR',
       'IATA_ORIGIN', 'NEAREST_STATION_ID_ORIGIN', 'NEAREST_STATION_DIST_ORIGIN',
       'IATA_DEST', 'NEAREST_STATION_ID_DEST', 'NEAREST_STATION_DIST_DEST',
       'IATA', 'AIRPORT_TZ_NAME', 'FLIGHT_TIME_UTC',
       'WEATHER_PREDICTION_TIME_UTC'] 
flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.drop(*flights_cols_to_drop)

# COMMAND ----------

display(flights_and_weather_pipeline_processed)

# COMMAND ----------

# MAGIC %md
# MAGIC Save the trimmed dataset back to Delta Lake

# COMMAND ----------

flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.withColumnRenamed('ORIGIN_SNOW_DEPTH_EQUIVALENT WATER CONDITION QUALITY_CODE','ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_QUALITY_CODE')
flights_and_weather_pipeline_processed = flights_and_weather_pipeline_processed.withColumnRenamed('DEST_SNOW_DEPTH_EQUIVALENT WATER CONDITION QUALITY_CODE','DEST_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_QUALITY_CODE')

# COMMAND ----------

flights_and_weather_pipeline_processed.write.option('overwriteSchema', True).mode('overwrite').format('delta').save(flights_and_weather_pipeline_loc + 'processed')

# COMMAND ----------

# MAGIC %md
# MAGIC Read the Delta Table back in because we checkpointed here at the end of a working session

# COMMAND ----------

flights_and_weather_pipeline_processed = spark.sql('SELECT * FROM flights_and_weather_pipeline_processed')

# COMMAND ----------

# MAGIC %md
# MAGIC The weather data uses specific values to denote missing values instead of nulls. We want to convert them to null for consistency

# COMMAND ----------

def correct_missing_codes(df, col_prefix, col_ref):
    # iterate over the columns that we have a split reference for
    for col_group_key, col_group in col_ref.items():
        for col, col_params in col_group.items():
            # prefix to make the actual column name
            if col_prefix != '':
                col_name = f'{col_prefix}_{col}'
            else:
                col_name = col
            # we only care about fields we will ultimately include
            if col_params['include'] == True: 
                # when the value in col_name is equal to the mapped missing value, replace it with None, otherwise, pass the value through
                df = df.withColumn(col_name, f.when(df[col_name] == col_params['missing_value'], f.lit(None)).otherwise(df[col_name]))
    return df

# COMMAND ----------

flights_and_weather_pipeline_processed = correct_missing_codes(flights_and_weather_pipeline_processed, 'ORIGIN', cols_to_split)
flights_and_weather_pipeline_processed = correct_missing_codes(flights_and_weather_pipeline_processed, 'DEST', cols_to_split)

# COMMAND ----------

flights_and_weather_pipeline_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_pipeline/"
flights_and_weather_pipeline_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_pipeline_loc + 'processed')

# COMMAND ----------

# MAGIC %md ### Create Novel Columns

# COMMAND ----------

# MAGIC %md #### Weather Rolling Average

# COMMAND ----------

def calculate_rolling_average(df, num_hours, avg_col):
    '''calculates the rolling average of avg_col over num_hours'''
    
    #oneline func to convert 1 hour to seconds
    convert_hours = lambda h: h*3600

    # grouped window to calculate n hour moving average
    w = Window.partitionBy('WEATHER_STATION').orderBy(f.col('WEATHER_DATE').cast('long')).rangeBetween(-convert_hours(num_hours),0)
    df = df.withColumn(f'{avg_col}_{num_hours}_RA',f.avg(avg_col).over(w))
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC Because the flights do not have every weather record, we have to calculate the rolling averages on the actual weather data and then join them later. Unfortunately, that requires a bit more work. First, we have to run the same column splits and drops we performed on the combined data on the weather data. Then we can calculate the rolling averages on all the numeric data, create a sub-table with just these averages, and finally join it to the combined dataset.

# COMMAND ----------

# read in the weather table from Delta Lake
weather_processed = spark.sql('SELECT * FROM weather_processed')

# COMMAND ----------

# split columns
weather_processed = process_weather_data(weather_processed, '', cols_to_split)

# COMMAND ----------

#individual fields from weather data that will not be processed and simply dropped
drop = ['AB1', 'AC1', 'AD1', 'AE1', 'AH1', 'AH2',\
        'AH3', 'AH4', 'AH5', 'AH6','AI1', 'AI2', 'AI3', 'AI4', 'AI5',\
        'AI6','AK1', 'AM1','AN1','AP1','AP3',\
        'AP3','AP4','AX1','AX2','AX3','AX4','AX5','AX6','AY1','AY2',\
        'AZ1','AZ2','CB1','CB2','CF1','CF2','CF3','CG1', 'CG2', 'CG3'\
        'CH1', 'CH2', 'CI1', 'CN1', 'CN2', 'CN3', 'CN4', 'CO1', 'CO2',\
        'CO1', 'CO2', 'CO3', 'CO4','CO5', 'CO6', 'CO7', 'CO8', 'CO9',\
        'CR1', 'CT1', 'CT2', 'CT3','CU1', 'CU2', 'CU3','CV1', 'CV2', 'CV3',\
        'CW1','CX1', 'CX2', 'CX3', 'GH1', 'GJ1', 'GK1', 'GL1', 'GM1', 'GN1',\
        'GO1', 'GP1', 'GQ1', 'GR1', 'IA2','IB1','IB2','IC1','KA1','KA2','KA3',\
        'KA4', 'KB1','KB2','KB3','KC1','KC2','KD1','KD2','KE1','KF1','KG1','KG2',\
        'MA1','MD1','ME1','MF1','MG1','MH1','MK1','MV1','MV2','MV3','MV4','MV5','MV6',\
        'MV7','MW1','MW2','MW3','MW4','MW5','MW6','MW7','OA1','OA2', 'OA3', 'OB1','OB2',\
        'OD1','OD2','OD3','SA1','ST1', 'UA1', 'UG1', 'UG2', 'WA1','WD1','WG1','WJ1','REM','EQD',\
        'QNN', 'RH1', 'RH2', 'RH3']

# remove groups of cols
group_letters = ['Q','P','R', 'C', 'D', 'N']
group_cols_to_drop = []
for gl in group_letters:
    for i in range(1,100):
        #single digits are pre-padded with 0
        if i < 10:
            group_cols_to_drop.append(f'{gl}0{i}')
        else:
            group_cols_to_drop.append(f'{gl}{i}')

# drop columns we don't care about
cols_to_drop = drop + group_cols_to_drop
weather_processed = weather_processed.drop(*cols_to_drop)

#drop the comma-delimited columns the splits were derived from
weather_processed = weather_processed.drop(*list(cols_to_split.keys()))

# prefix and drop other weather columns
ctrl_cols_to_drop = ['WEATHER_SOURCE','WEATHER_LAT', 'WEATHER_LON', 'WEATHER_ELEV', 'WEATHER_NAME', 'WEATHER_REPORT_TYPE', 'WEATHER_CALL_SIGN', 'QUALITY_CONTROL']
weather_processed = weather_processed.drop(*ctrl_cols_to_drop)

#misc columns to drop
misc_ctd = ['AW5', 'HL1', 'AW7', 'CH1', 'OE1', 'OE3', 'CG3', 'AW6', 'OE2']
weather_processed = weather_processed.drop(*misc_ctd)

# COMMAND ----------

# process the null values
weather_processed = correct_missing_codes(weather_processed, '', cols_to_split)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have eliminated the un-needed columns, we can calculate rolling averages for all the numeric fields

# COMMAND ----------

# MAGIC %md
# MAGIC #### BEGIN ROLLING AVERAGE TEST ON SAMPLE (DELETE LATER)

# COMMAND ----------

display(weather_sample)

# COMMAND ----------

weather_sample = weather_processed.where('WEATHER_DATE >= TO_DATE("01/01/2015", "MM/dd/yyyy") AND WEATHER_DATE <= TO_DATE("03/31/2015", "MM/dd/yyyy")')
num_hours = 3
for group, cols in cols_to_split.items():
    for col, col_properties in cols.items():
        if col_properties['data_type'] == 'int':
            weather_sample = calculate_rolling_average(weather_sample, num_hours, col)

# COMMAND ----------

# MAGIC %md
# MAGIC #### END ROLLING AVERAGE TEST ON SAMPLE (DELETE LATER)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we need to subset our weather data to only the stations we care about to save processing time.

# COMMAND ----------

airports_stations_processed = spark.sql('SELECT * FROM airport_stations_processed')

# COMMAND ----------

airports_stations_processed = airports_stations_processed.withColumn("NEAREST_STATION_ID",f.col("_2")["_1"]).withColumn("NEAREST_STATION_DIST",f.col("_2")["_2"])
airports_stations_processed = airports_stations_processed.drop("_2")

# COMMAND ----------

airports_stations_processed = airports_stations_processed.withColumnRenamed("_1","AIRPORT_IATA")

# COMMAND ----------

# MAGIC %md
# MAGIC Save this down to Delta

# COMMAND ----------

airports_stations_processed.write.option('overwriteSchema', True).mode('overwrite').format('delta').save(f'/airline_delays/{username}/DLRS/airport_stations/processed')

# COMMAND ----------

#replace Honolulu
#create udfs for fixing the HNL nearest stations and weather keys
def replace_HNL_weather_station_id(id):
    return "99999921515"
# register UDFs
spark.udf.register("replace_HNL_weather_station_id", replace_HNL_weather_station_id)

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE airport_stations_processed
# MAGIC SET NEAREST_STATION_ID = replace_HNL_weather_station_id(NEAREST_STATION_ID)
# MAGIC WHERE AIRPORT_IATA = "HNL"

# COMMAND ----------

airports_stations_processed = spark.sql('SELECT * FROM airport_stations_processed')

# COMMAND ----------

weather_processed = weather_processed.join(airports_stations_processed,weather_processed.WEATHER_STATION == airports_stations_processed.NEAREST_STATION_ID,'right')

# COMMAND ----------

weather_processed.count()

# COMMAND ----------

num_hours = 3
for group, cols in cols_to_split.items():
    for col, col_properties in cols.items():
        if col_properties['data_type'] == 'int':
            weather_processed = calculate_rolling_average(weather_processed, num_hours, col)

# COMMAND ----------

# MAGIC %md Save table with rolling averages to delta lake

# COMMAND ----------

weather_engineered_loc = f"/airline_delays/{username}/DLRS/weather_engineered/"

dbutils.fs.rm(weather_engineered_loc + 'processed', recurse=True)

# COMMAND ----------

weather_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(weather_engineered_loc + 'processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS weather_engineered_processed;
# MAGIC 
# MAGIC CREATE TABLE weather_engineered_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/weather_engineered/processed"

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the rolling averages for the weather records we care about, we can create a smaller table to join to the flights data.

# COMMAND ----------

weather_engineered_processed = spark.sql('select * from weather_engineered_processed')

# COMMAND ----------

# drop the columns that the RAs were derived from
features_to_drop = []
for col_group in cols_to_split.values():
    for sub_col in col_group.keys():
        features_to_drop.append(sub_col)
weather_engineered_processed = weather_engineered_processed.drop(*features_to_drop)

# COMMAND ----------

#drop misc columns we no longer need
misc_wra_cols_to_drop = ['WEATHER_DATE','WEATHER_STATION','AIRPORT_IATA','NEAREST_STATION_ID','NEAREST_STATION_DIST']
weather_engineered_processed = weather_engineered_processed.drop(*misc_wra_cols_to_drop)

# COMMAND ----------

display(weather_engineered_processed)

# COMMAND ----------

# MAGIC %md
# MAGIC Now overwite this table on the Delta Lake

# COMMAND ----------

weather_engineered_processed.write.option('overwriteSchema', True).mode('overwrite').format('delta').save(weather_engineered_loc + 'processed')

# COMMAND ----------

flights_and_weather_pipeline_processed.columns

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE flights_and_weather_ra_origin 
# MAGIC USING DELTA 
# MAGIC LOCATION '/airline_delays/$username/DLRS/flights_and_weather_ra_origin/processed' 
# MAGIC AS SELECT * FROM flights_and_weather_pipeline_processed fp LEFT
# MAGIC JOIN (SELECT * 
# MAGIC       FROM (SELECT *, ROW_NUMBER() 
# MAGIC             OVER (PARTITION BY wp.WEATHER_KEY
# MAGIC                   ORDER BY wp.WEATHER_KEY ASC) AS ORIGIN_ROW_NUM
# MAGIC             FROM weather_engineered_processed wp) AS w
# MAGIC       WHERE w.ORIGIN_ROW_NUM = 1) AS w1 
# MAGIC ON fp.ORIGIN_WEATHER_KEY = w1.WEATHER_KEY

# COMMAND ----------

spark.sql('SELECT * FROM flights_and_weather_ra_origin LIMIT 1').columns

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE flights_and_weather_ra_origin_renamed USING DELTA LOCATION '/airline_delays/$username/DLRS/flights_and_weather_ra_origin_renamed/processed' AS
# MAGIC SELECT  fp.YEAR,
# MAGIC         fp.QUARTER,
# MAGIC         fp.MONTH,
# MAGIC         fp.DAY_OF_WEEK,
# MAGIC         fp.TAIL_NUM,
# MAGIC         fp.ORIGIN,
# MAGIC         fp.ORIGIN_CITY_NAME,
# MAGIC         fp.DEST,
# MAGIC         fp.DEST_CITY_NAME,
# MAGIC         fp.DEP_DELAY,
# MAGIC         fp.DEP_DELAY_NEW,
# MAGIC         fp.DEP_DEL15,
# MAGIC         fp.DEP_DELAY_GROUP,
# MAGIC         fp.DISTANCE,
# MAGIC         fp.FL_DATE,
# MAGIC         fp.ORIGIN_WEATHER_KEY,
# MAGIC         fp.DEST_WEATHER_KEY,
# MAGIC         fp.ORIGIN_WND_DIRECTION_ANGLE,
# MAGIC         fp.ORIGIN_WND_QUALITY_CODE,
# MAGIC         fp.ORIGIN_WND_TYPE_CODE,
# MAGIC         fp.ORIGIN_WND_SPEED_RATE,
# MAGIC         fp.ORIGIN_WND_SPEED_QUALITY_CODE,
# MAGIC         fp.ORIGIN_CIG_CEILING_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_CIG_CEILING_QUALITY_CODE,
# MAGIC         fp.ORIGIN_CIG_CEILING_DETERMINATION_CODE,
# MAGIC         fp.ORIGIN_CIG_CAVOK_CODE,
# MAGIC         fp.ORIGIN_VIS_DISTANCE_DIMENSION,
# MAGIC         fp.ORIGIN_VIS_DISTANCE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_VIS_VARIABILITY_CODE,
# MAGIC         fp.ORIGIN_VIS_QUALITY_VARIABILITY_CODE,
# MAGIC         fp.ORIGIN_TMP_AIR_TEMP,
# MAGIC         fp.ORIGIN_TMP_AIR_TEMP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_DEW_POINT_TEMP,
# MAGIC         fp.ORIGIN_DEW_POINT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_SLP_SEA_LEVEL_PRES,
# MAGIC         fp.ORIGIN_SLP_SEA_LEVEL_PRES_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_CONDITION_CODE,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_QUALITY_CODE,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_CODE,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AW1_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.ORIGIN_AW1_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AW2_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.ORIGIN_AW2_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AW3_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.ORIGIN_AW3_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AW4_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.ORIGIN_AW4_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_RUNWAY_DESIGNATOR_CODE,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_CONVECTIVE_CLOUD_ATTRIBUTE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_VERTICAL_DATUM_ATTRIBUTE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_TOTAL_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_TOTAL_OPAQUE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_TOTAL_LOWEST_CLOUD_COVER_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_LOWEST_CLOUD_COVER_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_LOW_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_LOW_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_MID_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_MID_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_HIGH_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_HIGH_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GROUND_SURFACE_OBSERVATION_CODE,
# MAGIC         fp.ORIGIN_GROUND_SURFACE_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_WIND_GUST_OBSERVATION_SPEED_RATE,
# MAGIC         fp.ORIGIN_WIND_GUST_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_WND_DIRECTION_ANGLE,
# MAGIC         fp.DEST_WND_QUALITY_CODE,
# MAGIC         fp.DEST_WND_TYPE_CODE,
# MAGIC         fp.DEST_WND_SPEED_RATE,
# MAGIC         fp.DEST_WND_SPEED_QUALITY_CODE,
# MAGIC         fp.DEST_CIG_CEILING_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_CIG_CEILING_QUALITY_CODE,
# MAGIC         fp.DEST_CIG_CEILING_DETERMINATION_CODE,
# MAGIC         fp.DEST_CIG_CAVOK_CODE,
# MAGIC         fp.DEST_VIS_DISTANCE_DIMENSION,
# MAGIC         fp.DEST_VIS_DISTANCE_QUALITY_CODE,
# MAGIC         fp.DEST_VIS_VARIABILITY_CODE,
# MAGIC         fp.DEST_VIS_QUALITY_VARIABILITY_CODE,
# MAGIC         fp.DEST_TMP_AIR_TEMP,
# MAGIC         fp.DEST_TMP_AIR_TEMP_QUALITY_CODE,
# MAGIC         fp.DEST_DEW_POINT_TEMP,
# MAGIC         fp.DEST_DEW_POINT_QUALITY_CODE,
# MAGIC         fp.DEST_SLP_SEA_LEVEL_PRES,
# MAGIC         fp.DEST_SLP_SEA_LEVEL_PRES_QUALITY_CODE,
# MAGIC         fp.DEST_AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AA1_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AA1_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AA2_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AA2_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AA3_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AA3_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AA4_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AA4_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_SNOW_DEPTH_DIMENSION,
# MAGIC         fp.DEST_SNOW_DEPTH_CONDITION_CODE,
# MAGIC         fp.DEST_SNOW_DEPTH_QUALITY_CODE,
# MAGIC         fp.DEST_SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION,
# MAGIC         fp.DEST_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_CODE,
# MAGIC         fp.DEST_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_QUALITY_CODE,
# MAGIC         fp.DEST_AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AL1_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.DEST_AL1_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.DEST_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AL2_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.DEST_AL2_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.DEST_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AL3_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.DEST_AL3_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.DEST_AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AO1_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AO1_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AW1_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.DEST_AW1_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.DEST_AW2_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.DEST_AW2_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.DEST_AW3_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.DEST_AW3_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.DEST_AW4_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.DEST_AW4_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE,
# MAGIC         fp.DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_RUNWAY_DESIGNATOR_CODE,
# MAGIC         fp.DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION,
# MAGIC         fp.DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_CONVECTIVE_CLOUD_ATTRIBUTE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_VERTICAL_DATUM_ATTRIBUTE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_TOTAL_COVERAGE_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_TOTAL_OPAQUE_COVERAGE_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_COVERAGE_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_TOTAL_LOWEST_CLOUD_COVER_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_LOWEST_CLOUD_COVER_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_LOW_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_LOW_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_MID_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_MID_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_HIGH_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_HIGH_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.DEST_GROUND_SURFACE_OBSERVATION_CODE,
# MAGIC         fp.DEST_GROUND_SURFACE_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_WIND_GUST_OBSERVATION_SPEED_RATE,
# MAGIC         fp.DEST_WIND_GUST_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.WND_DIRECTION_ANGLE_3_RA AS ORIGIN_WND_DIRECTION_ANGLE_3_RA,
# MAGIC         fp.WND_SPEED_RATE_3_RA AS ORIGIN_WND_SPEED_RATE_3_RA,
# MAGIC         fp.CIG_CEILING_HEIGHT_DIMENSION_3_RA AS ORIGIN_CIG_CEILING_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.VIS_DISTANCE_DIMENSION_3_RA AS ORIGIN_VIS_DISTANCE_DIMENSION_3_RA,
# MAGIC         fp.TMP_AIR_TEMP_3_RA AS ORIGIN_TMP_AIR_TEMP_3_RA,
# MAGIC         fp.DEW_POINT_TEMP_3_RA AS ORIGIN_DEW_POINT_TEMP_3_RA,
# MAGIC         fp.SLP_SEA_LEVEL_PRES_3_RA AS ORIGIN_SLP_SEA_LEVEL_PRES_3_RA,
# MAGIC         fp.AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS ORIGIN_AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS ORIGIN_AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS ORIGIN_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS ORIGIN_AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS ORIGIN_AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS ORIGIN_AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS ORIGIN_AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS ORIGIN_AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.SNOW_DEPTH_DIMENSION_3_RA AS ORIGIN_SNOW_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION_3_RA AS ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA AS ORIGIN_AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA AS ORIGIN_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA AS ORIGIN_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA AS ORIGIN_AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA AS ORIGIN_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA AS ORIGIN_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS ORIGIN_AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS ORIGIN_AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE_3_RA AS ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE_3_RA,
# MAGIC         fp.RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION_3_RA AS ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION_3_RA,
# MAGIC         fp.GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE_3_RA AS ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE_3_RA,
# MAGIC         fp.SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE_3_RA AS ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE_3_RA,
# MAGIC         fp.SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION_3_RA AS ORIGIN_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA AS ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA AS ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA AS ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA AS ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.WIND_GUST_OBSERVATION_SPEED_RATE_3_RA AS ORIGIN_WIND_GUST_OBSERVATION_SPEED_RATE_3_RA
# MAGIC FROM   flights_and_weather_ra_origin fp 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM flights_and_weather_ra_origin_renamed

# COMMAND ----------

# MAGIC %md 
# MAGIC Add in rolling averages for destination

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE flights_and_weather_ra_dest
# MAGIC USING DELTA 
# MAGIC LOCATION '/airline_delays/$username/DLRS/flights_and_weather_ra_dest/processed' 
# MAGIC AS SELECT * 
# MAGIC FROM flights_and_weather_ra_origin_renamed fp 
# MAGIC LEFT JOIN (SELECT * 
# MAGIC            FROM (SELECT *,
# MAGIC                         ROW_NUMBER() 
# MAGIC                         OVER (PARTITION BY wp.WEATHER_KEY
# MAGIC                              ORDER BY wp.WEATHER_KEY ASC) 
# MAGIC                         AS DEST_ROW_NUM
# MAGIC                  FROM weather_engineered_processed wp) AS w
# MAGIC           WHERE w.DEST_ROW_NUM = 1) AS w1 
# MAGIC ON fp.DEST_WEATHER_KEY = w1.WEATHER_KEY

# COMMAND ----------

# MAGIC %sql CREATE OR REPLACE TABLE flights_and_weather_ra_dest_renamed USING DELTA LOCATION '/airline_delays/$username/DLRS/flights_and_weather_ra_dest_renamed/processed' AS
# MAGIC SELECT  fp.YEAR,
# MAGIC         fp.QUARTER,
# MAGIC         fp.MONTH,
# MAGIC         fp.DAY_OF_WEEK,
# MAGIC         fp.TAIL_NUM,
# MAGIC         fp.ORIGIN,
# MAGIC         fp.ORIGIN_CITY_NAME,
# MAGIC         fp.DEST,
# MAGIC         fp.DEST_CITY_NAME,
# MAGIC         fp.DEP_DELAY,
# MAGIC         fp.DEP_DELAY_NEW,
# MAGIC         fp.DEP_DEL15,
# MAGIC         fp.DEP_DELAY_GROUP,
# MAGIC         fp.DISTANCE,
# MAGIC         fp.FL_DATE,
# MAGIC         fp.ORIGIN_WEATHER_KEY,
# MAGIC         fp.DEST_WEATHER_KEY,
# MAGIC         fp.ORIGIN_WND_DIRECTION_ANGLE,
# MAGIC         fp.ORIGIN_WND_QUALITY_CODE,
# MAGIC         fp.ORIGIN_WND_TYPE_CODE,
# MAGIC         fp.ORIGIN_WND_SPEED_RATE,
# MAGIC         fp.ORIGIN_WND_SPEED_QUALITY_CODE,
# MAGIC         fp.ORIGIN_CIG_CEILING_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_CIG_CEILING_QUALITY_CODE,
# MAGIC         fp.ORIGIN_CIG_CEILING_DETERMINATION_CODE,
# MAGIC         fp.ORIGIN_CIG_CAVOK_CODE,
# MAGIC         fp.ORIGIN_VIS_DISTANCE_DIMENSION,
# MAGIC         fp.ORIGIN_VIS_DISTANCE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_VIS_VARIABILITY_CODE,
# MAGIC         fp.ORIGIN_VIS_QUALITY_VARIABILITY_CODE,
# MAGIC         fp.ORIGIN_TMP_AIR_TEMP,
# MAGIC         fp.ORIGIN_TMP_AIR_TEMP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_DEW_POINT_TEMP,
# MAGIC         fp.ORIGIN_DEW_POINT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_SLP_SEA_LEVEL_PRES,
# MAGIC         fp.ORIGIN_SLP_SEA_LEVEL_PRES_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_CONDITION_CODE,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_QUALITY_CODE,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_CODE,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.ORIGIN_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.ORIGIN_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.ORIGIN_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU1_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU2_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU3_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU4_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.ORIGIN_AU5_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_AW1_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.ORIGIN_AW1_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AW2_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.ORIGIN_AW2_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AW3_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.ORIGIN_AW3_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.ORIGIN_AW4_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.ORIGIN_AW4_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_RUNWAY_DESIGNATOR_CODE,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_CONVECTIVE_CLOUD_ATTRIBUTE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_VERTICAL_DATUM_ATTRIBUTE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_TOTAL_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_TOTAL_OPAQUE_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_TOTAL_LOWEST_CLOUD_COVER_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_LOWEST_CLOUD_COVER_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_LOW_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_LOW_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_MID_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_MID_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_HIGH_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_QUALITY_HIGH_CLOUD_GENUS_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.ORIGIN_GROUND_SURFACE_OBSERVATION_CODE,
# MAGIC         fp.ORIGIN_GROUND_SURFACE_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_WIND_GUST_OBSERVATION_SPEED_RATE,
# MAGIC         fp.ORIGIN_WIND_GUST_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_WND_DIRECTION_ANGLE,
# MAGIC         fp.DEST_WND_QUALITY_CODE,
# MAGIC         fp.DEST_WND_TYPE_CODE,
# MAGIC         fp.DEST_WND_SPEED_RATE,
# MAGIC         fp.DEST_WND_SPEED_QUALITY_CODE,
# MAGIC         fp.DEST_CIG_CEILING_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_CIG_CEILING_QUALITY_CODE,
# MAGIC         fp.DEST_CIG_CEILING_DETERMINATION_CODE,
# MAGIC         fp.DEST_CIG_CAVOK_CODE,
# MAGIC         fp.DEST_VIS_DISTANCE_DIMENSION,
# MAGIC         fp.DEST_VIS_DISTANCE_QUALITY_CODE,
# MAGIC         fp.DEST_VIS_VARIABILITY_CODE,
# MAGIC         fp.DEST_VIS_QUALITY_VARIABILITY_CODE,
# MAGIC         fp.DEST_TMP_AIR_TEMP,
# MAGIC         fp.DEST_TMP_AIR_TEMP_QUALITY_CODE,
# MAGIC         fp.DEST_DEW_POINT_TEMP,
# MAGIC         fp.DEST_DEW_POINT_QUALITY_CODE,
# MAGIC         fp.DEST_SLP_SEA_LEVEL_PRES,
# MAGIC         fp.DEST_SLP_SEA_LEVEL_PRES_QUALITY_CODE,
# MAGIC         fp.DEST_AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AA1_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AA1_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AA2_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AA2_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AA3_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AA3_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AA4_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AA4_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_SNOW_DEPTH_DIMENSION,
# MAGIC         fp.DEST_SNOW_DEPTH_CONDITION_CODE,
# MAGIC         fp.DEST_SNOW_DEPTH_QUALITY_CODE,
# MAGIC         fp.DEST_SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION,
# MAGIC         fp.DEST_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_CODE,
# MAGIC         fp.DEST_SNOW_DEPTH_EQUIVALENT_WATER_CONDITION_QUALITY_CODE,
# MAGIC         fp.DEST_AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AL1_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.DEST_AL1_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.DEST_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AL2_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.DEST_AL2_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.DEST_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AL3_SNOW_ACCUMULATION_CONDITION_CODE,
# MAGIC         fp.DEST_AL3_SNOW_ACCUMULATION_QUALITY_CODE,
# MAGIC         fp.DEST_AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY,
# MAGIC         fp.DEST_AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION,
# MAGIC         fp.DEST_AO1_LIQUID_PRECIPITATION_CONDITION_CODE,
# MAGIC         fp.DEST_AO1_LIQUID_PRECIPITATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT1_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT2_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT3_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT4_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT5_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT6_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT7_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_SOURCE_ELEMENT,
# MAGIC         fp.DEST_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE,
# MAGIC         fp.DEST_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_WEATHER_TYPE_ABBR,
# MAGIC         fp.DEST_AT8_DAILY_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU1_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU2_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU3_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU4_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_INTENSITY_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_DESCRIPTOR_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_PRECIPITATION_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_OBSCURATION_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_OTHER_WEATHER_PHENOMENA_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_COMBINATION_INDICATOR_CODE,
# MAGIC         fp.DEST_AU5_PRESENT_WEATHER_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_AW1_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.DEST_AW1_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.DEST_AW2_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.DEST_AW2_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.DEST_AW3_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.DEST_AW3_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.DEST_AW4_PRESENT_WEATHER_OBSERVATION_AUTOMATED_OCCURRENCE_IDENTIFIER,
# MAGIC         fp.DEST_AW4_PRESENT_WEATHER_OBSERVATION_QUALITY_AUTOMATED_ATMOSPHERIC_CONDITION_CODE,
# MAGIC         fp.DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE,
# MAGIC         fp.DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_RUNWAY_DESIGNATOR_CODE,
# MAGIC         fp.DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION,
# MAGIC         fp.DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA1_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA2_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA3_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA4_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA5_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_CLOUD_TYPE_CODE,
# MAGIC         fp.DEST_GA6_SKY_COVER_LAYER_CLOUD_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD1_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD2_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD3_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD4_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_CODE_2,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GD5_SKY_COVER_SUMMATION_STATE_CHARACTERISTIC_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_CONVECTIVE_CLOUD_ATTRIBUTE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_VERTICAL_DATUM_ATTRIBUTE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_TOTAL_COVERAGE_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_TOTAL_OPAQUE_COVERAGE_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_COVERAGE_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_TOTAL_LOWEST_CLOUD_COVER_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_TOTAL_LOWEST_CLOUD_COVER_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_LOW_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_LOW_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_QUALITY_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_MID_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_MID_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_HIGH_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_SKY_CONDITION_OBSERVATION_QUALITY_HIGH_CLOUD_GENUS_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_COVERAGE_QUALITY_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_QUALITY_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TYPE_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TYPE_QUALITY_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_CODE,
# MAGIC         fp.DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_QUALITY_CODE,
# MAGIC         fp.DEST_GROUND_SURFACE_OBSERVATION_CODE,
# MAGIC         fp.DEST_GROUND_SURFACE_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.DEST_WIND_GUST_OBSERVATION_SPEED_RATE,
# MAGIC         fp.DEST_WIND_GUST_OBSERVATION_QUALITY_CODE,
# MAGIC         fp.ORIGIN_WND_DIRECTION_ANGLE_3_RA,
# MAGIC         fp.ORIGIN_WND_SPEED_RATE_3_RA,
# MAGIC         fp.ORIGIN_CIG_CEILING_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_VIS_DISTANCE_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_TMP_AIR_TEMP_3_RA,
# MAGIC         fp.ORIGIN_DEW_POINT_TEMP_3_RA,
# MAGIC         fp.ORIGIN_SLP_SEA_LEVEL_PRES_3_RA,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.ORIGIN_AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.ORIGIN_AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.ORIGIN_AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.ORIGIN_AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.ORIGIN_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.ORIGIN_AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.ORIGIN_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.ORIGIN_AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE_3_RA,
# MAGIC         fp.ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE_3_RA,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE_3_RA,
# MAGIC         fp.ORIGIN_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.ORIGIN_WIND_GUST_OBSERVATION_SPEED_RATE_3_RA,
# MAGIC         fp.WND_DIRECTION_ANGLE_3_RA AS DEST_WND_DIRECTION_ANGLE_3_RA,
# MAGIC         fp.WND_SPEED_RATE_3_RA AS DEST_WND_SPEED_RATE_3_RA,
# MAGIC         fp.CIG_CEILING_HEIGHT_DIMENSION_3_RA AS DEST_CIG_CEILING_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.VIS_DISTANCE_DIMENSION_3_RA AS DEST_VIS_DISTANCE_DIMENSION_3_RA,
# MAGIC         fp.TMP_AIR_TEMP_3_RA AS DEST_TMP_AIR_TEMP_3_RA,
# MAGIC         fp.DEW_POINT_TEMP_3_RA AS DEST_DEW_POINT_TEMP_3_RA,
# MAGIC         fp.SLP_SEA_LEVEL_PRES_3_RA AS DEST_SLP_SEA_LEVEL_PRES_3_RA,
# MAGIC         fp.AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS DEST_AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS DEST_AA1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS DEST_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS DEST_AA2_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS DEST_AA3_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS DEST_AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS DEST_AA4_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS DEST_AA4_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.SNOW_DEPTH_DIMENSION_3_RA AS DEST_SNOW_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION_3_RA AS DEST_SNOW_DEPTH_EQUIVALENT_WATER_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA AS DEST_AL1_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA AS DEST_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA AS DEST_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA AS DEST_AL2_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA AS DEST_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA AS DEST_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA AS DEST_AO1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA,
# MAGIC         fp.AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA AS DEST_AO1_LIQUID_PRECIPITATION_DEPTH_DIMENSION_3_RA,
# MAGIC         fp.RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE_3_RA AS DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE_3_RA,
# MAGIC         fp.RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION_3_RA AS DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION_3_RA,
# MAGIC         fp.GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS DEST_GA1_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS DEST_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS DEST_GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS DEST_GA4_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS DEST_GA5_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA AS DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS DEST_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS DEST_GD2_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS DEST_GD3_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS DEST_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA AS DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE_3_RA AS DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE_3_RA,
# MAGIC         fp.SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE_3_RA AS DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE_3_RA,
# MAGIC         fp.SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION_3_RA AS DEST_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA AS DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA AS DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA AS DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA AS DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION_3_RA,
# MAGIC         fp.WIND_GUST_OBSERVATION_SPEED_RATE_3_RA AS DEST_WIND_GUST_OBSERVATION_SPEED_RATE_3_RA
# MAGIC FROM   flights_and_weather_ra_dest fp 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM flights_and_weather_ra_dest_renamed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split Data into Train/Test/Validation
# MAGIC TODO: Move to other notebook if we want to keep the train/validation/test feature engineering together

# COMMAND ----------

train_data = spark.sql("SELECT * FROM flights_and_weather_ra_dest_renamed WHERE YEAR IN (2015, 2016, 2017)").drop('YEAR')
validation_data = spark.sql("SELECT * FROM flights_and_weather_ra_dest_renamed WHERE YEAR = 2018").drop('YEAR')
test_data = spark.sql("SELECT * FROM flights_and_weather_ra_dest_renamed WHERE YEAR = 2019").drop('YEAR')

# COMMAND ----------

def make_imputation_dict(df):
  impute_dict = {}
  
  cat_cols = [item[0] for item in df.dtypes if item[1].startswith('string')]  # string
  num_cols = [item[0] for item in df.dtypes if item[1].startswith('int') | item[1].startswith('double')] # will select name of column with integer or double data type
  
  for x in cat_cols:                  
    mode = df.groupBy(x).count().sort(f.col("count").desc()).collect()
    impute_dict[x] = mode[0][0]

  # Fill the missing numerical values with the average of each #column
  for i in num_cols:
    mean_value = df.select(f.mean(i).cast(DoubleType())).collect()
    impute_dict[i] = mean_value[0][0]
    
  return impute_dict

# COMMAND ----------

impute_dict = make_imputation_dict(train_data)

# COMMAND ----------

def impute_missing_values(df, impute_dict):
  missing_count_list = []
  for c in df.columns:
      if df.where(f.col(c).isNull()).count() > 0:
          tup = (c,int(df.where(f.col(c).isNull()).count()))
          missing_count_list.append(tup)

  missing_column_list = [x[0] for x in missing_count_list]
  missing_df = df.select(missing_column_list)

  missing_cat_columns = [item[0] for item in missing_df.dtypes if item[1].startswith('string')]  # string 
  print("\nCategorical Columns with missing data:", missing_cat_columns)

  missing_num_columns = [item[0] for item in missing_df.dtypes if item[1].startswith('int') | item[1].startswith('double')] # will select name of column with integer or double data type
  print("\nNumerical Columns with missing data:", missing_num_columns)
  
  # Fill the missing categorical values with the most frequent category 
  for x in missing_cat_columns:                  
    mode = impute_dict[x]
    if mode:
      df = df.withColumn(x, f.when(df[x].isNull(), f.lit(mode)).otherwise(df[x]))
    else:
      df = df.withColumn(x, f.when(df[x].isNull(), 'None').otherwise(df[x]))

  # Fill the missing numerical values with the average of each #column
  for i in missing_num_columns:
    mean_value = impute_dict[x]
    if mean_value:
        df = df.withColumn(i, f.when(df[i].isNull(), mean_value).otherwise(df[i]))
    else:
        df = df.withColumn(i, f.when(df[i].isNull(), 0).otherwise(df[i]))
        
  return df

# COMMAND ----------

train_data = impute_missing_values(train_data, impute_dict)

# COMMAND ----------

flights_and_weather_train_ra_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_ra_train/"
dbutils.fs.rm(flights_and_weather_train_ra_loc + 'processed', recurse=True)
train_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_train_ra_loc + 'processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_and_weather_train_ra_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_and_weather_train_ra_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_and_weather_ra_train/processed"

# COMMAND ----------

validation_data = impute_missing_values(validation_data, impute_dict)

# COMMAND ----------

flights_and_weather_validation_ra_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_ra_validation/"
dbutils.fs.rm(flights_and_weather_validation_ra_loc + 'processed', recurse=True)
validation_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_validation_ra_loc + 'processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_and_weather_validation_ra_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_and_weather_validation_ra_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_and_weather_ra_validation/processed"

# COMMAND ----------

test_data = impute_missing_values(test_data, impute_dict)

# COMMAND ----------

flights_and_weather_test_ra_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_ra_test/"
dbutils.fs.rm(flights_and_weather_test_ra_loc + 'processed', recurse=True)
validation_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_test_ra_loc + 'processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_and_weather_test_ra_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_and_weather_test_ra_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_and_weather_ra_test/processed"

# COMMAND ----------

# MAGIC %md # Citations

# COMMAND ----------

# MAGIC %md Abdulwahab. Aljubairy, A., L. Atzori, A., L. Belcastro, F., Y. Chen, J., NR. Chopde, M., D. Georgakopoulos, P., . . . W. Wu, C. (1970, January 01). A system for effectively predicting flight delays based on IoT data. Retrieved July 25, 2020, from https://link.springer.com/article/10.1007/s00607-020-00794-w  
# MAGIC Ye, B., Liu, B., Tian, Y., &amp; Wan, L. (2020, April 1). A Methodology for Predicting Aggregate Flight Departure ... Retrieved July 25, 2020, from https://www.mdpi.com/2071-1050/12/7/2749/pdf

# COMMAND ----------

