# Databricks notebook source
# MAGIC %md # W261 Final Project - Airline Delays and Weather
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Summer 2020`__
# MAGIC ### Team 1:
# MAGIC * Sayan Das
# MAGIC * Kevin Hartman
# MAGIC * Hersh Solanki
# MAGIC * Nick Sylva

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Imports

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from pytz import timezone 
from datetime import  datetime, timedelta 
import os

sqlContext = SQLContext(sc)

from delta.tables import DeltaTable

# COMMAND ----------

# MAGIC %md ### Configure access to staging areas - bronze & silver

# COMMAND ----------

username = "kevin"
dbutils.widgets.text("username", username)
spark.sql(f"CREATE DATABASE IF NOT EXISTS airline_delays_{username}")
spark.sql(f"USE airline_delays_{username}")

flights_loc = f"/airline_delays/{username}/DLRS/flights/"
flights_3m_loc = f"/airline_delays/{username}/DLRS/flights_3m/"
flights_6m_loc = f"/airline_delays/{username}/DLRS/flights_6m/"
airports_loc = f"/airline_delays/{username}/DLRS/airports/"
weather_loc = f"/airline_delays/{username}/DLRS/weather/"
stations_loc = f"/airline_delays/{username}/DLRS/stations/"
flights_and_weather_loc = f"/airline_delays/{username}/DLRS/flights_and_weather/"

spark.conf.set("spark.sql.shuffle.partitions", 8)

# COMMAND ----------

# MAGIC %md # Table of Contents
# MAGIC ### 1. Introduction
# MAGIC ### 2. Data Sources
# MAGIC ### 3. Question Formulation
# MAGIC ### 4. Data Lake Prep
# MAGIC ### 5. EDA
# MAGIC ### 6. Feature Engineering
# MAGIC ### 7. Algorithm Exploration
# MAGIC ### 8. Algorithm Implementation
# MAGIC ### 9. Conclusion
# MAGIC ### (10. Application of Course Concepts)

# COMMAND ----------

# MAGIC %md # 1. Introduction
# MAGIC TBD

# COMMAND ----------

# MAGIC %md # 2. Data Sources
# MAGIC 
# MAGIC ## Airline delays 
# MAGIC ### Bureau of Transportation Statistics
# MAGIC https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236   
# MAGIC https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC 
# MAGIC Dates covered in dataset: 2015 - 2019

# COMMAND ----------

# MAGIC %md ### Additional sources
# MAGIC This might be useful in matching station codes to airports:
# MAGIC 1. http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html
# MAGIC 2. https://www.world-airport-codes.com/

# COMMAND ----------

# MAGIC %md
# MAGIC # WARNING: DO NOT RUN CODE IN THE NEXT SECTION UNLESS YOU NEED TO RECONSTRUCT THE BRONZE AND SILVER  DATA LAKES
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Please start execution from Section 4. EDA to load from already processed (silver) data

# COMMAND ----------

# MAGIC %md # 3. Data Lake Prep
# MAGIC ### Download and store data locally (Bronze)
# MAGIC #### Clear raw (bronze) landing zone

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

# MAGIC %md
# MAGIC #### Flights, Weather and Stations
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

# MAGIC %md
# MAGIC #### Airports
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

# MAGIC %md
# MAGIC #### Ingest data from staging or source zone (wherever data currently resides) and place into the bronze zone (in raw format)

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

# MAGIC %md
# MAGIC #### Re-read raw files as delta lake

# COMMAND ----------

flights_raw_df = spark.read.format("delta").load(flights_loc + "raw")
flights_3m_raw_df = spark.read.format("delta").load(flights_3m_loc + "raw")
flights_6m_raw_df = spark.read.format("delta").load(flights_6m_loc + "raw")
weather_raw_df = spark.read.format("delta").load(weather_loc + "raw")
stations_raw_df = spark.read.format("delta").load(stations_loc + "raw")
airports_raw_df = spark.read.format("delta").load(airports_loc + "raw")

# COMMAND ----------

flights_raw_df = spark.read.format("delta").load(flights_loc + "raw")

# COMMAND ----------

flights_raw_df.count()

# COMMAND ----------

airports_raw_df = spark.read.format("delta").load(airports_loc + "raw")

# COMMAND ----------

# MAGIC %md ## Perform data processing for analysis (Silver)
# MAGIC #### Clear processed staging folders

# COMMAND ----------

dbutils.fs.rm(flights_loc + "processed", recurse=True)

# COMMAND ----------

dbutils.fs.rm(weather_loc + "processed", recurse=True)

# COMMAND ----------

dbutils.fs.rm(flights_loc + "processed", recurse=True)
dbutils.fs.rm(flights_3m_loc + "processed", recurse=True)
dbutils.fs.rm(flights_6m_loc + "processed", recurse=True)
dbutils.fs.rm(airports_loc + "processed", recurse=True)
dbutils.fs.rm(weather_loc + "processed", recurse=True)
dbutils.fs.rm(stations_loc + "processed", recurse=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Flight data pre-processing

# COMMAND ----------

def process_flight_data(df):
  cols = df.columns
  cols.append('IN_FLIGHT_AIR_DELAY')
  return (
    df
    .withColumn('IN_FLIGHT_AIR_DELAY', f.lit(df['ARR_DELAY'] - df['DEP_DELAY'] )) # this column is the time difference between arrival and departure and does not include total flight delay
#    .withColumn("time", from_unixtime("time"))
#    .withColumnRenamed("device_id", "p_device_id")
#    .withColumn("time", col("time").cast("timestamp"))
#    .withColumn("dte", col("time").cast("date"))
#    .withColumn("p_device_id", col("p_device_id").cast("integer"))
#    .select("dte", "time", "heartrate", "name", "p_device_id")
    .select(cols)
    )

flights_processed_df = process_flight_data(flights_raw_df)
#flights_3m_processed_df = process_flight_data(flights_3m_raw_df)
#flights_6m_processed_df = process_flight_data(flights_6m_raw_df)

# COMMAND ----------

flights_processed_df.count()

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

# MAGIC  %md
# MAGIC  ### Weather data pre-processing
# MAGIC  

# COMMAND ----------

def process_weather_data(df):
  WND_col = f.split(df['WND'], ',')
  CIG_col = f.split(df['CIG'], ',')
  VIS_col = f.split(df['VIS'], ',')
  TMP_col = f.split(df['TMP'], ',')
  DEW_col = f.split(df['DEW'], ',')
  SLP_col = f.split(df['SLP'], ',')
  df = (df
    .withColumn("STATION", f.lpad(df.STATION, 11, '0'))
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
    
    .withColumnRenamed("DATE", "WEATHER_DATE")
    .withColumnRenamed("SOURCE", "WEATHER_SOURCE")
    .withColumnRenamed("STATION", "WEATHER_STATION")
       )


  cols = set(df.columns)
  remove_cols = set(['LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'WND', 'CIG','VIS','TMP', 'DEW', 'SLP'])
  cols = list(cols - remove_cols)
  df = df.select(cols)
  for col in df.columns:
    df = df.withColumnRenamed(col,f'WEATHER1_{col}')
  return df
  

weather_processed_df = process_weather_data(weather_raw_df)

# COMMAND ----------

(weather_processed_df.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("WEATHER1_WND_DIRECTION_ANGLE")
 .save(weather_loc + "processed"))

parquet_table = f"parquet.`{weather_loc}processed`"
partitioning_scheme = "WEATHER1_WND_DIRECTION_ANGLE string"

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

# MAGIC %md 
# MAGIC ### Station data pre-processing

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

# MAGIC  %md
# MAGIC ### Airport data pre-processing

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

# MAGIC %md # 4. EDA
# MAGIC 
# MAGIC #### Load data from processed (silver) staging area

# COMMAND ----------

# MAGIC %md
# MAGIC ## Flights EDA
# MAGIC 
# MAGIC Schema for flights: https://annettegreiner.com/vizcomm/OnTime_readme.html

# COMMAND ----------

flights_processed = spark.read.table("flights_processed")

# COMMAND ----------

flights_processed.printSchema()

# COMMAND ----------

f'{flights_processed.count():,}'

# COMMAND ----------

# we can filter out the first three months ourselves
#flights_sample = flights_processed.where('(ORIGIN = "ORD" OR ORIGIN = "ATL") AND QUARTER = 1 and YEAR = 2015').sample(False, .10, seed = 42)
# Or the code below results in the equivalent
flights_sample = flights_3m_processed.sample(False, .10, seed = 42)

# COMMAND ----------

display(flights_sample)

# COMMAND ----------

flights_sample.count()

# COMMAND ----------

flights_sample.dtypes

# COMMAND ----------

def get_dtype(df,colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]

# COMMAND ----------

get_dtype(flights_sample, 'ORIGIN')

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
        fig, ax = plt.subplots(nrows=round(len(self.df.columns)), ncols=2, figsize=(16,3*round(len(self.df.columns))))
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

# analyzer = Analyze(flights_sample)
# analyzer.print_eda_summary()

# COMMAND ----------



# COMMAND ----------

sns.set(rc={'figure.figsize':(100,100)})
sns.heatmap(flights_sample.toPandas().corr(), cmap='RdBu_r', annot=True, center=0.0)
sns.set(rc={'figure.figsize':(10,10)})

# COMMAND ----------

flights_sample.where('DEP_DELAY < 0').count() / flights_sample.count() # This statistic explains that 47% of flights depart earlier

# COMMAND ----------

flights_sample.where('DEP_DELAY == 0').count() / flights_sample.count()  # This statistic explains that 6.9% of flights depart EXACTLY on time

# COMMAND ----------

# MAGIC %md ### The cells below display histograms that analyze departures that are on time or early

# COMMAND ----------

bins, counts = flights_sample.select('DEP_DELAY').where('DEP_DELAY <= 0').rdd.flatMap(lambda x: x).histogram(100)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = flights_df_sample.select('DEP_DELAY').where('DEP_DELAY <= 0 AND DEP_DELAY > -25').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md ### The cells below are displaying histograms that analyze departures that are delayed

# COMMAND ----------

bins, counts = flights_sample.select('DEP_DELAY').where('DEP_DELAY > 0').rdd.flatMap(lambda x: x).histogram(100)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = flights_df_sample.select('DEP_DELAY').where('DEP_DELAY > 0 AND DEP_DELAY < 300').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = flights_sample.select('DEP_DELAY').where('DEP_DELAY > -25 AND DEP_DELAY < 50').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md **Analyzing the plot above, it is apparent that the distribution is right-skewed, implying that there is a heavy amount of data that is delayed and shifting the distribution towards the right, so therefore the median departure delay is higher than the mean.  Intuitively, this makes sense, for it is more likely that a flight will depart a day later compared to a flight departing a day earlier.  Moreover, we can see that much of the data revolves around flights that depart early or on time, and it is possible that the data is from airports that are smaller with less load; this would explain how the flights would be more likely to depart at an earlier time.  Further analysis of the locations of the actual airports and the distribution of these airports is necessary.**

# COMMAND ----------

# MAGIC %md ### Next, we will look into visualizing arrival delay.  However, we should note that arrival delay also encompasses any delay from the departure delay.  Therefore, we must first ensure that we create a new column that accounts for this discrepancy.

# COMMAND ----------

bins, counts = flights_sample.select('IN_FLIGHT_AIR_DELAY').where('IN_FLIGHT_AIR_DELAY > -50 AND IN_FLIGHT_AIR_DELAY < 50').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md **We can see that there is a normal distribution that is centered around -5; this indicates that the flight makes up 5 minutes of time after departing from the airport.  In general, this is implying that flights are making up time in the air time.  Further analysis should look into analyzing the amount of time made up in the air based on distance to see if flights make up more delay time with longer flight distances.**

# COMMAND ----------

# MAGIC %md # Weather EDA
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532
# MAGIC 
# MAGIC 
# MAGIC Schema for weather: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf

# COMMAND ----------

f'{weather_processed.count():,}'

# COMMAND ----------

weather_processed.printSchema()

# COMMAND ----------

display(weather_processed)

# COMMAND ----------

# subset to 1Q2015
weather_subset = weather_processed.where('WEATHER_DATE >= TO_DATE("01/01/2015", "MM/dd/yyyy") AND WEATHER_DATE <= TO_DATE("03/31/2015", "MM/dd/yyyy")') 

# COMMAND ----------

# MAGIC %md **Now we will plot our data to get a visual representation of these flattened points**

# COMMAND ----------

# create histogram of wind speed. Filtered to remove nulls and values higher than the world record of 253 mph (113 m/s)
bins, counts = weather_subset.where('WND_Speed_Rate <= 113').select('WND_Speed_Rate').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# create histogram of VIS distance code
bins, counts = weather_subset.where('VIS_Distance_Dimension < 999999').select('VIS_Distance_Dimension').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# create histogram of SLP level code
bins, counts = weather_subset.where('SLP_Sea_Level_Pres < 99999').select('SLP_Sea_Level_Pres').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# create histogram of DEW field code
bins, counts = weather_subset.where('DEW_Point_Temp < 9999').select('DEW_Point_Temp').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# create histogram of Air Temp code
bins, counts = weather_subset.where('TMP_Air_Temp < 9999').select('TMP_Air_Temp').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Joining Weather to Station Data
# MAGIC The most efficient way to do this will be to first identify the station associated with each airport, add a column for that to the flights data, and then join directly flights to weather. Note this will require a composite key because we care about both **time** and **location**. Note the cell after the initial join where the joined table is displayed with a filter will take a long time to load.

# COMMAND ----------

# MAGIC %md
# MAGIC Before we join the weather and station data, it is important to make sure each airport in our flight data is represented in our airports file. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT(ORIGIN)
# MAGIC FROM flights_processed
# MAGIC WHERE ORIGIN NOT IN (SELECT IATA FROM airports_processed);

# COMMAND ----------

# MAGIC %md
# MAGIC These airports were missing from the airport file. Let's add them to our silver repository. 
# MAGIC 
# MAGIC Tokeen, Kearney, Bullhead City and Williston

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO airports_processed SELECT t.* FROM (SELECT 9999,"Tokeen Seaplane Base", "Tokeen", "United States", "TKI", "57A", "55.937222", "-133.326667", 0, -8, "A", "airport", "Internet", "America/Metlakatla" ) t;
# MAGIC INSERT INTO airports_processed SELECT t.* FROM (SELECT 9998,"Kearney Regional Airport", "Kearney", "United States", "EAR", "KEAR", "40.7270012", "-99.0067978", 2133, -5, "A", "airport", "Internet", "America/Chicago" ) t;
# MAGIC INSERT INTO airports_processed SELECT t.* FROM (SELECT 9997,"Laughlin Bullhead International Airport", "Bullhead City", "United States", "IFP", "KIFP", "35.1573982", "-114.5599976", 695, -7, "A", "airport", "Internet", "America/Phoenix" ) t;
# MAGIC INSERT INTO airports_processed SELECT t.* FROM (SELECT 9996,"Williston Basin International Airport", "Williston", "United States", "XWA", "KXWA", "48.1778984", "-103.6419983", 1982, -5, "A", "airport", "Internet", "America/Chicago" ) t;

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have added the missing airports, let's check out the schema.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM airports_processed
# MAGIC LIMIT 1;

# COMMAND ----------

# MAGIC %md ### Get Distinct Stations From Weather Data
# MAGIC This ensures that we only use stations that are valid for our analysis period.

# COMMAND ----------

weather_processed = spark.read.table("weather_processed")

# COMMAND ----------

#create set of distinct ids from weather data
weather_distinct_ids = weather_processed.select('WEATHER1_WEATHER_STATION').distinct()

#join distinct ids to stations tables and subset for matches
valid_stations = weather_distinct_ids.join(stations_processed,\
                                           weather_distinct_ids.WEATHER1_WEATHER_STATION == stations_processed.STATION_USAF_WBAN,\
                                           'left').where('STATION_USAF_WBAN IS NOT NULL')


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT FL_DATE, COUNT(FL_DATE) 
# MAGIC FROM flights_processed
# MAGIC GROUP BY FL_DATE
# MAGIC ORDER BY FL_DATE DESC
# MAGIC LIMIT 3;

# COMMAND ----------

# MAGIC %md 
# MAGIC * Query the flights to get the minimum and maximum date. 
# MAGIC   * Per next two cells, flight data covers 1/1/2015-12/31/2019
# MAGIC * Query the stations to look at the start and end date to verify they are all active for the reporting period we are concerned with.

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

# MAGIC %md 
# MAGIC The stations table itself only has stations that show as active through March 2019, but is this reflected in the weather data?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT WEATHER_DATE
# MAGIC FROM weather_processed
# MAGIC ORDER BY WEATHER_DATE DESC
# MAGIC LIMIT 1;

# COMMAND ----------

# MAGIC %md
# MAGIC Weather data are collected through the end of 2019, so the ending date in the stations table is not reliable to use. Let's take a look at our valid stations.

# COMMAND ----------

display(valid_stations)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Nearest Station to Each Airport
# MAGIC Now that we have only the stations that are valid for our analysis, we can find the nearest one to each airport.

# COMMAND ----------

# MAGIC %md
# MAGIC Define a function for the Haversine distance. This is not perfect because it does not consider the projection used when determining the latitude and longitude in the stations and airport data. However, that information is unknown, so the Haversine distance should be a reasonable approximation.

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

# MAGIC %md
# MAGIC Spark job for finding the closest stations. There is probably a better way to do this without converting stuff to RDDs, but this will work.

# COMMAND ----------

def find_closest_station(airports,stations):
    '''
    airports: rdd
    stations: rdd
    '''
    
    stations = sc.broadcast(stations.collect())
    
    def calc_distances(airport):
        airport_list = list(airport)
        airport_lon = float(airport_list[7])
        airport_lat = float(airport_list[6])

        for station in stations.value:
            station_list = list(station)
            if not station_list[7] or not station_list[6]:
                continue
            station_lon = float(station_list[7])
            station_lat = float(station_list[6])
            station_id = station_list[1]
            yield (airport[4], (station_id, haversine(airport_lon, airport_lat, station_lon, station_lat)))
  
  
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

closest_stations = find_closest_station(airports_rdd,stations_rdd).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Need to look at Honolulu

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

airports_stations = sqlContext.createDataFrame(closest_stations)
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

flights_processed = spark.read.format('delta').load(f'{flights_loc}processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM flights_processed
# MAGIC LIMIT 1;

# COMMAND ----------

joined_flights_stations = flights_processed.join(f.broadcast(airports_stations_origin), flights_processed.ORIGIN == airports_stations_origin.IATA_ORIGIN, 'left')
joined_flights_stations = joined_flights_stations.join(f.broadcast(airports_stations_dest), joined_flights_stations.DEST == airports_stations_dest.IATA_DEST, 'left')

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have our nearest weather stations for each flight, we need to create composite keys based on the time of the flight. That will require flight time adjustment to UTC because flight times are in the local time zone. To do that, we join a subset of the airport table to our joined flights and stations table.

# COMMAND ----------

#subset airports
airports_tz = airports_processed.select(['IATA', 'AIRPORT_TZ_NAME'])

#join flights with stations to airport_tz subset on the origin airport because only the departure time needs the UTC adjustment
joined_flights_stations_tz = joined_flights_stations.join(airports_tz, joined_flights_stations.ORIGIN == airports_tz.IATA, 'left')

# COMMAND ----------

# MAGIC %md 
# MAGIC Before continuing, we will store this data into our Silver Delta Lake.

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

# MAGIC %md
# MAGIC ### Create Composite Keys in Flight Data

# COMMAND ----------

flights_processed = spark.read.table("flights_processed")

# COMMAND ----------

flights_processed.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Define UDFs for creating composite keys in the flights data

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

# MAGIC %md 
# MAGIC Create composite keys on flights data

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

# MAGIC %md
# MAGIC Now we have composite keys to join the weather to these data. But first, lets land this in the delta lake

# COMMAND ----------

flights_processed.count()

# COMMAND ----------

flights_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{flights_loc}processed')

# COMMAND ----------

# MAGIC %md
# MAGIC Honolulu is missing a lot of weather data so we will replace its weather key with that of Hilo
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
# MAGIC SELECT COUNT(*)
# MAGIC FROM flights_processed

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE flights_processed
# MAGIC SET nearest_station_id_ORIGIN = replace_HNL_weather_station_id(nearest_station_id_ORIGIN), ORIGIN_WEATHER_KEY = fix_HNL_weather_key(ORIGIN_WEATHER_KEY)
# MAGIC WHERE ORIGIN = "HNL"

# COMMAND ----------

flights_processed = spark.read.table("flights_processed")

# COMMAND ----------

display(flights_processed)

# COMMAND ----------

flights_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{flights_loc}processed')

# COMMAND ----------

# MAGIC %md ### Create Composite Keys in Weather Data

# COMMAND ----------

# MAGIC %md
# MAGIC First define a UDF

# COMMAND ----------

weather_processed = spark.read.table("weather_processed")

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM weather_processed
# MAGIC LIMIT 1

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

# MAGIC %md
# MAGIC Remove weather data with no station ID and land in Delta Lake

# COMMAND ----------

weather_processed.where("WEATHER1_WEATHER_STATION IS NOT NULL").write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{weather_loc}processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM weather_processed
# MAGIC WHERE create_composite_weather_key(WEATHER1_WEATHER_DATE, WEATHER1_WEATHER_STATION) IS NULL;

# COMMAND ----------

weather_processed = spark.sql("SELECT *, create_composite_weather_key(WEATHER1_WEATHER_DATE, WEATHER1_WEATHER_STATION) AS WEATHER1_WEATHER_KEY FROM weather_processed;")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM weather_processed
# MAGIC LIMIT 1

# COMMAND ----------

display(weather_processed.limit(1))

# COMMAND ----------

weather_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{weather_loc}processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM weather_processed
# MAGIC LIMIT 1

# COMMAND ----------

# MAGIC %md ## Join Weather to Flights

# COMMAND ----------

weather_processed = spark.read.table("weather_processed")

# COMMAND ----------

flights_processed.count()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM flights_processed
# MAGIC LEFT JOIN weather_processed
# MAGIC ON flights_processed.ORIGIN_WEATHER_KEY=weather_processed.WEATHER_KEY
# MAGIC LIMIT 10;

# COMMAND ----------

# join the origin weather
flights_processed = spark.sql("SELECT * FROM flights_processed LEFT SEMI JOIN weather_processed ON flights_processed.ORIGIN_WEATHER_KEY=weather_processed.WEATHER1_WEATHER_KEY;")

# COMMAND ----------

# save results of first join to Delta Lake
flights_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{flights_loc}processed')

# COMMAND ----------

flights_processed_test = spark.sql("SELECT * FROM flights_processed_rolled_back SEMI JOIN weather_processed ON flights_processed.ORIGIN_WEATHER_KEY = weather_processed.WEATHER1_WEATHER_KEY")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM (
# MAGIC     SELECT w.*, 
# MAGIC     ROW_NUMBER() OVER (ORDER BY WEATHER1_WEATHER_KEY DESC) rn 
# MAGIC     FROM weather_processed w
# MAGIC )
# MAGIC WHERE rn = 1

# COMMAND ----------

flights_and_weather_processed = spark.sql("SELECT * FROM flights_processed VERSION AS OF 3 fp JOIN ( \
                                                          SELECT * FROM (\
                                                                SELECT *, ROW_NUMBER() OVER ( \
                                                                      partition by WEATHER1_WEATHER_KEY \
                                                                      ORDER BY WEATHER1_WEATHER_DATE ASC \
                                                                ) as row_num \
                                                                FROM weather_processed \
                                                            ) as ordered_weather \
                                                            WHERE ordered_weather.row_num = 1 \
                                                        ) as w1 \
                                                        ON fp.ORIGIN_WEATHER_KEY = w1.WEATHER1_WEATHER_KEY")

# COMMAND ----------



# COMMAND ----------

weather_processed_first_records_only = spark.sql("SELECT most_recent_weather.* FROM weather_processed JOIN ( \
                                                          SELECT * FROM (\
                                                                SELECT ROW_NUMBER() OVER ( \
                                                                      partition by WEATHER1_WEATHER_KEY \
                                                                      ORDER BY WEATHER1_WEATHER_DATE ASC \
                                                                ) as row_num \
                                                                FROM weather_processed \
                                                            ) as ordered_weather \
                                                            WHERE ordered_weather.row_num = 1 \
                                                        ) as most_recent_weather \
                                                        ON weather_processed.WEATHER1_WEATHER_KEY = most_recent_weather.WEATHER1_WEATHER_KEY")

# COMMAND ----------

# save results to the weather  Delta Lake
weather_processed_first_records_only.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{weather_loc}processed')

# COMMAND ----------

flights_and_weather_loc = f"/airline_delays/{username}/DLRS/flights_and_weather/"

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE flights_and_weather_processed
# MAGIC USING DELTA
# MAGIC LOCATION '/airline_delays/kevin/DLRS/flights_and_weather'
# MAGIC AS SELECT *
# MAGIC FROM
# MAGIC   flights_processed AS OF VERSION 3 fp
# MAGIC   LEFT JOIN weather_processed w1 ON fp.ORIGIN_WEATHER_KEY = w1.WEATHER1_WEATHER_KEY
# MAGIC   LEFT JOIN (
# MAGIC     SELECT
# MAGIC       w.WEATHER1_GN1 AS WEATHER2_GN1,
# MAGIC       w.WEATHER1_GF1 AS WEATHER2_GF1,
# MAGIC       w.WEATHER1_UA1 AS WEATHER2_UA1,
# MAGIC       w.WEATHER1_AU2 AS WEATHER2_AU2,
# MAGIC       w.WEATHER1_AX5 AS WEATHER2_AX5,
# MAGIC       w.WEATHER1_TMP_AIR_TEMP AS WEATHER2_TMP_AIR_TEMP,
# MAGIC       w.WEATHER1_MK1 AS WEATHER2_MK1,
# MAGIC       w.WEATHER1_CN3 AS WEATHER2_CN3,
# MAGIC       w.WEATHER1_GM1 AS WEATHER2_GM1,
# MAGIC       w.WEATHER1_GA2 AS WEATHER2_GA2,
# MAGIC       w.WEATHER1_SLP_SEA_LEVEL_PRES_QUALITY_CODE AS WEATHER2_SLP_SEA_LEVEL_PRES_QUALITY_CODE,
# MAGIC       w.WEATHER1_AW7 AS WEATHER2_AW7,
# MAGIC       w.WEATHER1_MG1 AS WEATHER2_MG1,
# MAGIC       w.WEATHER1_CG3 AS WEATHER2_CG3,
# MAGIC       w.WEATHER1_VIS_VARIABILITY_CODE AS WEATHER2_VIS_VARIABILITY_CODE,
# MAGIC       w.WEATHER1_GO1 AS WEATHER2_GO1,
# MAGIC       w.WEATHER1_AL3 AS WEATHER2_AL3,
# MAGIC       w.WEATHER1_AI1 AS WEATHER2_AI1,
# MAGIC       w.WEATHER1_MF1 AS WEATHER2_MF1,
# MAGIC       w.WEATHER1_WEATHER_KEY AS WEATHER2_WEATHER_KEY,
# MAGIC       w.WEATHER1_KA3 AS WEATHER2_KA3,
# MAGIC       w.WEATHER1_AI4 AS WEATHER2_AI4,
# MAGIC       w.WEATHER1_AK1 AS WEATHER2_AK1,
# MAGIC       w.WEATHER1_OE3 AS WEATHER2_OE3,
# MAGIC       w.WEATHER1_AW2 AS WEATHER2_AW2,
# MAGIC       w.WEATHER1_REM AS WEATHER2_REM,
# MAGIC       w.WEATHER1_OD2 AS WEATHER2_OD2,
# MAGIC       w.WEATHER1_CN4 AS WEATHER2_CN4,
# MAGIC       w.WEATHER1_AO1 AS WEATHER2_AO1,
# MAGIC       w.WEATHER1_CO1 AS WEATHER2_CO1,
# MAGIC       w.WEATHER1_OE2 AS WEATHER2_OE2,
# MAGIC       w.WEATHER1_CG1 AS WEATHER2_CG1,
# MAGIC       w.WEATHER1_AX6 AS WEATHER2_AX6,
# MAGIC       w.WEATHER1_KA2 AS WEATHER2_KA2,
# MAGIC       w.WEATHER1_CU2 AS WEATHER2_CU2,
# MAGIC       w.WEATHER1_AH3 AS WEATHER2_AH3,
# MAGIC       w.WEATHER1_OE1 AS WEATHER2_OE1,
# MAGIC       w.WEATHER1_MA1 AS WEATHER2_MA1,
# MAGIC       w.WEATHER1_CT1 AS WEATHER2_CT1,
# MAGIC       w.WEATHER1_AW4 AS WEATHER2_AW4,
# MAGIC       w.WEATHER1_AU3 AS WEATHER2_AU3,
# MAGIC       w.WEATHER1_GA5 AS WEATHER2_GA5,
# MAGIC       w.WEATHER1_UG1 AS WEATHER2_UG1,
# MAGIC       w.WEATHER1_GE1 AS WEATHER2_GE1,
# MAGIC       w.WEATHER1_AI3 AS WEATHER2_AI3,
# MAGIC       w.WEATHER1_GD1 AS WEATHER2_GD1,
# MAGIC       w.WEATHER1_CIG_CEILING_QUALITY_CODE AS WEATHER2_CIG_CEILING_QUALITY_CODE,
# MAGIC       w.WEATHER1_AX3 AS WEATHER2_AX3,
# MAGIC       w.WEATHER1_KD1 AS WEATHER2_KD1,
# MAGIC       w.WEATHER1_CIG_CEILING_HEIGHT_DIMENSION AS WEATHER2_CIG_CEILING_HEIGHT_DIMENSION,
# MAGIC       w.WEATHER1_AW3 AS WEATHER2_AW3,
# MAGIC       w.WEATHER1_CV1 AS WEATHER2_CV1,
# MAGIC       w.WEATHER1_AY1 AS WEATHER2_AY1,
# MAGIC       w.WEATHER1_MV1 AS WEATHER2_MV1,
# MAGIC       w.WEATHER1_KA4 AS WEATHER2_KA4,
# MAGIC       w.WEATHER1_AJ1 AS WEATHER2_AJ1,
# MAGIC       w.WEATHER1_WEATHER_DATE AS WEATHER2_WEATHER_DATE,
# MAGIC       w.WEATHER1_CG2 AS WEATHER2_CG2,
# MAGIC       w.WEATHER1_KD2 AS WEATHER2_KD2,
# MAGIC       w.WEATHER1_DEW_POINT_TEMP AS WEATHER2_DEW_POINT_TEMP,
# MAGIC       w.WEATHER1_AH1 AS WEATHER2_AH1,
# MAGIC       w.WEATHER1_AU1 AS WEATHER2_AU1,
# MAGIC       w.WEATHER1_GL1 AS WEATHER2_GL1,
# MAGIC       w.WEATHER1_MW5 AS WEATHER2_MW5,
# MAGIC       w.WEATHER1_AU4 AS WEATHER2_AU4,
# MAGIC       w.WEATHER1_AT4 AS WEATHER2_AT4,
# MAGIC       w.WEATHER1_CN2 AS WEATHER2_CN2,
# MAGIC       w.WEATHER1_CV3 AS WEATHER2_CV3,
# MAGIC       w.WEATHER1_MW2 AS WEATHER2_MW2,
# MAGIC       w.WEATHER1_AT7 AS WEATHER2_AT7,
# MAGIC       w.WEATHER1_ED1 AS WEATHER2_ED1,
# MAGIC       w.WEATHER1_CU3 AS WEATHER2_CU3,
# MAGIC       w.WEATHER1_UG2 AS WEATHER2_UG2,
# MAGIC       w.WEATHER1_WD1 AS WEATHER2_WD1,
# MAGIC       w.WEATHER1_RH3 AS WEATHER2_RH3,
# MAGIC       w.WEATHER1_AT5 AS WEATHER2_AT5,
# MAGIC       w.WEATHER1_MW1 AS WEATHER2_MW1,
# MAGIC       w.WEATHER1_GK1 AS WEATHER2_GK1,
# MAGIC       w.WEATHER1_CU1 AS WEATHER2_CU1,
# MAGIC       w.WEATHER1_AA3 AS WEATHER2_AA3,
# MAGIC       w.WEATHER1_AM1 AS WEATHER2_AM1,
# MAGIC       w.WEATHER1_OD3 AS WEATHER2_OD3,
# MAGIC       w.WEATHER1_GA1 AS WEATHER2_GA1,
# MAGIC       w.WEATHER1_AT2 AS WEATHER2_AT2,
# MAGIC       w.WEATHER1_AI6 AS WEATHER2_AI6,
# MAGIC       w.WEATHER1_AT8 AS WEATHER2_AT8,
# MAGIC       w.WEATHER1_RH1 AS WEATHER2_RH1,
# MAGIC       w.WEATHER1_MW4 AS WEATHER2_MW4,
# MAGIC       w.WEATHER1_AA1 AS WEATHER2_AA1,
# MAGIC       w.WEATHER1_AN1 AS WEATHER2_AN1,
# MAGIC       w.WEATHER1_AH6 AS WEATHER2_AH6,
# MAGIC       w.WEATHER1_CF3 AS WEATHER2_CF3,
# MAGIC       w.WEATHER1_CF1 AS WEATHER2_CF1,
# MAGIC       w.WEATHER1_CIG_CEILING_DETERMINATION_CODE AS WEATHER2_CIG_CEILING_DETERMINATION_CODE,
# MAGIC       w.WEATHER1_AI2 AS WEATHER2_AI2,
# MAGIC       w.WEATHER1_CN1 AS WEATHER2_CN1,
# MAGIC       w.WEATHER1_CT3 AS WEATHER2_CT3,
# MAGIC       w.WEATHER1_GG4 AS WEATHER2_GG4,
# MAGIC       w.WEATHER1_CB1 AS WEATHER2_CB1,
# MAGIC       w.WEATHER1_GD2 AS WEATHER2_GD2,
# MAGIC       w.WEATHER1_GA4 AS WEATHER2_GA4,
# MAGIC       w.WEATHER1_KA1 AS WEATHER2_KA1,
# MAGIC       w.WEATHER1_AD1 AS WEATHER2_AD1,
# MAGIC       w.WEATHER1_WA1 AS WEATHER2_WA1,
# MAGIC       w.WEATHER1_AW1 AS WEATHER2_AW1,
# MAGIC       w.WEATHER1_MH1 AS WEATHER2_MH1,
# MAGIC       w.WEATHER1_KB2 AS WEATHER2_KB2,
# MAGIC       w.WEATHER1_KG1 AS WEATHER2_KG1,
# MAGIC       w.WEATHER1_AA2 AS WEATHER2_AA2,
# MAGIC       w.WEATHER1_GG2 AS WEATHER2_GG2,
# MAGIC       w.WEATHER1_KC1 AS WEATHER2_KC1,
# MAGIC       w.WEATHER1_OC1 AS WEATHER2_OC1,
# MAGIC       w.WEATHER1_IA1 AS WEATHER2_IA1,
# MAGIC       w.WEATHER1_AW5 AS WEATHER2_AW5,
# MAGIC       w.WEATHER1_WND_DIRECTION_ANGLE AS WEATHER2_WND_DIRECTION_ANGLE,
# MAGIC       w.WEATHER1_IA2 AS WEATHER2_IA2,
# MAGIC       w.WEATHER1_GJ1 AS WEATHER2_GJ1,
# MAGIC       w.WEATHER1_GA3 AS WEATHER2_GA3,
# MAGIC       w.WEATHER1_GD5 AS WEATHER2_GD5,
# MAGIC       w.WEATHER1_CR1 AS WEATHER2_CR1,
# MAGIC       w.WEATHER1_CF2 AS WEATHER2_CF2,
# MAGIC       w.WEATHER1_DEW_POINT_QUALITY_CODE AS WEATHER2_DEW_POINT_QUALITY_CODE,
# MAGIC       w.WEATHER1_AT1 AS WEATHER2_AT1,
# MAGIC       w.WEATHER1_KB1 AS WEATHER2_KB1,
# MAGIC       w.WEATHER1_GD3 AS WEATHER2_GD3,
# MAGIC       w.WEATHER1_KC2 AS WEATHER2_KC2,
# MAGIC       w.WEATHER1_CV2 AS WEATHER2_CV2,
# MAGIC       w.WEATHER1_AL2 AS WEATHER2_AL2,
# MAGIC       w.WEATHER1_AH5 AS WEATHER2_AH5,
# MAGIC       w.WEATHER1_KG2 AS WEATHER2_KG2,
# MAGIC       w.WEATHER1_ME1 AS WEATHER2_ME1,
# MAGIC       w.WEATHER1_AE1 AS WEATHER2_AE1,
# MAGIC       w.WEATHER1_AW6 AS WEATHER2_AW6,
# MAGIC       w.WEATHER1_AX1 AS WEATHER2_AX1,
# MAGIC       w.WEATHER1_KE1 AS WEATHER2_KE1,
# MAGIC       w.WEATHER1_SA1 AS WEATHER2_SA1,
# MAGIC       w.WEATHER1_OB1 AS WEATHER2_OB1,
# MAGIC       w.WEATHER1_AZ1 AS WEATHER2_AZ1,
# MAGIC       w.WEATHER1_MD1 AS WEATHER2_MD1,
# MAGIC       w.WEATHER1_AA4 AS WEATHER2_AA4,
# MAGIC       w.WEATHER1_MV2 AS WEATHER2_MV2,
# MAGIC       w.WEATHER1_WND_TYPE_CODE AS WEATHER2_WND_TYPE_CODE,
# MAGIC       w.WEATHER1_TMP_AIR_TEMP_QUALITY_CODE AS WEATHER2_TMP_AIR_TEMP_QUALITY_CODE,
# MAGIC       w.WEATHER1_GD4 AS WEATHER2_GD4,
# MAGIC       w.WEATHER1_AL1 AS WEATHER2_AL1,
# MAGIC       w.WEATHER1_GA6 AS WEATHER2_GA6,
# MAGIC       w.WEATHER1_OD1 AS WEATHER2_OD1,
# MAGIC       w.WEATHER1_AB1 AS WEATHER2_AB1,
# MAGIC       w.WEATHER1_CW1 AS WEATHER2_CW1,
# MAGIC       w.WEATHER1_AH4 AS WEATHER2_AH4,
# MAGIC       w.WEATHER1_SLP_SEA_LEVEL_PRES AS WEATHER2_SLP_SEA_LEVEL_PRES,
# MAGIC       w.WEATHER1_AX2 AS WEATHER2_AX2,
# MAGIC       w.WEATHER1_IB2 AS WEATHER2_IB2,
# MAGIC       w.WEATHER1_IB1 AS WEATHER2_IB1,
# MAGIC       w.WEATHER1_AX4 AS WEATHER2_AX4,
# MAGIC       w.WEATHER1_WND_SPEED_RATE AS WEATHER2_WND_SPEED_RATE,
# MAGIC       w.WEATHER1_AT6 AS WEATHER2_AT6,
# MAGIC       w.WEATHER1_KB3 AS WEATHER2_KB3,
# MAGIC       w.WEATHER1_MW6 AS WEATHER2_MW6,
# MAGIC       w.WEATHER1_AH2 AS WEATHER2_AH2,
# MAGIC       w.WEATHER1_GG1 AS WEATHER2_GG1,
# MAGIC       w.WEATHER1_AZ2 AS WEATHER2_AZ2,
# MAGIC       w.WEATHER1_QUALITY_CONTROL AS WEATHER2_QUALITY_CONTROL,
# MAGIC       w.WEATHER1_AY2 AS WEATHER2_AY2,
# MAGIC       w.WEATHER1_CH1 AS WEATHER2_CH1,
# MAGIC       w.WEATHER1_AU5 AS WEATHER2_AU5,
# MAGIC       w.WEATHER1_HL1 AS WEATHER2_HL1,
# MAGIC       w.WEATHER1_CIG_CAVOK_CODE AS WEATHER2_CIG_CAVOK_CODE,
# MAGIC       w.WEATHER1_VIS_DISTANCE_DIMENSION AS WEATHER2_VIS_DISTANCE_DIMENSION,
# MAGIC       w.WEATHER1_GH1 AS WEATHER2_GH1,
# MAGIC       w.WEATHER1_WEATHER_STATION AS WEATHER2_WEATHER_STATION,
# MAGIC       w.WEATHER1_AI5 AS WEATHER2_AI5,
# MAGIC       w.WEATHER1_GG3 AS WEATHER2_GG3,
# MAGIC       w.WEATHER1_AT3 AS WEATHER2_AT3,
# MAGIC       w.WEATHER1_MW3 AS WEATHER2_MW3,
# MAGIC       w.WEATHER1_CT2 AS WEATHER2_CT2,
# MAGIC       w.WEATHER1_KF1 AS WEATHER2_KF1,
# MAGIC       w.WEATHER1_RH2 AS WEATHER2_RH2,
# MAGIC       w.WEATHER1_EQD AS WEATHER2_EQD,
# MAGIC       w.WEATHER1_CI1 AS WEATHER2_CI1,
# MAGIC       w.WEATHER1_VIS_DISTANCE_QUALITY_CODE AS WEATHER2_VIS_DISTANCE_QUALITY_CODE,
# MAGIC       w.WEATHER1_VIS_QUALITY_VARIABILITY_CODE AS WEATHER2_VIS_QUALITY_VARIABILITY_CODE,
# MAGIC       w.WEATHER1_WEATHER_SOURCE AS WEATHER2_WEATHER_SOURCE,
# MAGIC       w.WEATHER1_WND_QUALITY_CODE AS WEATHER2_WND_QUALITY_CODE,
# MAGIC       w.WEATHER1_WND_SPEED_QUALITY_CODE AS WEATHER2_WND_SPEED_QUALITY_CODE
# MAGIC     FROM
# MAGIC       weather_processed w
# MAGIC   ) w2 ON fp.ORIGIN_WEATHER_KEY = w2.WEATHER2_WEATHER_KEY

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE flights_and_weather_processed
# MAGIC USING DELTA
# MAGIC LOCATION '/airline_delays/kevin/DLRS/flights_and_weather'
# MAGIC AS SELECT * FROM flights_processed VERSION AS OF 3 fp JOIN ( 
# MAGIC                                                           SELECT * FROM (
# MAGIC                                                                 SELECT *, ROW_NUMBER() OVER ( 
# MAGIC                                                                       partition by WEATHER1_WEATHER_KEY 
# MAGIC                                                                       ORDER BY WEATHER1_WEATHER_DATE ASC 
# MAGIC                                                                 ) as w1_row_num 
# MAGIC                                                                 FROM weather_processed 
# MAGIC                                                             ) as ordered_weather 
# MAGIC                                                             WHERE ordered_weather.w1_row_num = 1 
# MAGIC                                                         ) as w1 
# MAGIC                                                         ON fp.ORIGIN_WEATHER_KEY = w1.WEATHER1_WEATHER_KEY
# MAGIC                                                       JOIN (
# MAGIC                                                         SELECT  w.WEATHER1_GN1 AS WEATHER2_GN1,
# MAGIC                                                                 w.WEATHER1_GF1 AS WEATHER2_GF1,
# MAGIC                                                                 w.WEATHER1_UA1 AS WEATHER2_UA1,
# MAGIC                                                                 w.WEATHER1_AU2 AS WEATHER2_AU2,
# MAGIC                                                                 w.WEATHER1_AX5 AS WEATHER2_AX5,
# MAGIC                                                                 w.WEATHER1_TMP_AIR_TEMP AS WEATHER2_TMP_AIR_TEMP,
# MAGIC                                                                 w.WEATHER1_MK1 AS WEATHER2_MK1,
# MAGIC                                                                 w.WEATHER1_CN3 AS WEATHER2_CN3,
# MAGIC                                                                 w.WEATHER1_GM1 AS WEATHER2_GM1,
# MAGIC                                                                 w.WEATHER1_GA2 AS WEATHER2_GA2,
# MAGIC                                                                 w.WEATHER1_SLP_SEA_LEVEL_PRES_QUALITY_CODE AS WEATHER2_SLP_SEA_LEVEL_PRES_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_AW7 AS WEATHER2_AW7,
# MAGIC                                                                 w.WEATHER1_MG1 AS WEATHER2_MG1,
# MAGIC                                                                 w.WEATHER1_CG3 AS WEATHER2_CG3,
# MAGIC                                                                 w.WEATHER1_VIS_VARIABILITY_CODE AS WEATHER2_VIS_VARIABILITY_CODE,
# MAGIC                                                                 w.WEATHER1_GO1 AS WEATHER2_GO1,
# MAGIC                                                                 w.WEATHER1_AL3 AS WEATHER2_AL3,
# MAGIC                                                                 w.WEATHER1_AI1 AS WEATHER2_AI1,
# MAGIC                                                                 w.WEATHER1_MF1 AS WEATHER2_MF1,
# MAGIC                                                                 w.WEATHER1_WEATHER_KEY AS WEATHER2_WEATHER_KEY,
# MAGIC                                                                 w.WEATHER1_KA3 AS WEATHER2_KA3,
# MAGIC                                                                 w.WEATHER1_AI4 AS WEATHER2_AI4,
# MAGIC                                                                 w.WEATHER1_AK1 AS WEATHER2_AK1,
# MAGIC                                                                 w.WEATHER1_OE3 AS WEATHER2_OE3,
# MAGIC                                                                 w.WEATHER1_AW2 AS WEATHER2_AW2,
# MAGIC                                                                 w.WEATHER1_REM AS WEATHER2_REM,
# MAGIC                                                                 w.WEATHER1_OD2 AS WEATHER2_OD2,
# MAGIC                                                                 w.WEATHER1_CN4 AS WEATHER2_CN4,
# MAGIC                                                                 w.WEATHER1_AO1 AS WEATHER2_AO1,
# MAGIC                                                                 w.WEATHER1_CO1 AS WEATHER2_CO1,
# MAGIC                                                                 w.WEATHER1_OE2 AS WEATHER2_OE2,
# MAGIC                                                                 w.WEATHER1_CG1 AS WEATHER2_CG1,
# MAGIC                                                                 w.WEATHER1_AX6 AS WEATHER2_AX6,
# MAGIC                                                                 w.WEATHER1_KA2 AS WEATHER2_KA2,
# MAGIC                                                                 w.WEATHER1_CU2 AS WEATHER2_CU2,
# MAGIC                                                                 w.WEATHER1_AH3 AS WEATHER2_AH3,
# MAGIC                                                                 w.WEATHER1_OE1 AS WEATHER2_OE1,
# MAGIC                                                                 w.WEATHER1_MA1 AS WEATHER2_MA1,
# MAGIC                                                                 w.WEATHER1_CT1 AS WEATHER2_CT1,
# MAGIC                                                                 w.WEATHER1_AW4 AS WEATHER2_AW4,
# MAGIC                                                                 w.WEATHER1_AU3 AS WEATHER2_AU3,
# MAGIC                                                                 w.WEATHER1_GA5 AS WEATHER2_GA5,
# MAGIC                                                                 w.WEATHER1_UG1 AS WEATHER2_UG1,
# MAGIC                                                                 w.WEATHER1_GE1 AS WEATHER2_GE1,
# MAGIC                                                                 w.WEATHER1_AI3 AS WEATHER2_AI3,
# MAGIC                                                                 w.WEATHER1_GD1 AS WEATHER2_GD1,
# MAGIC                                                                 w.WEATHER1_CIG_CEILING_QUALITY_CODE AS WEATHER2_CIG_CEILING_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_AX3 AS WEATHER2_AX3,
# MAGIC                                                                 w.WEATHER1_KD1 AS WEATHER2_KD1,
# MAGIC                                                                 w.WEATHER1_CIG_CEILING_HEIGHT_DIMENSION AS WEATHER2_CIG_CEILING_HEIGHT_DIMENSION,
# MAGIC                                                                 w.WEATHER1_AW3 AS WEATHER2_AW3,
# MAGIC                                                                 w.WEATHER1_CV1 AS WEATHER2_CV1,
# MAGIC                                                                 w.WEATHER1_AY1 AS WEATHER2_AY1,
# MAGIC                                                                 w.WEATHER1_MV1 AS WEATHER2_MV1,
# MAGIC                                                                 w.WEATHER1_KA4 AS WEATHER2_KA4,
# MAGIC                                                                 w.WEATHER1_AJ1 AS WEATHER2_AJ1,
# MAGIC                                                                 w.WEATHER1_WEATHER_DATE AS WEATHER2_WEATHER_DATE,
# MAGIC                                                                 w.WEATHER1_CG2 AS WEATHER2_CG2,
# MAGIC                                                                 w.WEATHER1_KD2 AS WEATHER2_KD2,
# MAGIC                                                                 w.WEATHER1_DEW_POINT_TEMP AS WEATHER2_DEW_POINT_TEMP,
# MAGIC                                                                 w.WEATHER1_AH1 AS WEATHER2_AH1,
# MAGIC                                                                 w.WEATHER1_AU1 AS WEATHER2_AU1,
# MAGIC                                                                 w.WEATHER1_GL1 AS WEATHER2_GL1,
# MAGIC                                                                 w.WEATHER1_MW5 AS WEATHER2_MW5,
# MAGIC                                                                 w.WEATHER1_AU4 AS WEATHER2_AU4,
# MAGIC                                                                 w.WEATHER1_AT4 AS WEATHER2_AT4,
# MAGIC                                                                 w.WEATHER1_CN2 AS WEATHER2_CN2,
# MAGIC                                                                 w.WEATHER1_CV3 AS WEATHER2_CV3,
# MAGIC                                                                 w.WEATHER1_MW2 AS WEATHER2_MW2,
# MAGIC                                                                 w.WEATHER1_AT7 AS WEATHER2_AT7,
# MAGIC                                                                 w.WEATHER1_ED1 AS WEATHER2_ED1,
# MAGIC                                                                 w.WEATHER1_CU3 AS WEATHER2_CU3,
# MAGIC                                                                 w.WEATHER1_UG2 AS WEATHER2_UG2,
# MAGIC                                                                 w.WEATHER1_WD1 AS WEATHER2_WD1,
# MAGIC                                                                 w.WEATHER1_RH3 AS WEATHER2_RH3,
# MAGIC                                                                 w.WEATHER1_AT5 AS WEATHER2_AT5,
# MAGIC                                                                 w.WEATHER1_MW1 AS WEATHER2_MW1,
# MAGIC                                                                 w.WEATHER1_GK1 AS WEATHER2_GK1,
# MAGIC                                                                 w.WEATHER1_CU1 AS WEATHER2_CU1,
# MAGIC                                                                 w.WEATHER1_AA3 AS WEATHER2_AA3,
# MAGIC                                                                 w.WEATHER1_AM1 AS WEATHER2_AM1,
# MAGIC                                                                 w.WEATHER1_OD3 AS WEATHER2_OD3,
# MAGIC                                                                 w.WEATHER1_GA1 AS WEATHER2_GA1,
# MAGIC                                                                 w.WEATHER1_AT2 AS WEATHER2_AT2,
# MAGIC                                                                 w.WEATHER1_AI6 AS WEATHER2_AI6,
# MAGIC                                                                 w.WEATHER1_AT8 AS WEATHER2_AT8,
# MAGIC                                                                 w.WEATHER1_RH1 AS WEATHER2_RH1,
# MAGIC                                                                 w.WEATHER1_MW4 AS WEATHER2_MW4,
# MAGIC                                                                 w.WEATHER1_AA1 AS WEATHER2_AA1,
# MAGIC                                                                 w.WEATHER1_AN1 AS WEATHER2_AN1,
# MAGIC                                                                 w.WEATHER1_AH6 AS WEATHER2_AH6,
# MAGIC                                                                 w.WEATHER1_CF3 AS WEATHER2_CF3,
# MAGIC                                                                 w.WEATHER1_CF1 AS WEATHER2_CF1,
# MAGIC                                                                 w.WEATHER1_CIG_CEILING_DETERMINATION_CODE AS WEATHER2_CIG_CEILING_DETERMINATION_CODE,
# MAGIC                                                                 w.WEATHER1_AI2 AS WEATHER2_AI2,
# MAGIC                                                                 w.WEATHER1_CN1 AS WEATHER2_CN1,
# MAGIC                                                                 w.WEATHER1_CT3 AS WEATHER2_CT3,
# MAGIC                                                                 w.WEATHER1_GG4 AS WEATHER2_GG4,
# MAGIC                                                                 w.WEATHER1_CB1 AS WEATHER2_CB1,
# MAGIC                                                                 w.WEATHER1_GD2 AS WEATHER2_GD2,
# MAGIC                                                                 w.WEATHER1_GA4 AS WEATHER2_GA4,
# MAGIC                                                                 w.WEATHER1_KA1 AS WEATHER2_KA1,
# MAGIC                                                                 w.WEATHER1_AD1 AS WEATHER2_AD1,
# MAGIC                                                                 w.WEATHER1_WA1 AS WEATHER2_WA1,
# MAGIC                                                                 w.WEATHER1_AW1 AS WEATHER2_AW1,
# MAGIC                                                                 w.WEATHER1_MH1 AS WEATHER2_MH1,
# MAGIC                                                                 w.WEATHER1_KB2 AS WEATHER2_KB2,
# MAGIC                                                                 w.WEATHER1_KG1 AS WEATHER2_KG1,
# MAGIC                                                                 w.WEATHER1_AA2 AS WEATHER2_AA2,
# MAGIC                                                                 w.WEATHER1_GG2 AS WEATHER2_GG2,
# MAGIC                                                                 w.WEATHER1_KC1 AS WEATHER2_KC1,
# MAGIC                                                                 w.WEATHER1_OC1 AS WEATHER2_OC1,
# MAGIC                                                                 w.WEATHER1_IA1 AS WEATHER2_IA1,
# MAGIC                                                                 w.WEATHER1_AW5 AS WEATHER2_AW5,
# MAGIC                                                                 w.WEATHER1_WND_DIRECTION_ANGLE AS WEATHER2_WND_DIRECTION_ANGLE,
# MAGIC                                                                 w.WEATHER1_IA2 AS WEATHER2_IA2,
# MAGIC                                                                 w.WEATHER1_GJ1 AS WEATHER2_GJ1,
# MAGIC                                                                 w.WEATHER1_GA3 AS WEATHER2_GA3,
# MAGIC                                                                 w.WEATHER1_GD5 AS WEATHER2_GD5,
# MAGIC                                                                 w.WEATHER1_CR1 AS WEATHER2_CR1,
# MAGIC                                                                 w.WEATHER1_CF2 AS WEATHER2_CF2,
# MAGIC                                                                 w.WEATHER1_DEW_POINT_QUALITY_CODE AS WEATHER2_DEW_POINT_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_AT1 AS WEATHER2_AT1,
# MAGIC                                                                 w.WEATHER1_KB1 AS WEATHER2_KB1,
# MAGIC                                                                 w.WEATHER1_GD3 AS WEATHER2_GD3,
# MAGIC                                                                 w.WEATHER1_KC2 AS WEATHER2_KC2,
# MAGIC                                                                 w.WEATHER1_CV2 AS WEATHER2_CV2,
# MAGIC                                                                 w.WEATHER1_AL2 AS WEATHER2_AL2,
# MAGIC                                                                 w.WEATHER1_AH5 AS WEATHER2_AH5,
# MAGIC                                                                 w.WEATHER1_KG2 AS WEATHER2_KG2,
# MAGIC                                                                 w.WEATHER1_ME1 AS WEATHER2_ME1,
# MAGIC                                                                 w.WEATHER1_AE1 AS WEATHER2_AE1,
# MAGIC                                                                 w.WEATHER1_AW6 AS WEATHER2_AW6,
# MAGIC                                                                 w.WEATHER1_AX1 AS WEATHER2_AX1,
# MAGIC                                                                 w.WEATHER1_KE1 AS WEATHER2_KE1,
# MAGIC                                                                 w.WEATHER1_SA1 AS WEATHER2_SA1,
# MAGIC                                                                 w.WEATHER1_OB1 AS WEATHER2_OB1,
# MAGIC                                                                 w.WEATHER1_AZ1 AS WEATHER2_AZ1,
# MAGIC                                                                 w.WEATHER1_MD1 AS WEATHER2_MD1,
# MAGIC                                                                 w.WEATHER1_AA4 AS WEATHER2_AA4,
# MAGIC                                                                 w.WEATHER1_MV2 AS WEATHER2_MV2,
# MAGIC                                                                 w.WEATHER1_WND_TYPE_CODE AS WEATHER2_WND_TYPE_CODE,
# MAGIC                                                                 w.WEATHER1_TMP_AIR_TEMP_QUALITY_CODE AS WEATHER2_TMP_AIR_TEMP_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_GD4 AS WEATHER2_GD4,
# MAGIC                                                                 w.WEATHER1_AL1 AS WEATHER2_AL1,
# MAGIC                                                                 w.WEATHER1_GA6 AS WEATHER2_GA6,
# MAGIC                                                                 w.WEATHER1_OD1 AS WEATHER2_OD1,
# MAGIC                                                                 w.WEATHER1_AB1 AS WEATHER2_AB1,
# MAGIC                                                                 w.WEATHER1_CW1 AS WEATHER2_CW1,
# MAGIC                                                                 w.WEATHER1_AH4 AS WEATHER2_AH4,
# MAGIC                                                                 w.WEATHER1_SLP_SEA_LEVEL_PRES AS WEATHER2_SLP_SEA_LEVEL_PRES,
# MAGIC                                                                 w.WEATHER1_AX2 AS WEATHER2_AX2,
# MAGIC                                                                 w.WEATHER1_IB2 AS WEATHER2_IB2,
# MAGIC                                                                 w.WEATHER1_IB1 AS WEATHER2_IB1,
# MAGIC                                                                 w.WEATHER1_AX4 AS WEATHER2_AX4,
# MAGIC                                                                 w.WEATHER1_WND_SPEED_RATE AS WEATHER2_WND_SPEED_RATE,
# MAGIC                                                                 w.WEATHER1_AT6 AS WEATHER2_AT6,
# MAGIC                                                                 w.WEATHER1_KB3 AS WEATHER2_KB3,
# MAGIC                                                                 w.WEATHER1_MW6 AS WEATHER2_MW6,
# MAGIC                                                                 w.WEATHER1_AH2 AS WEATHER2_AH2,
# MAGIC                                                                 w.WEATHER1_GG1 AS WEATHER2_GG1,
# MAGIC                                                                 w.WEATHER1_AZ2 AS WEATHER2_AZ2,
# MAGIC                                                                 w.WEATHER1_QUALITY_CONTROL AS WEATHER2_QUALITY_CONTROL,
# MAGIC                                                                 w.WEATHER1_AY2 AS WEATHER2_AY2,
# MAGIC                                                                 w.WEATHER1_CH1 AS WEATHER2_CH1,
# MAGIC                                                                 w.WEATHER1_AU5 AS WEATHER2_AU5,
# MAGIC                                                                 w.WEATHER1_HL1 AS WEATHER2_HL1,
# MAGIC                                                                 w.WEATHER1_CIG_CAVOK_CODE AS WEATHER2_CIG_CAVOK_CODE,
# MAGIC                                                                 w.WEATHER1_VIS_DISTANCE_DIMENSION AS WEATHER2_VIS_DISTANCE_DIMENSION,
# MAGIC                                                                 w.WEATHER1_GH1 AS WEATHER2_GH1,
# MAGIC                                                                 w.WEATHER1_WEATHER_STATION AS WEATHER2_WEATHER_STATION,
# MAGIC                                                                 w.WEATHER1_AI5 AS WEATHER2_AI5,
# MAGIC                                                                 w.WEATHER1_GG3 AS WEATHER2_GG3,
# MAGIC                                                                 w.WEATHER1_AT3 AS WEATHER2_AT3,
# MAGIC                                                                 w.WEATHER1_MW3 AS WEATHER2_MW3,
# MAGIC                                                                 w.WEATHER1_CT2 AS WEATHER2_CT2,
# MAGIC                                                                 w.WEATHER1_KF1 AS WEATHER2_KF1,
# MAGIC                                                                 w.WEATHER1_RH2 AS WEATHER2_RH2,
# MAGIC                                                                 w.WEATHER1_EQD AS WEATHER2_EQD,
# MAGIC                                                                 w.WEATHER1_CI1 AS WEATHER2_CI1,
# MAGIC                                                                 w.WEATHER1_VIS_DISTANCE_QUALITY_CODE AS WEATHER2_VIS_DISTANCE_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_VIS_QUALITY_VARIABILITY_CODE AS WEATHER2_VIS_QUALITY_VARIABILITY_CODE,
# MAGIC                                                                 w.WEATHER1_WEATHER_SOURCE AS WEATHER2_WEATHER_SOURCE,
# MAGIC                                                                 w.WEATHER1_WND_QUALITY_CODE AS WEATHER2_WND_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_WND_SPEED_QUALITY_CODE AS WEATHER2_WND_SPEED_QUALITY_CODE,
# MAGIC                                                                 w.w2_row_num
# MAGIC                                                           FROM (
# MAGIC                                                                 SELECT *, ROW_NUMBER() OVER ( 
# MAGIC                                                                       partition by WEATHER1_WEATHER_KEY 
# MAGIC                                                                       ORDER BY WEATHER1_WEATHER_DATE ASC 
# MAGIC                                                                 ) as w2_row_num 
# MAGIC                                                                 FROM weather_processed 
# MAGIC                                                             ) as w
# MAGIC                                                             WHERE w.w2_row_num = 1 
# MAGIC                                                         ) as w2
# MAGIC                                                        ON fp.DEST_WEATHER_KEY = w2.WEATHER2_WEATHER_KEY

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE flights_and_weather_processed
# MAGIC USING DELTA
# MAGIC LOCATION '/airline_delays/kevin/DLRS/flights_and_weather'
# MAGIC AS SELECT * FROM flights_processed VERSION AS OF 3 fp JOIN ( 
# MAGIC                                                           SELECT * FROM (
# MAGIC                                                                 SELECT *, ROW_NUMBER() OVER ( 
# MAGIC                                                                       partition by WEATHER1_WEATHER_KEY 
# MAGIC                                                                       ORDER BY WEATHER1_WEATHER_DATE ASC 
# MAGIC                                                                 ) as w1_row_num 
# MAGIC                                                                 FROM weather_processed 
# MAGIC                                                             ) as ordered_weather 
# MAGIC                                                             WHERE ordered_weather.w1_row_num = 1 
# MAGIC                                                         ) as w1 
# MAGIC                                                         ON fp.ORIGIN_WEATHER_KEY = w1.WEATHER1_WEATHER_KEY

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from flights_and_weather_processed

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE flights_and_weather_combined_processed
# MAGIC USING DELTA
# MAGIC LOCATION '/airline_delays/kevin/DLRS/flights_and_weather_combined'
# MAGIC AS SELECT * FROM flights_and_weather_processed fp JOIN (
# MAGIC                                                         SELECT  w.WEATHER1_GN1 AS WEATHER2_GN1,
# MAGIC                                                                 w.WEATHER1_GF1 AS WEATHER2_GF1,
# MAGIC                                                                 w.WEATHER1_UA1 AS WEATHER2_UA1,
# MAGIC                                                                 w.WEATHER1_AU2 AS WEATHER2_AU2,
# MAGIC                                                                 w.WEATHER1_AX5 AS WEATHER2_AX5,
# MAGIC                                                                 w.WEATHER1_TMP_AIR_TEMP AS WEATHER2_TMP_AIR_TEMP,
# MAGIC                                                                 w.WEATHER1_MK1 AS WEATHER2_MK1,
# MAGIC                                                                 w.WEATHER1_CN3 AS WEATHER2_CN3,
# MAGIC                                                                 w.WEATHER1_GM1 AS WEATHER2_GM1,
# MAGIC                                                                 w.WEATHER1_GA2 AS WEATHER2_GA2,
# MAGIC                                                                 w.WEATHER1_SLP_SEA_LEVEL_PRES_QUALITY_CODE AS WEATHER2_SLP_SEA_LEVEL_PRES_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_AW7 AS WEATHER2_AW7,
# MAGIC                                                                 w.WEATHER1_MG1 AS WEATHER2_MG1,
# MAGIC                                                                 w.WEATHER1_CG3 AS WEATHER2_CG3,
# MAGIC                                                                 w.WEATHER1_VIS_VARIABILITY_CODE AS WEATHER2_VIS_VARIABILITY_CODE,
# MAGIC                                                                 w.WEATHER1_GO1 AS WEATHER2_GO1,
# MAGIC                                                                 w.WEATHER1_AL3 AS WEATHER2_AL3,
# MAGIC                                                                 w.WEATHER1_AI1 AS WEATHER2_AI1,
# MAGIC                                                                 w.WEATHER1_MF1 AS WEATHER2_MF1,
# MAGIC                                                                 w.WEATHER1_WEATHER_KEY AS WEATHER2_WEATHER_KEY,
# MAGIC                                                                 w.WEATHER1_KA3 AS WEATHER2_KA3,
# MAGIC                                                                 w.WEATHER1_AI4 AS WEATHER2_AI4,
# MAGIC                                                                 w.WEATHER1_AK1 AS WEATHER2_AK1,
# MAGIC                                                                 w.WEATHER1_OE3 AS WEATHER2_OE3,
# MAGIC                                                                 w.WEATHER1_AW2 AS WEATHER2_AW2,
# MAGIC                                                                 w.WEATHER1_REM AS WEATHER2_REM,
# MAGIC                                                                 w.WEATHER1_OD2 AS WEATHER2_OD2,
# MAGIC                                                                 w.WEATHER1_CN4 AS WEATHER2_CN4,
# MAGIC                                                                 w.WEATHER1_AO1 AS WEATHER2_AO1,
# MAGIC                                                                 w.WEATHER1_CO1 AS WEATHER2_CO1,
# MAGIC                                                                 w.WEATHER1_OE2 AS WEATHER2_OE2,
# MAGIC                                                                 w.WEATHER1_CG1 AS WEATHER2_CG1,
# MAGIC                                                                 w.WEATHER1_AX6 AS WEATHER2_AX6,
# MAGIC                                                                 w.WEATHER1_KA2 AS WEATHER2_KA2,
# MAGIC                                                                 w.WEATHER1_CU2 AS WEATHER2_CU2,
# MAGIC                                                                 w.WEATHER1_AH3 AS WEATHER2_AH3,
# MAGIC                                                                 w.WEATHER1_OE1 AS WEATHER2_OE1,
# MAGIC                                                                 w.WEATHER1_MA1 AS WEATHER2_MA1,
# MAGIC                                                                 w.WEATHER1_CT1 AS WEATHER2_CT1,
# MAGIC                                                                 w.WEATHER1_AW4 AS WEATHER2_AW4,
# MAGIC                                                                 w.WEATHER1_AU3 AS WEATHER2_AU3,
# MAGIC                                                                 w.WEATHER1_GA5 AS WEATHER2_GA5,
# MAGIC                                                                 w.WEATHER1_UG1 AS WEATHER2_UG1,
# MAGIC                                                                 w.WEATHER1_GE1 AS WEATHER2_GE1,
# MAGIC                                                                 w.WEATHER1_AI3 AS WEATHER2_AI3,
# MAGIC                                                                 w.WEATHER1_GD1 AS WEATHER2_GD1,
# MAGIC                                                                 w.WEATHER1_CIG_CEILING_QUALITY_CODE AS WEATHER2_CIG_CEILING_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_AX3 AS WEATHER2_AX3,
# MAGIC                                                                 w.WEATHER1_KD1 AS WEATHER2_KD1,
# MAGIC                                                                 w.WEATHER1_CIG_CEILING_HEIGHT_DIMENSION AS WEATHER2_CIG_CEILING_HEIGHT_DIMENSION,
# MAGIC                                                                 w.WEATHER1_AW3 AS WEATHER2_AW3,
# MAGIC                                                                 w.WEATHER1_CV1 AS WEATHER2_CV1,
# MAGIC                                                                 w.WEATHER1_AY1 AS WEATHER2_AY1,
# MAGIC                                                                 w.WEATHER1_MV1 AS WEATHER2_MV1,
# MAGIC                                                                 w.WEATHER1_KA4 AS WEATHER2_KA4,
# MAGIC                                                                 w.WEATHER1_AJ1 AS WEATHER2_AJ1,
# MAGIC                                                                 w.WEATHER1_WEATHER_DATE AS WEATHER2_WEATHER_DATE,
# MAGIC                                                                 w.WEATHER1_CG2 AS WEATHER2_CG2,
# MAGIC                                                                 w.WEATHER1_KD2 AS WEATHER2_KD2,
# MAGIC                                                                 w.WEATHER1_DEW_POINT_TEMP AS WEATHER2_DEW_POINT_TEMP,
# MAGIC                                                                 w.WEATHER1_AH1 AS WEATHER2_AH1,
# MAGIC                                                                 w.WEATHER1_AU1 AS WEATHER2_AU1,
# MAGIC                                                                 w.WEATHER1_GL1 AS WEATHER2_GL1,
# MAGIC                                                                 w.WEATHER1_MW5 AS WEATHER2_MW5,
# MAGIC                                                                 w.WEATHER1_AU4 AS WEATHER2_AU4,
# MAGIC                                                                 w.WEATHER1_AT4 AS WEATHER2_AT4,
# MAGIC                                                                 w.WEATHER1_CN2 AS WEATHER2_CN2,
# MAGIC                                                                 w.WEATHER1_CV3 AS WEATHER2_CV3,
# MAGIC                                                                 w.WEATHER1_MW2 AS WEATHER2_MW2,
# MAGIC                                                                 w.WEATHER1_AT7 AS WEATHER2_AT7,
# MAGIC                                                                 w.WEATHER1_ED1 AS WEATHER2_ED1,
# MAGIC                                                                 w.WEATHER1_CU3 AS WEATHER2_CU3,
# MAGIC                                                                 w.WEATHER1_UG2 AS WEATHER2_UG2,
# MAGIC                                                                 w.WEATHER1_WD1 AS WEATHER2_WD1,
# MAGIC                                                                 w.WEATHER1_RH3 AS WEATHER2_RH3,
# MAGIC                                                                 w.WEATHER1_AT5 AS WEATHER2_AT5,
# MAGIC                                                                 w.WEATHER1_MW1 AS WEATHER2_MW1,
# MAGIC                                                                 w.WEATHER1_GK1 AS WEATHER2_GK1,
# MAGIC                                                                 w.WEATHER1_CU1 AS WEATHER2_CU1,
# MAGIC                                                                 w.WEATHER1_AA3 AS WEATHER2_AA3,
# MAGIC                                                                 w.WEATHER1_AM1 AS WEATHER2_AM1,
# MAGIC                                                                 w.WEATHER1_OD3 AS WEATHER2_OD3,
# MAGIC                                                                 w.WEATHER1_GA1 AS WEATHER2_GA1,
# MAGIC                                                                 w.WEATHER1_AT2 AS WEATHER2_AT2,
# MAGIC                                                                 w.WEATHER1_AI6 AS WEATHER2_AI6,
# MAGIC                                                                 w.WEATHER1_AT8 AS WEATHER2_AT8,
# MAGIC                                                                 w.WEATHER1_RH1 AS WEATHER2_RH1,
# MAGIC                                                                 w.WEATHER1_MW4 AS WEATHER2_MW4,
# MAGIC                                                                 w.WEATHER1_AA1 AS WEATHER2_AA1,
# MAGIC                                                                 w.WEATHER1_AN1 AS WEATHER2_AN1,
# MAGIC                                                                 w.WEATHER1_AH6 AS WEATHER2_AH6,
# MAGIC                                                                 w.WEATHER1_CF3 AS WEATHER2_CF3,
# MAGIC                                                                 w.WEATHER1_CF1 AS WEATHER2_CF1,
# MAGIC                                                                 w.WEATHER1_CIG_CEILING_DETERMINATION_CODE AS WEATHER2_CIG_CEILING_DETERMINATION_CODE,
# MAGIC                                                                 w.WEATHER1_AI2 AS WEATHER2_AI2,
# MAGIC                                                                 w.WEATHER1_CN1 AS WEATHER2_CN1,
# MAGIC                                                                 w.WEATHER1_CT3 AS WEATHER2_CT3,
# MAGIC                                                                 w.WEATHER1_GG4 AS WEATHER2_GG4,
# MAGIC                                                                 w.WEATHER1_CB1 AS WEATHER2_CB1,
# MAGIC                                                                 w.WEATHER1_GD2 AS WEATHER2_GD2,
# MAGIC                                                                 w.WEATHER1_GA4 AS WEATHER2_GA4,
# MAGIC                                                                 w.WEATHER1_KA1 AS WEATHER2_KA1,
# MAGIC                                                                 w.WEATHER1_AD1 AS WEATHER2_AD1,
# MAGIC                                                                 w.WEATHER1_WA1 AS WEATHER2_WA1,
# MAGIC                                                                 w.WEATHER1_AW1 AS WEATHER2_AW1,
# MAGIC                                                                 w.WEATHER1_MH1 AS WEATHER2_MH1,
# MAGIC                                                                 w.WEATHER1_KB2 AS WEATHER2_KB2,
# MAGIC                                                                 w.WEATHER1_KG1 AS WEATHER2_KG1,
# MAGIC                                                                 w.WEATHER1_AA2 AS WEATHER2_AA2,
# MAGIC                                                                 w.WEATHER1_GG2 AS WEATHER2_GG2,
# MAGIC                                                                 w.WEATHER1_KC1 AS WEATHER2_KC1,
# MAGIC                                                                 w.WEATHER1_OC1 AS WEATHER2_OC1,
# MAGIC                                                                 w.WEATHER1_IA1 AS WEATHER2_IA1,
# MAGIC                                                                 w.WEATHER1_AW5 AS WEATHER2_AW5,
# MAGIC                                                                 w.WEATHER1_WND_DIRECTION_ANGLE AS WEATHER2_WND_DIRECTION_ANGLE,
# MAGIC                                                                 w.WEATHER1_IA2 AS WEATHER2_IA2,
# MAGIC                                                                 w.WEATHER1_GJ1 AS WEATHER2_GJ1,
# MAGIC                                                                 w.WEATHER1_GA3 AS WEATHER2_GA3,
# MAGIC                                                                 w.WEATHER1_GD5 AS WEATHER2_GD5,
# MAGIC                                                                 w.WEATHER1_CR1 AS WEATHER2_CR1,
# MAGIC                                                                 w.WEATHER1_CF2 AS WEATHER2_CF2,
# MAGIC                                                                 w.WEATHER1_DEW_POINT_QUALITY_CODE AS WEATHER2_DEW_POINT_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_AT1 AS WEATHER2_AT1,
# MAGIC                                                                 w.WEATHER1_KB1 AS WEATHER2_KB1,
# MAGIC                                                                 w.WEATHER1_GD3 AS WEATHER2_GD3,
# MAGIC                                                                 w.WEATHER1_KC2 AS WEATHER2_KC2,
# MAGIC                                                                 w.WEATHER1_CV2 AS WEATHER2_CV2,
# MAGIC                                                                 w.WEATHER1_AL2 AS WEATHER2_AL2,
# MAGIC                                                                 w.WEATHER1_AH5 AS WEATHER2_AH5,
# MAGIC                                                                 w.WEATHER1_KG2 AS WEATHER2_KG2,
# MAGIC                                                                 w.WEATHER1_ME1 AS WEATHER2_ME1,
# MAGIC                                                                 w.WEATHER1_AE1 AS WEATHER2_AE1,
# MAGIC                                                                 w.WEATHER1_AW6 AS WEATHER2_AW6,
# MAGIC                                                                 w.WEATHER1_AX1 AS WEATHER2_AX1,
# MAGIC                                                                 w.WEATHER1_KE1 AS WEATHER2_KE1,
# MAGIC                                                                 w.WEATHER1_SA1 AS WEATHER2_SA1,
# MAGIC                                                                 w.WEATHER1_OB1 AS WEATHER2_OB1,
# MAGIC                                                                 w.WEATHER1_AZ1 AS WEATHER2_AZ1,
# MAGIC                                                                 w.WEATHER1_MD1 AS WEATHER2_MD1,
# MAGIC                                                                 w.WEATHER1_AA4 AS WEATHER2_AA4,
# MAGIC                                                                 w.WEATHER1_MV2 AS WEATHER2_MV2,
# MAGIC                                                                 w.WEATHER1_WND_TYPE_CODE AS WEATHER2_WND_TYPE_CODE,
# MAGIC                                                                 w.WEATHER1_TMP_AIR_TEMP_QUALITY_CODE AS WEATHER2_TMP_AIR_TEMP_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_GD4 AS WEATHER2_GD4,
# MAGIC                                                                 w.WEATHER1_AL1 AS WEATHER2_AL1,
# MAGIC                                                                 w.WEATHER1_GA6 AS WEATHER2_GA6,
# MAGIC                                                                 w.WEATHER1_OD1 AS WEATHER2_OD1,
# MAGIC                                                                 w.WEATHER1_AB1 AS WEATHER2_AB1,
# MAGIC                                                                 w.WEATHER1_CW1 AS WEATHER2_CW1,
# MAGIC                                                                 w.WEATHER1_AH4 AS WEATHER2_AH4,
# MAGIC                                                                 w.WEATHER1_SLP_SEA_LEVEL_PRES AS WEATHER2_SLP_SEA_LEVEL_PRES,
# MAGIC                                                                 w.WEATHER1_AX2 AS WEATHER2_AX2,
# MAGIC                                                                 w.WEATHER1_IB2 AS WEATHER2_IB2,
# MAGIC                                                                 w.WEATHER1_IB1 AS WEATHER2_IB1,
# MAGIC                                                                 w.WEATHER1_AX4 AS WEATHER2_AX4,
# MAGIC                                                                 w.WEATHER1_WND_SPEED_RATE AS WEATHER2_WND_SPEED_RATE,
# MAGIC                                                                 w.WEATHER1_AT6 AS WEATHER2_AT6,
# MAGIC                                                                 w.WEATHER1_KB3 AS WEATHER2_KB3,
# MAGIC                                                                 w.WEATHER1_MW6 AS WEATHER2_MW6,
# MAGIC                                                                 w.WEATHER1_AH2 AS WEATHER2_AH2,
# MAGIC                                                                 w.WEATHER1_GG1 AS WEATHER2_GG1,
# MAGIC                                                                 w.WEATHER1_AZ2 AS WEATHER2_AZ2,
# MAGIC                                                                 w.WEATHER1_QUALITY_CONTROL AS WEATHER2_QUALITY_CONTROL,
# MAGIC                                                                 w.WEATHER1_AY2 AS WEATHER2_AY2,
# MAGIC                                                                 w.WEATHER1_CH1 AS WEATHER2_CH1,
# MAGIC                                                                 w.WEATHER1_AU5 AS WEATHER2_AU5,
# MAGIC                                                                 w.WEATHER1_HL1 AS WEATHER2_HL1,
# MAGIC                                                                 w.WEATHER1_CIG_CAVOK_CODE AS WEATHER2_CIG_CAVOK_CODE,
# MAGIC                                                                 w.WEATHER1_VIS_DISTANCE_DIMENSION AS WEATHER2_VIS_DISTANCE_DIMENSION,
# MAGIC                                                                 w.WEATHER1_GH1 AS WEATHER2_GH1,
# MAGIC                                                                 w.WEATHER1_WEATHER_STATION AS WEATHER2_WEATHER_STATION,
# MAGIC                                                                 w.WEATHER1_AI5 AS WEATHER2_AI5,
# MAGIC                                                                 w.WEATHER1_GG3 AS WEATHER2_GG3,
# MAGIC                                                                 w.WEATHER1_AT3 AS WEATHER2_AT3,
# MAGIC                                                                 w.WEATHER1_MW3 AS WEATHER2_MW3,
# MAGIC                                                                 w.WEATHER1_CT2 AS WEATHER2_CT2,
# MAGIC                                                                 w.WEATHER1_KF1 AS WEATHER2_KF1,
# MAGIC                                                                 w.WEATHER1_RH2 AS WEATHER2_RH2,
# MAGIC                                                                 w.WEATHER1_EQD AS WEATHER2_EQD,
# MAGIC                                                                 w.WEATHER1_CI1 AS WEATHER2_CI1,
# MAGIC                                                                 w.WEATHER1_VIS_DISTANCE_QUALITY_CODE AS WEATHER2_VIS_DISTANCE_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_VIS_QUALITY_VARIABILITY_CODE AS WEATHER2_VIS_QUALITY_VARIABILITY_CODE,
# MAGIC                                                                 w.WEATHER1_WEATHER_SOURCE AS WEATHER2_WEATHER_SOURCE,
# MAGIC                                                                 w.WEATHER1_WND_QUALITY_CODE AS WEATHER2_WND_QUALITY_CODE,
# MAGIC                                                                 w.WEATHER1_WND_SPEED_QUALITY_CODE AS WEATHER2_WND_SPEED_QUALITY_CODE,
# MAGIC                                                                 w.w2_row_num
# MAGIC                                                           FROM (
# MAGIC                                                                 SELECT *, ROW_NUMBER() OVER ( 
# MAGIC                                                                       partition by WEATHER1_WEATHER_KEY 
# MAGIC                                                                       ORDER BY WEATHER1_WEATHER_DATE ASC 
# MAGIC                                                                 ) as w2_row_num 
# MAGIC                                                                 FROM weather_processed 
# MAGIC                                                             ) as w
# MAGIC                                                             WHERE w.w2_row_num = 1 
# MAGIC                                                         ) as w2
# MAGIC                                                        ON fp.DEST_WEATHER_KEY = w2.WEATHER2_WEATHER_KEY

# COMMAND ----------

flights_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{flights_loc}processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*)
# MAGIC FROM flights_processed

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT WEATHER1_WEATHER_KEY)
# MAGIC FROM weather_processed

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM flights_processed
# MAGIC LIMIT 1

# COMMAND ----------

# MAGIC %md
# MAGIC Before doing the second join we need to create a copy of the weather data with the columns renamed.

# COMMAND ----------

weather_cols = weather_processed.columns
for col in weather_cols:
  #print(f'WEATHER2_{col.split("WEATHER1_")[1]}')
  weather_processed = weather_processed.withColumnRenamed(col,f'WEATHER2_{col.split("WEATHER1_")[1]}')

# COMMAND ----------

weather_processed.printSchema()

# COMMAND ----------

#save the duplicated weather data in Delta Lake
weather_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{weather_loc}processed_2')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS weather_processed_2;
# MAGIC 
# MAGIC CREATE TABLE weather_processed_2
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/weather/processed_2"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM weather_processed_2
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*)
# MAGIC FROM flights_processed

# COMMAND ----------

display(flights_processed.count())

# COMMAND ----------

flights_raw_df.count()

# COMMAND ----------

# second join
# join the origin weather
flights_processed =  flights_processed LEFT JOIN weather_processed_2 ON flights_processed.DEST_WEATHER_KEY=weather_processed_2.WEATHER2_WEATHER_KEY;")

# COMMAND ----------

flights_processed.write.option('mergeSchema', True).mode('overwrite').format('delta').save(f'{flights_loc}processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ORIGIN, count(*) AS CNT FROM flights_processed
# MAGIC WHERE WEATHER_KEY IS NULL
# MAGIC GROUP BY ORIGIN
# MAGIC ORDER BY CNT DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DEST, count(*) AS CNT FROM flights_processed
# MAGIC WHERE WEATHER2_WEATHER_KEY IS NULL
# MAGIC GROUP BY DEST
# MAGIC ORDER BY CNT DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT a.ORIGIN, ((CASE WHEN b.CNT_MISSING IS NOT NULL THEN b.CNT_MISSING
# MAGIC                         ELSE 0 END)/a.TOTAL) AS PROP_MISSING, a.TOTAL, b.CNT_MISSING
# MAGIC FROM (SELECT ORIGIN, COUNT(*) AS TOTAL 
# MAGIC       FROM flights_processed
# MAGIC       GROUP BY ORIGIN) a
# MAGIC JOIN (SELECT ORIGIN, count(*) AS CNT_MISSING
# MAGIC       FROM flights_processed
# MAGIC       WHERE WEATHER_KEY IS NULL
# MAGIC       GROUP BY ORIGIN) b
# MAGIC ON a.ORIGIN=b.ORIGIN
# MAGIC ORDER BY PROP_MISSING DESC;

# COMMAND ----------

# MAGIC %sql

# COMMAND ----------

# MAGIC %md
# MAGIC # Working Section

# COMMAND ----------

# MAGIC %md
# MAGIC # Remaining analysis on fully joined data table

# COMMAND ----------

#flights_train = spark.sql("SELECT * FROM flights_processed WHERE YEAR IN (2015, 2016, 2017);")
#flights_validate = spark.sql("SELECT * FROM flights_processed WHERE YEAR = 2018;")
#flights_test = spark.sql("SELECT * FROM flights_processed WHERE YEAR = 2019;")

# COMMAND ----------

# review categorical and numerical features:
cat_cols = [item[0] for item in flights_processed.dtypes if item[1].startswith('string')] 
print(str(len(cat_cols)) + '  categorical features')
num_cols = [item[0] for item in flights_processed.dtypes if item[1].startswith('int') | item[1].startswith('double')][1:]
print(str(len(num_cols)) + '  numerical features')

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

sample_df = spark.sql("SELECT * FROM flights_processed WHERE YEAR IN (2015, 2016, 2017)").sample(False, 0.01)

# COMMAND ----------

sample_panda_df = sample_df.toPandas()

# COMMAND ----------

sample_recs = info_missing_table(sample_panda_df)
sample_recs

# COMMAND ----------

colToDrop = sample_recs[sample_recs['% of Total Values'] > 95].index.values
colToDrop

# COMMAND ----------

colToDrop = ['DIV4_LONGEST_GTIME', 'DIV4_TOTAL_GTIME', 'DIV3_WHEELS_ON',
       'DIV3_TOTAL_GTIME', 'DIV3_LONGEST_GTIME', 'DIV3_WHEELS_OFF',
       'DIV3_TAIL_NUM', 'DIV4_AIRPORT', 'DIV4_AIRPORT_ID',
       'DIV4_AIRPORT_SEQ_ID', 'DIV4_WHEELS_ON', 'DIV4_WHEELS_OFF',
       'DIV3_AIRPORT', 'DIV4_TAIL_NUM', 'DIV5_AIRPORT', 'DIV5_AIRPORT_ID',
       'DIV5_AIRPORT_SEQ_ID', 'DIV5_WHEELS_ON', 'DIV5_TOTAL_GTIME',
       'DIV5_LONGEST_GTIME', 'DIV5_WHEELS_OFF', 'DIV5_TAIL_NUM',
       'DIV3_AIRPORT_SEQ_ID', 'DIV3_AIRPORT_ID', 'DIV2_WHEELS_OFF',
       'DIV2_TAIL_NUM', 'DIV2_AIRPORT_ID', 'DIV2_TOTAL_GTIME',
       'DIV2_WHEELS_ON', 'DIV2_AIRPORT_SEQ_ID', 'DIV2_AIRPORT',
       'DIV2_LONGEST_GTIME', 'DIV_ARR_DELAY', 'DIV_ACTUAL_ELAPSED_TIME',
       'DIV1_WHEELS_OFF', 'DIV1_TAIL_NUM', 'DIV_DISTANCE',
       'DIV_REACHED_DEST', 'DIV1_AIRPORT_ID', 'DIV1_AIRPORT_SEQ_ID',
       'DIV1_AIRPORT', 'DIV1_TOTAL_GTIME', 'DIV1_LONGEST_GTIME',
       'DIV1_WHEELS_ON', 'TOTAL_ADD_GTIME', 'FIRST_DEP_TIME',
       'LONGEST_ADD_GTIME', 'CANCELLATION_CODE']

# COMMAND ----------

colToKeep = sample_recs[sample_recs['% of Total Values'] < 95].index.values
colToKeep

# COMMAND ----------

colToKeep = ['ACTUAL_ELAPSED_TIME',
 'AIR_TIME',
 'ARR_DEL15',
 'ARR_DELAY',
 'ARR_DELAY_GROUP',
 'ARR_DELAY_NEW',
 'ARR_TIME',
 'CARRIER_DELAY',
 'CRS_ELAPSED_TIME',
 'DEP_DEL15',
 'DEP_DELAY',
 'DEP_DELAY_GROUP',
 'DEP_DELAY_NEW',
 'DEP_TIME',
 'IN_FLIGHT_AIR_DELAY',
 'LATE_AIRCRAFT_DELAY',
 'NAS_DELAY',
 'SECURITY_DELAY',
 'TAIL_NUM',
 'TAXI_IN',
 'TAXI_OUT',
 'WEATHER_DELAY',
 'WHEELS_OFF',
 'WHEELS_ON']

# COMMAND ----------

weather1ColToKeep = ['GN1', 'GF1', 'UA1', 'AU2',
       'AX5', 'TMP_Air_Temp', 'MK1',
       'CN3', 'GM1', 'GA2',
       'SLP_Sea_Level_Pres_Quality_Code', 'AW7',
       'MG1', 'CG3', 'VIS_Variability_Code',
       'GO1', 'AL3', 'AI1', 'MF1',
       'WEATHER_KEY', 'KA3', 'AI4',
       'AK1', 'OE3', 'AW2', 'REM',
       'OD2', 'CN4', 'AO1', 'CO1',
       'OE2', 'CG1', 'AX6', 'KA2',
       'CU2', 'AH3', 'OE1', 'MA1',
       'CT1', 'AW4', 'AU3', 'GA5',
       'UG1', 'GE1', 'AI3', 'GD1',
       'CIG_Ceiling_Quality_Code', 'AX3',
       'KD1', 'CIG_Ceiling_Height_Dimension',
       'AW3', 'CV1', 'AY1', 'MV1',
       'KA4', 'AJ1', 'WEATHER_DATE',
       'CG2', 'KD2', 'DEW_Point_Temp',
       'AH1', 'AU1', 'GL1', 'MW5',
       'AU4', 'AT4', 'CN2', 'CV3',
       'MW2', 'AT7', 'ED1', 'CU3',
       'UG2', 'WD1', 'RH3', 'AT5',
       'MW1', 'GK1', 'CU1', 'AA3',
       'AM1', 'OD3', 'GA1', 'AT2',
       'AI6', 'AT8', 'RH1', 'MW4',
       'AA1', 'AN1', 'AH6', 'CF3',
       'CF1', 'CIG_Ceiling_Determination_Code',
       'AI2', 'CN1', 'CT3', 'GG4',
       'CB1', 'GD2', 'GA4', 'KA1',
       'AD1', 'WA1', 'AW1', 'MH1',
       'KB2', 'KG1', 'AA2', 'GG2',
       'KC1', 'OC1', 'IA1', 'AW5',
       'WND_Direction_Angle', 'IA2', 'GJ1',
       'GA3', 'GD5', 'CR1', 'CF2',
       'DEW_Point_Quality_Code', 'AT1', 'KB1',
       'GD3', 'KC2', 'CV2', 'AL2',
       'AH5', 'KG2', 'ME1', 'AE1',
       'AW6', 'AX1', 'KE1', 'SA1',
       'OB1', 'AZ1', 'MD1', 'AA4',
       'MV2', 'WND_Type_Code',
       'TMP_Air_Temp_Quality_Code', 'GD4',
       'AL1', 'GA6', 'OD1', 'AB1',
       'CW1', 'AH4', 'SLP_Sea_Level_Pres',
       'AX2', 'IB2', 'IB1', 'AX4',
       'WND_Speed_Rate', 'AT6', 'KB3',
       'MW6', 'AH2', 'GG1', 'AZ2',
       'QUALITY_CONTROL', 'AY2', 'CH1',
       'AU5', 'HL1', 'CIG_CAVOK_code',
       'VIS_Distance_Dimension', 'GH1',
       'WEATHER_STATION', 'AI5', 'GG3',
       'AT3', 'MW3', 'CT2', 'KF1',
       'RH2', 'EQD', 'CI1',
 'VIS_Distance_Quality_Code',
 'VIS_Quality_Variability_Code',
 'WEATHER_SOURCE',
 'WND_Quality_Code',
 'WND_Speed_Quality_Code']

# COMMAND ----------

weather2ColToKeep = ['WEATHER2_GN1', 'WEATHER2_GF1', 'WEATHER2_UA1', 'WEATHER2_AU2',
       'WEATHER2_AX5', 'WEATHER2_TMP_Air_Temp', 'WEATHER2_MK1',
       'WEATHER2_CN3', 'WEATHER2_GM1', 'WEATHER2_GA2',
       'WEATHER2_SLP_Sea_Level_Pres_Quality_Code', 'WEATHER2_AW7',
       'WEATHER2_MG1', 'WEATHER2_CG3', 'WEATHER2_VIS_Variability_Code',
       'WEATHER2_GO1', 'WEATHER2_AL3', 'WEATHER2_AI1', 'WEATHER2_MF1',
       'WEATHER2_WEATHER_KEY', 'WEATHER2_KA3', 'WEATHER2_AI4',
       'WEATHER2_AK1', 'WEATHER2_OE3', 'WEATHER2_AW2', 'WEATHER2_REM',
       'WEATHER2_OD2', 'WEATHER2_CN4', 'WEATHER2_AO1', 'WEATHER2_CO1',
       'WEATHER2_OE2', 'WEATHER2_CG1', 'WEATHER2_AX6', 'WEATHER2_KA2',
       'WEATHER2_CU2', 'WEATHER2_AH3', 'WEATHER2_OE1', 'WEATHER2_MA1',
       'WEATHER2_CT1', 'WEATHER2_AW4', 'WEATHER2_AU3', 'WEATHER2_GA5',
       'WEATHER2_UG1', 'WEATHER2_GE1', 'WEATHER2_AI3', 'WEATHER2_GD1',
       'WEATHER2_CIG_Ceiling_Quality_Code', 'WEATHER2_AX3',
       'WEATHER2_KD1', 'WEATHER2_CIG_Ceiling_Height_Dimension',
       'WEATHER2_AW3', 'WEATHER2_CV1', 'WEATHER2_AY1', 'WEATHER2_MV1',
       'WEATHER2_KA4', 'WEATHER2_AJ1', 'WEATHER2_WEATHER_DATE',
       'WEATHER2_CG2', 'WEATHER2_KD2', 'WEATHER2_DEW_Point_Temp',
       'WEATHER2_AH1', 'WEATHER2_AU1', 'WEATHER2_GL1', 'WEATHER2_MW5',
       'WEATHER2_AU4', 'WEATHER2_AT4', 'WEATHER2_CN2', 'WEATHER2_CV3',
       'WEATHER2_MW2', 'WEATHER2_AT7', 'WEATHER2_ED1', 'WEATHER2_CU3',
       'WEATHER2_UG2', 'WEATHER2_WD1', 'WEATHER2_RH3', 'WEATHER2_AT5',
       'WEATHER2_MW1', 'WEATHER2_GK1', 'WEATHER2_CU1', 'WEATHER2_AA3',
       'WEATHER2_AM1', 'WEATHER2_OD3', 'WEATHER2_GA1', 'WEATHER2_AT2',
       'WEATHER2_AI6', 'WEATHER2_AT8', 'WEATHER2_RH1', 'WEATHER2_MW4',
       'WEATHER2_AA1', 'WEATHER2_AN1', 'WEATHER2_AH6', 'WEATHER2_CF3',
       'WEATHER2_CF1', 'WEATHER2_CIG_Ceiling_Determination_Code',
       'WEATHER2_AI2', 'WEATHER2_CN1', 'WEATHER2_CT3', 'WEATHER2_GG4',
       'WEATHER2_CB1', 'WEATHER2_GD2', 'WEATHER2_GA4', 'WEATHER2_KA1',
       'WEATHER2_AD1', 'WEATHER2_WA1', 'WEATHER2_AW1', 'WEATHER2_MH1',
       'WEATHER2_KB2', 'WEATHER2_KG1', 'WEATHER2_AA2', 'WEATHER2_GG2',
       'WEATHER2_KC1', 'WEATHER2_OC1', 'WEATHER2_IA1', 'WEATHER2_AW5',
       'WEATHER2_WND_Direction_Angle', 'WEATHER2_IA2', 'WEATHER2_GJ1',
       'WEATHER2_GA3', 'WEATHER2_GD5', 'WEATHER2_CR1', 'WEATHER2_CF2',
       'WEATHER2_DEW_Point_Quality_Code', 'WEATHER2_AT1', 'WEATHER2_KB1',
       'WEATHER2_GD3', 'WEATHER2_KC2', 'WEATHER2_CV2', 'WEATHER2_AL2',
       'WEATHER2_AH5', 'WEATHER2_KG2', 'WEATHER2_ME1', 'WEATHER2_AE1',
       'WEATHER2_AW6', 'WEATHER2_AX1', 'WEATHER2_KE1', 'WEATHER2_SA1',
       'WEATHER2_OB1', 'WEATHER2_AZ1', 'WEATHER2_MD1', 'WEATHER2_AA4',
       'WEATHER2_MV2', 'WEATHER2_WND_Type_Code',
       'WEATHER2_TMP_Air_Temp_Quality_Code', 'WEATHER2_GD4',
       'WEATHER2_AL1', 'WEATHER2_GA6', 'WEATHER2_OD1', 'WEATHER2_AB1',
       'WEATHER2_CW1', 'WEATHER2_AH4', 'WEATHER2_SLP_Sea_Level_Pres',
       'WEATHER2_AX2', 'WEATHER2_IB2', 'WEATHER2_IB1', 'WEATHER2_AX4',
       'WEATHER2_WND_Speed_Rate', 'WEATHER2_AT6', 'WEATHER2_KB3',
       'WEATHER2_MW6', 'WEATHER2_AH2', 'WEATHER2_GG1', 'WEATHER2_AZ2',
       'WEATHER2_QUALITY_CONTROL', 'WEATHER2_AY2', 'WEATHER2_CH1',
       'WEATHER2_AU5', 'WEATHER2_HL1', 'WEATHER2_CIG_CAVOK_code',
       'WEATHER2_VIS_Distance_Dimension', 'WEATHER2_GH1',
       'WEATHER2_WEATHER_STATION', 'WEATHER2_AI5', 'WEATHER2_GG3',
       'WEATHER2_AT3', 'WEATHER2_MW3', 'WEATHER2_CT2', 'WEATHER2_KF1',
       'WEATHER2_RH2', 'WEATHER2_EQD', 'WEATHER2_CI1',
 'WEATHER2_VIS_Distance_Quality_Code',
 'WEATHER2_VIS_Quality_Variability_Code',
 'WEATHER2_WEATHER_SOURCE',
 'WEATHER2_WND_Quality_Code',
 'WEATHER2_WND_Speed_Quality_Code']

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



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
        fig, ax = plt.subplots(nrows=round(len(self.df.columns)), ncols=2, figsize=(16,5*round(len(self.df.columns))))
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

analyzer = Analyze(sample_panda_df[colToKeep])
analyzer.print_eda_summary()

# COMMAND ----------

print(weather1ColToKeep)

# COMMAND ----------

cols  = weather1ColToKeep.append('DEP_DEL15')
analyzer = Analyze(sample_panda_df[cols])
analyzer.print_eda_summary()

# COMMAND ----------

analyzer = Analyze(sample_panda_df[weather1ColToKeep.append('DEP_DEL15')])
analyzer.print_eda_summary()

# COMMAND ----------

sample_panda_df

# COMMAND ----------

# MAGIC %md **Note that above must be moved after feature engineering** 

# COMMAND ----------

# MAGIC %md # Data Wrangling and Cleaning

# COMMAND ----------

def clean_data(df):  
    
    #
    # drop useless columns based on preliminary analysis
    #
    numericColumns = 
    df = df.drop(columns=['XXX','YYY'])
    
    #
    # get rid of hex char codes, keep the actual code number
    #
    char_treatment = [
        'weird character field 1',
        'weird character field 2'
    ]
    
    print("-- replacing weird characters ...")
    for col in char_treatment:
        if df[col].dtype.name == 'object':
            df[col] = df[col].str.replace(r'&#x(\d\d);', '\1', regex=True)
            df[col] = df[col].str.replace(r'[\x00-\x1f]', '', regex=True)
        else:
            print("col", col,"flagged for char replacement is not of type string")
            

   # make all strings lower case
    case_treatment = [
        'string field 1',
        'string field 2'
    ]
   
    print("-- lower-casing where appropriate ...")
    for col in case_treatment:
        if df[col].dtype.name == 'object':
            df[col] = df[col].str.lower()
             
    #
    # Fix categorization discrepencies
    #           
    df.replace({'SmartScreen':
        {'Enabled':'on',
         'RequiredAdmin':'requireadmin',
         'of':'off',
         'Promt':'prompt',
         'Promprt':'prompt'}})
            
    #
    # make strings into categories
    #
    categories = [
        'SmartScreen',
        'Census_InternalBatteryType',
        'Census_ChassisTypeName',
        'Census_OSEdition',
        'Census_PowerPlatformRoleName',
        'OsBuildLab'
    ]
    
    print("-- making categories from strings that needed massaging ...")
    for col in categories:
        df[col] = df[col].astype('category')

        
    #
    # add 'unknown' categories where necessary and replace the NAs
    # ADD COLUMNS NAMES HERE TO HAVE THEIR CATEGORIES AUGMENTED AND NAS FILLED WITH 'unknown'
    #
    
    categories = [
        'SmartScreen',
        'Census_PrimaryDiskTypeName',  # ['HDD' 'SSD' 'UNKNOWN' 'Unspecified']
        'Census_InternalBatteryType',
        'Census_OSEdition',
        'Census_PowerPlatformRoleName', # also had 'unknown' as well as Nas
        'Census_GenuineStateName',       # and this one too
        'Census_ChassisTypeName'
        
    ]

    print("-- adding categories ..")
    for col in categories:
        print("   ", col)
        if 'unknown' not in df[col].cat.categories:
            df[col].cat.add_categories(['unknown'], inplace=True)
        df[col].fillna('unknown', inplace=True)
    # add one manually because it needs a special unknown value
    df["OsBuildLab"].cat.add_categories(["0.0.-.-.0-0"], inplace=True)
    df["OsBuildLab"].fillna("0.0.-.-.0-0", inplace=True)
    # and this one already had some 'unknown' values
    #df['Census_ChassisTypeName'].fillna('unknown', inplace=True)


    #
    # flag and fill selected NAs
    # ADD COLUMN NAMES HERE IN nafill TO HAVE COLUMNS FLAGGED AND FILLED WITH PROVIDED VALUES
    #   
    print("-- replacing selected NA values")
    nafill = {
        "RtpStateBitfield":0,
        "DefaultBrowsersIdentifier":0,
        "AVProductStatesIdentifier":0,
        "AVProductsInstalled":0,
        "AVProductsEnabled":0,
        "CityIdentifier":0,
        "OrganizationIdentifier":0,
        "GeoNameIdentifier":0,
        "IsProtected":0,
        "SMode":0,
        "IeVerIdentifier":0,
        "Firewall":0,
        "UacLuaenable":0,
        "Census_OEMNameIdentifier":0,
        "Census_OEMModelIdentifier":0,
        "Census_ProcessorCoreCount":0,
        "Census_ProcessorManufacturerIdentifier":0,
        "Census_ProcessorModelIdentifier":0,
        "Census_PrimaryDiskTotalCapacity":0,
        "Census_SystemVolumeTotalCapacity":0,
        "Census_TotalPhysicalRAM":0,
        "Census_InternalPrimaryDiagonalDisplaySizeInInches":0,
        "Census_InternalPrimaryDisplayResolutionHorizontal":0,
        "Census_InternalPrimaryDisplayResolutionVertical":0,
        "Census_InternalBatteryNumberOfCharges":0,
        "Census_OSInstallLanguageIdentifier":0,
        "Census_IsFlightingInternal":0,
        "Census_IsFlightsDisabled":0,
        "Census_ThresholdOptIn":0,
        "Census_FirmwareManufacturerIdentifier":0,
        "Census_IsWIMBootEnabled":0,
        "Census_IsVirtualDevice":0,
        "Census_IsAlwaysOnAlwaysConnectedCapable":0,
        "Wdft_IsGamer":0,
        "Wdft_RegionIdentifier":0,
        "Census_FirmwareVersionIdentifier":0
    }

    for col in nafill:
        df[col+'_wasna'] = df[col].isna()
    df.fillna(value=nafill, inplace=True)
    

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

def count_missings(df):
    miss_counts = list()
    for c in df.columns:
        if df.where(f.col(c).isNull()).count() > 0:
            tup = (c,int(df.where(f.col(c).isNull()).count()))
            miss_counts.append(tup)
    return miss_counts


# COMMAND ----------

missing_counts = count_missings(flightsCache)

# COMMAND ----------

missing_counts

# COMMAND ----------

list_cols_miss=[x[0] for x in missing_counts]
df_miss= flightsCache.select(*list_cols_miss)
#categorical columns
catcolums_miss=[item[0] for item in df_miss.dtypes if item[1].startswith('string')]  #will select name of column with string data type
print("\ncateogrical columns with missing data:", catcolums_miss)
### numerical columns
numcolumns_miss = [item[0] for item in df_miss.dtypes if item[1].startswith('int') | item[1].startswith('double')] #will select name of column with integer or double data type
print("\nnumerical columns with missing data:", numcolumns_miss)

# COMMAND ----------

# fill in the missing categorical values with the most frequent category 

cleaned_airlines_sample = flightsCache

df_Nomiss=cleaned_airlines_sample.na.drop()
for x in catcolums_miss:                  
  mode=df_Nomiss.groupBy(x).count().sort(f.col("count").desc()).collect()
  if mode:
    print(x, mode[0][0]) #print name of columns and it's most categories 
    cleaned_airlines_sample = cleaned_airlines_sample.na.fill({x:mode[0][0]})
    
# fill the missing numerical values with the average of each #column
for i in numcolumns_miss:
  meanvalue = cleaned_airlines_sample.select(f.round(f.mean(i))).collect()
  if meanvalue:
    print(i, meanvalue[0][0]) 
    cleaned_airlines_sample=cleaned_airlines_sample.na.fill({i:meanvalue[0][0]})

# COMMAND ----------

# Use the OneHotEncoderEstimator to convert categorical features into one-hot vectors
# Use VectorAssembler to combine vector of one-hots and the numerical features
# Append the process into the stages array to reproduce
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

stages = j
for categoricalCol in cat_cols:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
stages += [stringIndexer, encoder]
assemblerInputs = [c + "classVec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

# set up a pipeline to apply all the stages of transformation
from pyspark.ml import Pipeline
cols = cleaned_airlines_sample.columns
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(cleaned_airlines_sample)
cleaned_airlines_sample = pipelineModel.transform(cleaned_airlines_sample)
selectedCols = ['features']+cols
cleaned_airlines_sample = cleaned_airlines_sample.select(selectedCols)
pd.DataFrame(cleaned_airlines_sample.take(5), columns=cleaned_airlines_sample.columns)

# COMMAND ----------

# MAGIC %md # Feature Engineering

# COMMAND ----------

def process_weather_data(df):
  WND_col = f.split(df['WND'], ',')
  CIG_col = f.split(df['CIG'], ',')
  VIS_col = f.split(df['VIS'], ',')
  TMP_col = f.split(df['TMP'], ',')
  DEW_col = f.split(df['DEW'], ',')
  SLP_col = f.split(df['SLP'], ',')
  df = (df
    .withColumn("STATION", f.lpad(df.STATION, 11, '0'))
    # WND Fields [direction angle, quality code, type code, speed rate, speed quality code]
    .withColumn('WND_Direction_Angle', WND_col.getItem(0).cast('int')) # continuous
    .withColumn('WND_Quality_Code', WND_col.getItem(1).cast('int')) # categorical
    .withColumn('WND_Type_Code', WND_col.getItem(2).cast('string')) # categorical
    .withColumn('WND_Speed_Rate', WND_col.getItem(3).cast('int')) # continuous
    .withColumn('WND_Speed_Quality_Code', WND_col.getItem(4).cast('int')) # categorical
    # CIG Fields
    .withColumn('CIG_Ceiling_Height_Dimension', CIG_col.getItem(0).cast('int')) # continuous 
    .withColumn('CIG_Ceiling_Quality_Code', CIG_col.getItem(1).cast('int')) # categorical
    .withColumn('CIG_Ceiling_Determination_Code', CIG_col.getItem(2).cast('string')) # categorical 
    .withColumn('CIG_CAVOK_code', CIG_col.getItem(3).cast('string')) # categorical/binary
    # VIS Fields
    .withColumn('VIS_Distance_Dimension', VIS_col.getItem(0).cast('int')) # continuous
    .withColumn('VIS_Distance_Quality_Code', VIS_col.getItem(1).cast('int')) # categorical
    .withColumn('VIS_Variability_Code', VIS_col.getItem(2).cast('string')) # categorical/binary
    .withColumn('VIS_Quality_Variability_Code', VIS_col.getItem(3).cast('int')) # categorical
    # TMP Fields
    .withColumn('TMP_Air_Temp', TMP_col.getItem(0).cast('int')) # continuous
    .withColumn('TMP_Air_Temp_Quality_Code', TMP_col.getItem(1).cast('string')) # categorical
    # DEW Fields
    .withColumn('DEW_Point_Temp', DEW_col.getItem(0).cast('int')) # continuous
    .withColumn('DEW_Point_Quality_Code', DEW_col.getItem(1).cast('string')) # categorical
    # SLP Fields
    .withColumn('SLP_Sea_Level_Pres', SLP_col.getItem(0).cast('int')) # continuous
    .withColumn('SLP_Sea_Level_Pres_Quality_Code', SLP_col.getItem(1).cast('int')) # categorical
    # SNOW Fields
    
    .withColumnRenamed("DATE", "WEATHER_DATE")
    .withColumnRenamed("SOURCE", "WEATHER_SOURCE")
    .withColumnRenamed("STATION", "WEATHER_STATION")
       )


  cols = set(df.columns)
  remove_cols = set(['LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'WND', 'CIG','VIS','TMP', 'DEW', 'SLP'])
  cols = list(cols - remove_cols)
  return df.select(cols)
  

weather_processed_df = process_weather_data(weather_raw_df)

# COMMAND ----------

def avgWeatherData(weatherData, timePeriod):
  columns = ['WND_Speed_Rate', 'CIG_Ceiling_Height_Dimension', 'VIS_Distance_Dimension']
  for column in columns:
    weatherAvgData = (weatherData
                      .withColumn(column + '_AVG', f.col(column) / timePeriod)
                      
    )
  
  return weatherAvgData
weatherAvgData = avgWeatherData(weather_processed_df, 3)

# COMMAND ----------

display(weather_raw_df)

# COMMAND ----------

display(weather_processed_df)

# COMMAND ----------

weatherAvgData = spark.sql("""select station, year(DATE) as YEAR, month(DATE) as MONTH,
                            dayofmonth(DATE) as DAY, count(*) as TOTAL
                            AVG(case when substr(WND, 1, 3)) == '999' then null else int(substr(WND, 1, 3)) end) as WND_ANGLE, from weather_raw_df group by station, YEAR, MONTH, DAY;""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT station, year(DATE) AS YEAR, month(DATE) AS MONTH, 
# MAGIC                             dayofmonth(DATE) AS DAY, count(*) AS TOTAL, 
# MAGIC                             AVG(CASE WHEN SUBSTR(WND, 1, 3) == '999' THEN null 
# MAGIC                                      ELSE int(substr(WND, 1, 3)) END) AS WND_ANGLE_AVG,
# MAGIC                             AVG(CASE WHEN SUBSTR(WND, 9, 4) == '9999' THEN null
# MAGIC                                      ELSE int(substr(WND, 9, 4)) END) AS WND_SPEED_AVG,
# MAGIC                             AVG(CASE WHEN SUBSTR(CIG, 1, 5) == '99999' THEN null
# MAGIC                                      ELSE int(substr(CIG, 1, 5)) END) AS CIG_AVG,
# MAGIC                             AVG(CASE WHEN SUBSTR(VIS, 1, 6) == '999999' THEN null
# MAGIC                                      ELSE int(substr(VIS, 1, 6)) END) AS VIS_AVG,
# MAGIC                             AVG(CASE WHEN SUBSTR(DEW, 1, 5) == '+9999' THEN null
# MAGIC                                      ELSE int(substr(DEW, 1, 5)) END) AS DEW_AVG,
# MAGIC                             FROM weather_processed GROUP BY station, YEAR, MONTH, DAY;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT WEATHER_STATION, year(DATE) AS YEAR, month(DATE) AS MONTH, 
# MAGIC                             dayofmonth(DATE) AS DAY, count(*) AS TOTAL, 
# MAGIC                             AVG(CASE WHEN WND_Direction_Angle == 999 THEN null 
# MAGIC                                      ELSE WND_Direction_Angle END) AS WND_ANGLE_AVG,
# MAGIC                             AVG(CASE WHEN WND_Speed_Rate == 9999 THEN null
# MAGIC                                      ELSE WND_Speed_Rate END) AS WND_SPEED_AVG,
# MAGIC                             AVG(CASE WHEN CIG_Ceiling_Height_Dimension == 99999 THEN null
# MAGIC                                      ELSE CIG_Ceiling_Height_Dimension END) AS CIG_AVG,
# MAGIC                             AVG(CASE WHEN VIS_Distance_Dimension == 999999 THEN null
# MAGIC                                      ELSE VIS_Distance_Dimension END) AS VIS_AVG,
# MAGIC                             AVG(CASE WHEN DEW_Point_Temp == +9999 THEN null
# MAGIC                                      ELSE DEW_Point_Temp END) AS DEW_AVG,
# MAGIC                             FROM weather_processed GROUP BY WEATHER_STATION, YEAR, MONTH, DAY;

# COMMAND ----------

# Maybe we do an inner join with airport station id's and weather and then run query below and select on the airport station id.  Then we can use that to go back to our final join 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT WEATHER_STATION, 
# MAGIC        year(WEATHER_DATE) AS YEAR, 
# MAGIC        month(WEATHER_DATE) AS MONTH, 
# MAGIC        dayofmonth(WEATHER_DATE) AS DAY, 
# MAGIC        count(*) AS TOTAL, 
# MAGIC        AVG(CASE WHEN WND_Direction_Angle == 999 THEN null 
# MAGIC            ELSE WND_Direction_Angle END) AS WND_ANGLE_AVG,
# MAGIC        AVG(CASE WHEN WND_Speed_Rate == 9999 THEN null
# MAGIC            ELSE WND_Speed_Rate END) AS WND_SPEED_AVG,
# MAGIC        AVG(CASE WHEN CIG_Ceiling_Height_Dimension == 99999 THEN null
# MAGIC            ELSE CIG_Ceiling_Height_Dimension END) AS CIG_AVG,
# MAGIC        AVG(CASE WHEN VIS_Distance_Dimension == 999999 THEN null
# MAGIC            ELSE VIS_Distance_Dimension END) AS VIS_AVG,
# MAGIC        AVG(CASE WHEN DEW_Point_Temp == 9999 THEN null
# MAGIC            ELSE DEW_Point_Temp END) AS DEW_AVG
# MAGIC FROM weather_processed 
# MAGIC GROUP BY WEATHER_STATION, YEAR, MONTH, DAY;

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

def getHour(date):
    datestr = date.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    date, time = datestr.split("T")
    id_hour_portion = time[:2]
    return id_hour_portion
spark.udf.register("getHour", getHour)

# COMMAND ----------

weather_raw_df.createOrReplaceTempView('weather_raw_df')
# weatherAvgData.createOrReplaceTempView("weather_daily")
# display(weather_daily)

# COMMAND ----------

display(weather_processed)

# COMMAND ----------

def process_weather_data(df):
  WND_col = f.split(df['WND'], ',')
  CIG_col = f.split(df['CIG'], ',')
  VIS_col = f.split(df['VIS'], ',')
  TMP_col = f.split(df['TMP'], ',')
  DEW_col = f.split(df['DEW'], ',')
  SLP_col = f.split(df['SLP'], ',')
  df = (df
    .withColumn("STATION", f.lpad(df.STATION, 11, '0'))
    # WND Fields [direction angle, quality code, type code, speed rate, speed quality code]
    .withColumn('WND_Direction_Angle', WND_col.getItem(0).cast('int')) # continuous
    .withColumn('WND_Quality_Code', WND_col.getItem(1).cast('int')) # categorical
    .withColumn('WND_Type_Code', WND_col.getItem(2).cast('string')) # categorical
    .withColumn('WND_Speed_Rate', WND_col.getItem(3).cast('int')) # categorical
    .withColumn('WND_Speed_Quality_Code', WND_col.getItem(4).cast('int')) # categorical
    # CIG Fields
    .withColumn('CIG_Ceiling_Height_Dimension', CIG_col.getItem(0).cast('int')) # continuous 
    .withColumn('CIG_Ceiling_Quality_Code', CIG_col.getItem(1).cast('int')) # categorical
    .withColumn('CIG_Ceiling_Determination_Code', CIG_col.getItem(2).cast('string')) # categorical 
    .withColumn('CIG_CAVOK_code', CIG_col.getItem(3).cast('string')) # categorical/binary
    # VIS Fields
    .withColumn('VIS_Distance_Dimension', VIS_col.getItem(0).cast('int')) # continuous
    .withColumn('VIS_Distance_Quality_Code', VIS_col.getItem(1).cast('int')) # categorical
    .withColumn('VIS_Variability_Code', VIS_col.getItem(2).cast('string')) # categorical/binary
    .withColumn('VIS_Quality_Variability_Code', VIS_col.getItem(3).cast('int')) # categorical
    # TMP Fields
    .withColumn('TMP_Air_Temp', TMP_col.getItem(0).cast('int')) # continuous
    .withColumn('TMP_Air_Temp_Quality_Code', TMP_col.getItem(1).cast('string')) # categorical
    # DEW Fields
    .withColumn('DEW_Point_Temp', DEW_col.getItem(0).cast('int')) # continuous
    .withColumn('DEW_Point_Quality_Code', DEW_col.getItem(1).cast('string')) # categorical
    # SLP Fields
    .withColumn('SLP_Sea_Level_Pres', SLP_col.getItem(0).cast('int')) # continuous
    .withColumn('SLP_Sea_Level_Pres_Quality_Code', SLP_col.getItem(1).cast('int')) # categorical
    # SNOW Fields
    
    .withColumnRenamed("DATE", "WEATHER_DATE")
    .withColumnRenamed("SOURCE", "WEATHER_SOURCE")
    .withColumnRenamed("STATION", "WEATHER_STATION")
       )


  cols = set(df.columns)
  remove_cols = set(['LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'WND', 'CIG','VIS','TMP', 'DEW', 'SLP'])
  cols = list(cols - remove_cols)
  return df.select(cols)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## PageRank Features

# COMMAND ----------

flights_processed_pr = spark.sql("SELECT * from flights_processed")

# COMMAND ----------

display(flights_processed_pr) 

# COMMAND ----------

!pip install networkx

# COMMAND ----------

import networkx as nx 
# airlineGraph = {'nodes': flights_processed_pr.select('ORIGIN', 'DEST').rdd.flatMap(list).distinct().collect(), 
#                 'edges': flights_processed_pr.select('ORIGIN', 'DEST').rdd.map(tuple).collect()}

# directedGraph = nx.DiGraph()
# directedGraph.add_nodes_from(airlineGraph['nodes'])
# directedGraph.add_edges_from(airlineGraph['edges'])

pageRank = nx.pagerank(directedGraph, alpha = 0.85)
pandasPageRank = pd.DataFrame(pageRank.items(), columns = ['Station', 'PageRank'])
pandasPageRank = spark.createDataFrame(pandasPageRank)
pandasPageRank.createOrReplaceTempView("pandasPageRank")
flights_processed_pr.createOrReplaceTempView("flights_processed_pr")
# Now we want to separate the pagerank for the stations based on destination and origin
flights_processed_pr = spark.sql("SELECT * from flights_processed_pr LEFT JOIN pandasPageRank ON flights_processed_pr.ORIGIN == pandasPageRank.Station").drop('Station')
flights_processed_pr = flights_processed_pr.withColumnRenamed('PageRank', 'PAGERANK_ORIGIN')
flights_processed_pr.createOrReplaceTempView("flights_processed_pr")
# Repeat for Dest
flights_processed_pr = spark.sql("SELECT * from flights_processed_pr LEFT JOIN pandasPageRank ON flights_processed_pr.DEST == pandasPageRank.Station").drop('Station')
flights_processed_pr = flights_processed_pr.withColumnRenamed('PageRank', 'PAGERANK_DEST')
flights_processed_pr.createOrReplaceTempView('flights_processed_pr')
display(flights_processed_pr)

# COMMAND ----------

# MAGIC %md # Feature Engineering

# COMMAND ----------

stages = []
def oneHotEncoder(catCols):
  for categoricalCol in cat_cols:
      stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
      encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
  return catCols

stages += [stringIndexer, encoder]
assemblerInputs = [c + "classVec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

def encodeData():
  numericColumns
  categoricalColumns

# COMMAND ----------

# MAGIC %md
# MAGIC # Split the train/validation/test sets and normalize the data

# COMMAND ----------

# MAGIC %md # Split data into train, test, validation

# COMMAND ----------



# COMMAND ----------

# data distribution across RDD partitions is not idempotent, and could be rearranged or updated during the query execution, thus affecting the output of the randomSplit method
# to resolve the issue, we can repartition, or apply an aggregate function, or we can cache (https://kb.databricks.com/data/random-split-behavior.html)
# also add a unique ID (monotonically_increasing_id)
flightsCache = airlines_sample.withColumn("id", f.monotonically_increasing_id()).cache()

flightsCache = flightsCache.na.drop(subset=["DEP_DEL15"])

# COMMAND ----------

# function to create stratified train, test and validate sets from supplied ratios 
def generate_train_test_validate_sets(train_ratio, test_ratio, df, label, join_on, seed):
    reserved_size = 1-train_ratio
    
    fractions = df.select(label).distinct().withColumn("fraction", f.lit(train_ratio)).rdd.collectAsMap()
    df_train = df.stat.sampleBy(label, fractions, seed)
    df_remaining = df.join(df_train, on=join_on, how="left_anti")
    
    reserved_size = 1 - (test_ratio / reserved_size)

    fractions = df_remaining.select(label).distinct().withColumn("fraction", f.lit(reserved_size)).rdd.collectAsMap()
    df_test = df_remaining.stat.sampleBy(label, fractions, seed)
    df_validate = df_remaining.join(df_test, on=join_on, how="left_anti")
   
    return df_train, df_test, df_validate

# create our train, validate, and test sets for modeling, analysis and final confirmation
df_train, df_test, df_validate = generate_train_test_validate_sets(train_ratio=.8, test_ratio=.1, df=flightsCache, label='DEP_DEL15', join_on="id", seed=42)



