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

!pip install networkx
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
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


# Model Imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder 
from pyspark.ml.classification import LogisticRegression

%matplotlib inline
sqlContext = SQLContext(sc)

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
# MAGIC ### 5. Exploratory Data Analysis
# MAGIC ### 6. Data Wrangling, Cleanup and Prep
# MAGIC ### 7. Model Exploration (Pipeline)
# MAGIC #### a. Feature Selection
# MAGIC #### b. Feature Engineering
# MAGIC #### c. Transformations (Encoding & Scaling)
# MAGIC #### d. Evaluation
# MAGIC ### 8. Model Selection and Tuning
# MAGIC ### 9. Conclusion
# MAGIC ### (10. Application of Course Concepts)

# COMMAND ----------

flights_and_weather_combined_processed = spark.sql("SELECT * FROM flights_and_weather_combined_processed WHERE CANCELLED = 0")

#flights_train = spark.sql("SELECT * FROM flights_and_weather_combined_processed WHERE YEAR IN (2015, 2016, 2017)")
#flights_validate = spark.sql("SELECT * FROM flights_and_weather_combined_processed WHERE YEAR = 2018;")
#flights_test = spark.sql("SELECT * FROM flights_and_weather_combined_processed WHERE YEAR = 2019;")

#flights_train_sample = flights_train.sample(False, 0.01)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM flights_and_weather_combined_processed
# MAGIC WHERE DEP_DEL15 IS NULL AND CRS_DEP_TIME - DEP_TIME == 0 AND CANCELLED = 0

# COMMAND ----------

# MAGIC %md
# MAGIC Assumptions for NULL **DEP_DEL15** values:
# MAGIC * Most are cancelled - drop those rows for now. Will revisit cancellations later.
# MAGIC * If DEP_DEL_15 is NULL and the difference between schedule departure and actual departure is 0 , set DEP_DEL15 -> 0
# MAGIC * If DEP_DEL_15 is NULL and the difference between schedule departure and actual departure is not 0 (5 records), -> drop records

# COMMAND ----------

flights_and_weather_combined_processed = spark.sql("SELECT * FROM flights_and_weather_combined_processed WHERE CANCELLED = 0")

# COMMAND ----------

flights_and_weather_combined_processed.count()

# COMMAND ----------

display(flights_and_weather_combined_processed.select('DEP_DEL15').distinct())

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

flights_and_weather_combined_processed = flights_and_weather_combined_processed.withColumn("DEP_DEL15", f.when(flights_and_weather_combined_processed["DEP_DEL15"].isNull(), fix_missing_dep_del15("DEP_DEL15","CRS_DEP_TIME","DEP_TIME")).otherwise(flights_and_weather_combined_processed["DEP_DEL15"]))

# COMMAND ----------

flights_and_weather_combined_processed.where('DEP_DEL15 IS NULL').count()

# COMMAND ----------

flights_and_weather_combined_processed = flights_and_weather_combined_processed.where('DEP_DEL15 IS NOT NULL')

# COMMAND ----------

f'{flights_train_sample.count():,}'

# COMMAND ----------

display(flights_train_sample)

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
        sns.set(rc={'figure.figsize':(10*2,16*8)})
        #sns.set()
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

# MAGIC %md
# MAGIC # Remaining analysis on fully joined data table

# COMMAND ----------

# review categorical and numerical features:
cat_cols = [item[0] for item in flights_train_sample.dtypes if item[1].startswith('string')] 
print(str(len(cat_cols)) + '  categorical features')
num_cols = [item[0] for item in flights_train_sample.dtypes if item[1].startswith('int') | item[1].startswith('double')][1:]
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

flights_train_sample_pdf = flights_train_sample.toPandas()

# COMMAND ----------

missing_recs = info_missing_table(flights_train_sample_pdf)
missing_recs

# COMMAND ----------

colToDrop = missing_recs[missing_recs['% of Total Values'] > 95].index.values
colToDrop

# COMMAND ----------

colToDrop = ['DIV5_TAIL_NUM', 'DIV4_WHEELS_ON', 'DIV3_AIRPORT',
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
       'LONGEST_ADD_GTIME', 'TOTAL_ADD_GTIME', 'CANCELLATION_CODE']

# COMMAND ----------

cols = flights_train_sample_pdf.columns
colToKeep = []
removeCols = set(colToDrop)
for col in cols:
  if not col in removeCols:
    colToKeep.append(col)

colToKeep    

# COMMAND ----------

colToKeep = ['YEAR', # probably won't be useful unless unseen holdout data not provided to us are randomly sampled from the same timeframe as our current dataset
 'QUARTER', # useful for seasonal effects
 'MONTH', #useful for seasonal effects
 'DAY_OF_WEEK', # may be useful
 'FL_DATE', # captured by previous columns (note DAY_OF_MONTH is far down)
 'OP_UNIQUE_CARRIER', #may be useful if certain carrier are more delayed than others
 'OP_CARRIER_AIRLINE_ID', #redundant
 'OP_CARRIER', #redundant
 'TAIL_NUM', #irrelevant, just the number on the tail
 'OP_CARRIER_FL_NUM', #redundant (just use one metric for origin)
 'ORIGIN_AIRPORT_ID', #redundant
 'ORIGIN_AIRPORT_SEQ_ID', #redundant
 'ORIGIN_CITY_MARKET_ID', #redundant
 'ORIGIN', #useful given certain origin metrics could conicide with certain weather 
 'ORIGIN_CITY_NAME', #useful given city
 'ORIGIN_STATE_ABR', #redundant
 'ORIGIN_STATE_FIPS', #redundant
 'ORIGIN_STATE_NM', #redundant
 'ORIGIN_WAC', #redundant
 'DEST_AIRPORT_ID', #redundant
 'DEST_AIRPORT_SEQ_ID', #redundant
 'DEST_CITY_MARKET_ID',#redundant
 'DEST',#useful given certain destination metrics could conicide with certain weather
 'DEST_CITY_NAME', #redundant
 'DEST_STATE_ABR', #redundant
 'DEST_STATE_FIPS',  #redundant
 'DEST_STATE_NM', #redundant
 'DEST_WAC', #redundant
 'CRS_DEP_TIME', # cant use
 'DEP_TIME', # cant use
 'DEP_DELAY', #target column if treating as regression problem
 'DEP_DELAY_NEW', # maybe
 'DEP_DEL15', #target column
 'DEP_DELAY_GROUP', # maybe
 'DEP_TIME_BLK', # cant use
 'TAXI_OUT',# cant use unless average by oriub airport
 'WHEELS_OFF',# cant use
 'WHEELS_ON',# cant use
 'TAXI_IN', # cant use unless average by dest airport
 'CRS_ARR_TIME',# cant use
 'ARR_TIME',# cant use
 'ARR_DELAY',# cant use
 'ARR_DELAY_NEW',# cant use
 'ARR_DEL15',# cant use
 'ARR_DELAY_GROUP',# cant use
 'ARR_TIME_BLK', # cant use
 'CANCELLED', #redundant
 'DIVERTED', #redundant
 'CRS_ELAPSED_TIME',# cant use
 'ACTUAL_ELAPSED_TIME', # # cant use, but create route avg feature
 'AIR_TIME', # # cant use, but create route avg feature
 'FLIGHTS', #redundant
 'DISTANCE', #might be useful
 'DISTANCE_GROUP', #redundant
 'CARRIER_DELAY',# cant use
 'WEATHER_DELAY',# cant use, unless we use an average for route, consider completeness
 'NAS_DELAY',# cant use, unless we use an average for route, consider completeness
 'SECURITY_DELAY',# cant use, unless we use an average for route, consider completeness
 'LATE_AIRCRAFT_DELAY',# cant use, unless we use an average for route, consider completeness
 'DIV_AIRPORT_LANDINGS', #could be useful for past flights 
 'IN_FLIGHT_AIR_DELAY', # cant use, but create route avg feature
 'DAY_OF_MONTH', #likely not useful
 'CRS_DEP_TIME_HOUR', #can't use
 'IATA_ORIGIN', #can't use
 'NEAREST_STATION_ID_ORIGIN',  #redundant
 'NEAREST_STATION_DIST_ORIGIN', #redundant
 'IATA_DEST', #redundant
 'NEAREST_STATION_ID_DEST', #redundant
 'NEAREST_STATION_DIST_DEST', #redundant
 'IATA', #redundant
 'AIRPORT_TZ_NAME', #redundant
 'FLIGHT_TIME_UTC', #redundant
 'WEATHER_PREDICTION_TIME_UTC', 
 'ORIGIN_WEATHER_KEY', #redundant
 'DEST_WEATHER_KEY'] #redundant 

# COMMAND ----------

addl_col_to_drop = ['YEAR', # probably won't be useful unless unseen holdout data not provided to us are randomly sampled from the same timeframe as our current dataset
 'OP_UNIQUE_CARRIER', #may be useful if certain carrier are more delayed than others
 'OP_CARRIER_AIRLINE_ID', #redundant
 'OP_CARRIER', #redundant
 'OP_CARRIER_FL_NUM', #redundant (just use one metric for origin)
 'ORIGIN_AIRPORT_ID', #redundant
 'ORIGIN_AIRPORT_SEQ_ID', #redundant
 'ORIGIN_CITY_MARKET_ID', #redundant
 'ORIGIN_STATE_ABR', #redundant
 'ORIGIN_STATE_FIPS', #redundant
 'ORIGIN_STATE_NM', #redundant
 'ORIGIN_WAC', #redundant
 'DEST_AIRPORT_ID', #redundant
 'DEST_AIRPORT_SEQ_ID', #redundant
 'DEST_CITY_MARKET_ID',#redundant
 'DEST_STATE_ABR', #redundant
 'DEST_STATE_FIPS',  #redundant
 'DEST_STATE_NM', #redundant
 'DEST_WAC', #redundant
 'CRS_DEP_TIME', # cant use
 'DEP_TIME', # cant use
 'DEP_TIME_BLK', # cant use
 'TAXI_OUT',# cant use unless average by oriub airport
 'WHEELS_OFF',# cant use
 'WHEELS_ON',# cant use
 'TAXI_IN', # cant use unless average by dest airport
 'CRS_ARR_TIME',# cant use
 'ARR_TIME',# cant use
 'ARR_DELAY',# cant use
 'ARR_DELAY_NEW',# cant use
 'ARR_DEL15',# cant use
 'ARR_DELAY_GROUP',# cant use
 'ARR_TIME_BLK', # cant use
 'CANCELLED', #redundant
 'DIVERTED', #redundant
 'CRS_ELAPSED_TIME',# cant use
 'ACTUAL_ELAPSED_TIME', # # cant use, but create route avg feature
 'AIR_TIME', # # cant use, but create route avg feature
 'FLIGHTS', #redundant
 'DISTANCE_GROUP', #redundant
 'CARRIER_DELAY',# cant use
 'WEATHER_DELAY',# cant use, unless we use an average for route, consider completeness
 'NAS_DELAY',# cant use, unless we use an average for route, consider completeness
 'SECURITY_DELAY',# cant use, unless we use an average for route, consider completeness
 'LATE_AIRCRAFT_DELAY',# cant use, unless we use an average for route, consider completeness
 'DIV_AIRPORT_LANDINGS', #could be useful for past flights 
 'IN_FLIGHT_AIR_DELAY', # cant use, but create route avg feature
 'DAY_OF_MONTH', #likely not useful
 'CRS_DEP_TIME_HOUR', #can't use
 'IATA_ORIGIN', #can't use
 'NEAREST_STATION_ID_ORIGIN',  #redundant
 'NEAREST_STATION_DIST_ORIGIN', #redundant
 'IATA_DEST', #redundant
 'NEAREST_STATION_ID_DEST', #redundant
 'NEAREST_STATION_DIST_DEST', #redundant
 'IATA', #redundant
 'AIRPORT_TZ_NAME', #redundant
 'FLIGHT_TIME_UTC', #redundant
 'WEATHER_PREDICTION_TIME_UTC', 
 'ORIGIN_WEATHER_KEY', #redundant
 'DEST_WEATHER_KEY'] #redundant 

# COMMAND ----------

weather1ColToKeep = ['WEATHER1_GN1',
 'WEATHER1_GF1',
 'WEATHER1_UA1',
 'WEATHER1_AU2',
 'WEATHER1_AX5',
 'WEATHER1_TMP_AIR_TEMP',
 'WEATHER1_MK1',
 'WEATHER1_CN3',
 'WEATHER1_GM1',
 'WEATHER1_GA2',
 'WEATHER1_SLP_SEA_LEVEL_PRES_QUALITY_CODE',
 'WEATHER1_AW7',
 'WEATHER1_MG1',
 'WEATHER1_CG3',
 'WEATHER1_VIS_VARIABILITY_CODE',
 'WEATHER1_GO1',
 'WEATHER1_AL3',
 'WEATHER1_AI1',
 'WEATHER1_MF1',
 'WEATHER1_WEATHER_KEY',
 'WEATHER1_KA3',
 'WEATHER1_AI4',
 'WEATHER1_AK1',
 'WEATHER1_OE3',
 'WEATHER1_AW2',
 'WEATHER1_REM',
 'WEATHER1_OD2',
 'WEATHER1_CN4',
 'WEATHER1_AO1',
 'WEATHER1_CO1',
 'WEATHER1_OE2',
 'WEATHER1_CG1',
 'WEATHER1_AX6',
 'WEATHER1_KA2',
 'WEATHER1_CU2',
 'WEATHER1_AH3',
 'WEATHER1_OE1',
 'WEATHER1_MA1',
 'WEATHER1_CT1',
 'WEATHER1_AW4',
 'WEATHER1_AU3',
 'WEATHER1_GA5',
 'WEATHER1_UG1',
 'WEATHER1_GE1',
 'WEATHER1_AI3',
 'WEATHER1_GD1',
 'WEATHER1_CIG_CEILING_QUALITY_CODE',
 'WEATHER1_AX3',
 'WEATHER1_KD1',
 'WEATHER1_CIG_CEILING_HEIGHT_DIMENSION',
 'WEATHER1_AW3',
 'WEATHER1_CV1',
 'WEATHER1_AY1',
 'WEATHER1_MV1',
 'WEATHER1_KA4',
 'WEATHER1_AJ1',
 'WEATHER1_WEATHER_DATE',
 'WEATHER1_CG2',
 'WEATHER1_KD2',
 'WEATHER1_DEW_POINT_TEMP',
 'WEATHER1_AH1',
 'WEATHER1_AU1',
 'WEATHER1_GL1',
 'WEATHER1_MW5',
 'WEATHER1_AU4',
 'WEATHER1_AT4',
 'WEATHER1_CN2',
 'WEATHER1_CV3',
 'WEATHER1_MW2',
 'WEATHER1_AT7',
 'WEATHER1_ED1',
 'WEATHER1_CU3',
 'WEATHER1_UG2',
 'WEATHER1_WD1',
 'WEATHER1_RH3',
 'WEATHER1_AT5',
 'WEATHER1_MW1',
 'WEATHER1_GK1',
 'WEATHER1_CU1',
 'WEATHER1_AA3',
 'WEATHER1_AM1',
 'WEATHER1_OD3',
 'WEATHER1_GA1',
 'WEATHER1_AT2',
 'WEATHER1_AI6',
 'WEATHER1_AT8',
 'WEATHER1_RH1',
 'WEATHER1_MW4',
 'WEATHER1_AA1',
 'WEATHER1_AN1',
 'WEATHER1_AH6',
 'WEATHER1_CF3',
 'WEATHER1_CF1',
 'WEATHER1_CIG_CEILING_DETERMINATION_CODE',
 'WEATHER1_AI2',
 'WEATHER1_CN1',
 'WEATHER1_CT3',
 'WEATHER1_GG4',
 'WEATHER1_CB1',
 'WEATHER1_GD2',
 'WEATHER1_GA4',
 'WEATHER1_KA1',
 'WEATHER1_AD1',
 'WEATHER1_WA1',
 'WEATHER1_AW1',
 'WEATHER1_MH1',
 'WEATHER1_KB2',
 'WEATHER1_KG1',
 'WEATHER1_AA2',
 'WEATHER1_GG2',
 'WEATHER1_KC1',
 'WEATHER1_OC1',
 'WEATHER1_IA1',
 'WEATHER1_AW5',
 'WEATHER1_WND_DIRECTION_ANGLE',
 'WEATHER1_IA2',
 'WEATHER1_GJ1',
 'WEATHER1_GA3',
 'WEATHER1_GD5',
 'WEATHER1_CR1',
 'WEATHER1_CF2',
 'WEATHER1_DEW_POINT_QUALITY_CODE',
 'WEATHER1_AT1',
 'WEATHER1_KB1',
 'WEATHER1_GD3',
 'WEATHER1_KC2',
 'WEATHER1_CV2',
 'WEATHER1_AL2',
 'WEATHER1_AH5',
 'WEATHER1_KG2',
 'WEATHER1_ME1',
 'WEATHER1_AE1',
 'WEATHER1_AW6',
 'WEATHER1_AX1',
 'WEATHER1_KE1',
 'WEATHER1_SA1',
 'WEATHER1_OB1',
 'WEATHER1_AZ1',
 'WEATHER1_MD1',
 'WEATHER1_AA4',
 'WEATHER1_MV2',
 'WEATHER1_WND_TYPE_CODE',
 'WEATHER1_TMP_AIR_TEMP_QUALITY_CODE',
 'WEATHER1_GD4',
 'WEATHER1_AL1',
 'WEATHER1_GA6',
 'WEATHER1_OD1',
 'WEATHER1_AB1',
 'WEATHER1_CW1',
 'WEATHER1_AH4',
 'WEATHER1_SLP_SEA_LEVEL_PRES',
 'WEATHER1_AX2',
 'WEATHER1_IB2',
 'WEATHER1_IB1',
 'WEATHER1_AX4',
 'WEATHER1_WND_SPEED_RATE',
 'WEATHER1_AT6',
 'WEATHER1_KB3',
 'WEATHER1_MW6',
 'WEATHER1_AH2',
 'WEATHER1_GG1',
 'WEATHER1_AZ2',
 'WEATHER1_QUALITY_CONTROL',
 'WEATHER1_AY2',
 'WEATHER1_CH1',
 'WEATHER1_AU5',
 'WEATHER1_HL1',
 'WEATHER1_CIG_CAVOK_CODE',
 'WEATHER1_VIS_DISTANCE_DIMENSION',
 'WEATHER1_GH1',
 'WEATHER1_WEATHER_STATION',
 'WEATHER1_AI5',
 'WEATHER1_GG3',
 'WEATHER1_AT3',
 'WEATHER1_MW3',
 'WEATHER1_CT2',
 'WEATHER1_KF1',
 'WEATHER1_RH2',
 'WEATHER1_EQD',
 'WEATHER1_CI1',
 'WEATHER1_VIS_DISTANCE_QUALITY_CODE',
 'WEATHER1_VIS_QUALITY_VARIABILITY_CODE',
 'WEATHER1_WEATHER_SOURCE',
 'WEATHER1_WND_QUALITY_CODE',
 'WEATHER1_WND_SPEED_QUALITY_CODE']

# COMMAND ----------

weather2ColToKeep = ['WEATHER2_GN1',
 'WEATHER2_GF1',
 'WEATHER2_UA1',
 'WEATHER2_AU2',
 'WEATHER2_AX5',
 'WEATHER2_TMP_AIR_TEMP',
 'WEATHER2_MK1',
 'WEATHER2_CN3',
 'WEATHER2_GM1',
 'WEATHER2_GA2',
 'WEATHER2_SLP_SEA_LEVEL_PRES_QUALITY_CODE',
 'WEATHER2_AW7',
 'WEATHER2_MG1',
 'WEATHER2_CG3',
 'WEATHER2_VIS_VARIABILITY_CODE',
 'WEATHER2_GO1',
 'WEATHER2_AL3',
 'WEATHER2_AI1',
 'WEATHER2_MF1',
 'WEATHER2_WEATHER_KEY',
 'WEATHER2_KA3',
 'WEATHER2_AI4',
 'WEATHER2_AK1',
 'WEATHER2_OE3',
 'WEATHER2_AW2',
 'WEATHER2_REM',
 'WEATHER2_OD2',
 'WEATHER2_CN4',
 'WEATHER2_AO1',
 'WEATHER2_CO1',
 'WEATHER2_OE2',
 'WEATHER2_CG1',
 'WEATHER2_AX6',
 'WEATHER2_KA2',
 'WEATHER2_CU2',
 'WEATHER2_AH3',
 'WEATHER2_OE1',
 'WEATHER2_MA1',
 'WEATHER2_CT1',
 'WEATHER2_AW4',
 'WEATHER2_AU3',
 'WEATHER2_GA5',
 'WEATHER2_UG1',
 'WEATHER2_GE1',
 'WEATHER2_AI3',
 'WEATHER2_GD1',
 'WEATHER2_CIG_CEILING_QUALITY_CODE',
 'WEATHER2_AX3',
 'WEATHER2_KD1',
 'WEATHER2_CIG_CEILING_HEIGHT_DIMENSION',
 'WEATHER2_AW3',
 'WEATHER2_CV1',
 'WEATHER2_AY1',
 'WEATHER2_MV1',
 'WEATHER2_KA4',
 'WEATHER2_AJ1',
 'WEATHER2_WEATHER_DATE',
 'WEATHER2_CG2',
 'WEATHER2_KD2',
 'WEATHER2_DEW_POINT_TEMP',
 'WEATHER2_AH1',
 'WEATHER2_AU1',
 'WEATHER2_GL1',
 'WEATHER2_MW5',
 'WEATHER2_AU4',
 'WEATHER2_AT4',
 'WEATHER2_CN2',
 'WEATHER2_CV3',
 'WEATHER2_MW2',
 'WEATHER2_AT7',
 'WEATHER2_ED1',
 'WEATHER2_CU3',
 'WEATHER2_UG2',
 'WEATHER2_WD1',
 'WEATHER2_RH3',
 'WEATHER2_AT5',
 'WEATHER2_MW1',
 'WEATHER2_GK1',
 'WEATHER2_CU1',
 'WEATHER2_AA3',
 'WEATHER2_AM1',
 'WEATHER2_OD3',
 'WEATHER2_GA1',
 'WEATHER2_AT2',
 'WEATHER2_AI6',
 'WEATHER2_AT8',
 'WEATHER2_RH1',
 'WEATHER2_MW4',
 'WEATHER2_AA1',
 'WEATHER2_AN1',
 'WEATHER2_AH6',
 'WEATHER2_CF3',
 'WEATHER2_CF1',
 'WEATHER2_CIG_CEILING_DETERMINATION_CODE',
 'WEATHER2_AI2',
 'WEATHER2_CN1',
 'WEATHER2_CT3',
 'WEATHER2_GG4',
 'WEATHER2_CB1',
 'WEATHER2_GD2',
 'WEATHER2_GA4',
 'WEATHER2_KA1',
 'WEATHER2_AD1',
 'WEATHER2_WA1',
 'WEATHER2_AW1',
 'WEATHER2_MH1',
 'WEATHER2_KB2',
 'WEATHER2_KG1',
 'WEATHER2_AA2',
 'WEATHER2_GG2',
 'WEATHER2_KC1',
 'WEATHER2_OC1',
 'WEATHER2_IA1',
 'WEATHER2_AW5',
 'WEATHER2_WND_DIRECTION_ANGLE',
 'WEATHER2_IA2',
 'WEATHER2_GJ1',
 'WEATHER2_GA3',
 'WEATHER2_GD5',
 'WEATHER2_CR1',
 'WEATHER2_CF2',
 'WEATHER2_DEW_POINT_QUALITY_CODE',
 'WEATHER2_AT1',
 'WEATHER2_KB1',
 'WEATHER2_GD3',
 'WEATHER2_KC2',
 'WEATHER2_CV2',
 'WEATHER2_AL2',
 'WEATHER2_AH5',
 'WEATHER2_KG2',
 'WEATHER2_ME1',
 'WEATHER2_AE1',
 'WEATHER2_AW6',
 'WEATHER2_AX1',
 'WEATHER2_KE1',
 'WEATHER2_SA1',
 'WEATHER2_OB1',
 'WEATHER2_AZ1',
 'WEATHER2_MD1',
 'WEATHER2_AA4',
 'WEATHER2_MV2',
 'WEATHER2_WND_TYPE_CODE',
 'WEATHER2_TMP_AIR_TEMP_QUALITY_CODE',
 'WEATHER2_GD4',
 'WEATHER2_AL1',
 'WEATHER2_GA6',
 'WEATHER2_OD1',
 'WEATHER2_AB1',
 'WEATHER2_CW1',
 'WEATHER2_AH4',
 'WEATHER2_SLP_SEA_LEVEL_PRES',
 'WEATHER2_AX2',
 'WEATHER2_IB2',
 'WEATHER2_IB1',
 'WEATHER2_AX4',
 'WEATHER2_WND_SPEED_RATE',
 'WEATHER2_AT6',
 'WEATHER2_KB3',
 'WEATHER2_MW6',
 'WEATHER2_AH2',
 'WEATHER2_GG1',
 'WEATHER2_AZ2',
 'WEATHER2_QUALITY_CONTROL',
 'WEATHER2_AY2',
 'WEATHER2_CH1',
 'WEATHER2_AU5',
 'WEATHER2_HL1',
 'WEATHER2_CIG_CAVOK_CODE',
 'WEATHER2_VIS_DISTANCE_DIMENSION',
 'WEATHER2_GH1',
 'WEATHER2_WEATHER_STATION',
 'WEATHER2_AI5',
 'WEATHER2_GG3',
 'WEATHER2_AT3',
 'WEATHER2_MW3',
 'WEATHER2_CT2',
 'WEATHER2_KF1',
 'WEATHER2_RH2',
 'WEATHER2_EQD',
 'WEATHER2_CI1',
 'WEATHER2_VIS_DISTANCE_QUALITY_CODE',
 'WEATHER2_VIS_QUALITY_VARIABILITY_CODE',
 'WEATHER2_WEATHER_SOURCE',
 'WEATHER2_WND_QUALITY_CODE',
 'WEATHER2_WND_SPEED_QUALITY_CODE']

# COMMAND ----------

alias_string = ''
for col in weather1ColToKeep:
  alias_string += f"w.{col} AS WEATHER2_{col.split('WEATHER1_')[1]}, "
alias_string = alias_string[:-2]
print(alias_string)

# COMMAND ----------

# Which weather line items to keep
weather2BaselineCol_final = [
 'WEATHER2_TMP_AIR_TEMP',
  'WEATHER2_VIS_VARIABILITY_CODE', # categorical
 'WEATHER2_CIG_CEILING_HEIGHT_DIMENSION',
'WEATHER2_DEW_POINT_TEMP',
 'WEATHER2_CIG_CEILING_DETERMINATION_CODE', # categorical
'WEATHER2_WND_DIRECTION_ANGLE',
 'WEATHER2_WND_TYPE_CODE',
 'WEATHER2_SLP_SEA_LEVEL_PRES',
 'WEATHER2_WND_SPEED_RATE',
 'WEATHER2_CIG_CAVOK_CODE', #categorial
 'WEATHER2_VIS_DISTANCE_DIMENSION',
]
'''
weather2BaselineCol_final = [
 'WEATHER2_TMP_AIR_TEMP',
 'WEATHER2_SLP_SEA_LEVEL_PRES_QUALITY_CODE', # categorical
  'WEATHER2_VIS_VARIABILITY_CODE', # categorical
 'WEATHER2_WEATHER_KEY',
 'WEATHER2_CIG_CEILING_QUALITY_CODE', # categorical
 'WEATHER2_CIG_CEILING_HEIGHT_DIMENSION',
'WEATHER2_DEW_POINT_TEMP',
 'WEATHER2_CIG_CEILING_DETERMINATION_CODE', # categorical
'WEATHER2_WND_DIRECTION_ANGLE',
 'WEATHER2_DEW_POINT_QUALITY_CODE', # categorical
 'WEATHER2_WND_TYPE_CODE',
 'WEATHER2_TMP_AIR_TEMP_QUALITY_CODE', # categorical
 'WEATHER2_SLP_SEA_LEVEL_PRES',
 'WEATHER2_WND_SPEED_RATE',
 'WEATHER2_QUALITY_CONTROL',
 'WEATHER2_CIG_CAVOK_CODE',
 'WEATHER2_VIS_DISTANCE_DIMENSION',
'WEATHER2_WEATHER_STATION',
'WEATHER2_VIS_DISTANCE_QUALITY_CODE', # categorical
'WEATHER2_VIS_QUALITY_VARIABILITY_CODE', # categorical
'WEATHER2_WEATHER_SOURCE',
'WEATHER2_WND_QUALITY_CODE', # categorical
 'WEATHER2_WND_SPEED_QUALITY_CODE' # categorical
]
''''''

# COMMAND ----------

final_cols_to_keep = [
 'QUARTER', 
 'MONTH', 
 'DAY_OF_WEEK',
 'OP_UNIQUE_CARRIER',
 'TAIL_NUM',
 'ORIGIN',
 'DEST',
 'DEP_DELAY', 
 'DEP_DELAY_NEW', 
 'DEP_DEL15', 
 'ACTUAL_ELAPSED_TIME', 
 'AIR_TIME',
 'DISTANCE',
 'CARRIER_DELAY',
 'WEATHER_DELAY',
 'NAS_DELAY',
 'SECURITY_DELAY',
 'LATE_AIRCRAFT_DELAY',
 'IN_FLIGHT_AIR_DELAY', 
 'WEATHER1_TMP_AIR_TEMP',
 'WEATHER1_VIS_VARIABILITY_CODE', # categorical
 'WEATHER1_CIG_CEILING_HEIGHT_DIMENSION',
 'WEATHER1_DEW_POINT_TEMP',
 'WEATHER1_CIG_CEILING_DETERMINATION_CODE', # categorical
 'WEATHER1_WND_DIRECTION_ANGLE',
 'WEATHER1_WND_TYPE_CODE',
 'WEATHER1_SLP_SEA_LEVEL_PRES',
 'WEATHER1_WND_SPEED_RATE',
 'WEATHER1_CIG_CAVOK_CODE', #categorial
 'WEATHER1_VIS_DISTANCE_DIMENSION',
 'WEATHER2_TMP_AIR_TEMP',
 'WEATHER2_VIS_VARIABILITY_CODE', # categorical
 'WEATHER2_CIG_CEILING_HEIGHT_DIMENSION',
 'WEATHER2_DEW_POINT_TEMP',
 'WEATHER2_CIG_CEILING_DETERMINATION_CODE', # categorical
 'WEATHER2_WND_DIRECTION_ANGLE',
 'WEATHER2_WND_TYPE_CODE',
 'WEATHER2_SLP_SEA_LEVEL_PRES',
 'WEATHER2_WND_SPEED_RATE',
 'WEATHER2_CIG_CAVOK_CODE', #categorial
 'WEATHER2_VIS_DISTANCE_DIMENSION'] 

# COMMAND ----------

sample_df_weather = spark.sql("SELECT * FROM weather_processed").sample(False, 0.01)

# COMMAND ----------

# ['WEATHER1_CIG_CEILING_DETERMINATION_CODE', 'WEATHER1_VIS_QUALITY_VARIABILITY_CODE', 'WEATHER1_TMP_AIR_TEMP', 'WEATHER1_CG2', 'WEATHER1_WEATHER_DATE', 'WEATHER1_GE1', 'WEATHER1_TMP_AIR_TEMP_QUALITY_CODE', 'WEATHER1_VIS_DISTANCE_QUALITY_CODE', 'WEATHER1_WND_SPEED_QUALITY_CODE', 'WEATHER1_CIG_CEILING_HEIGHT_DIMENSION', 'WEATHER1_GD1', 'WEATHER1_CIG_CEILING_QUALITY_CODE', 'WEATHER1_WND_SPEED_RATE', 'WEATHER1_SLP_SEA_LEVEL_PRES_QUALITY_CODE', 'WEATHER1_CT1', 'WEATHER1_CG1', 'WEATHER1_CO1', 'WEATHER1_CW1', 'WEATHER1_WND_TYPE_CODE', 'WEATHER1_WND_QUALITY_CODE', 'WEATHER1_VIS_VARIABILITY_CODE', 'WEATHER1_WEATHER_SOURCE', 'WEATHER1_DEW_POINT_TEMP', 'WEATHER1_VIS_DISTANCE_DIMENSION', 'WEATHER1_WEATHER_KEY']
display(sample_df_weather)

# COMMAND ----------

newColToKeep = ['ACTUAL_ELAPSED_TIME',
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

from copy import deepcopy
cols = weather1ColToKeep[:len(weather1ColToKeep)//2]
cols.append('DEP_DEL15')
analyze_df = deepcopy(flights_train_sample_pdf[cols])
analyzer = Analyze(analyze_df)
analyzer.print_eda_summary()


# analyzer = Analyze(flights_train_sample_pdf[colToKeep])
# analyzer.print_eda_summary()

#analyzer = Analyze(flights_train_sample_pdf[colToKeep])
#analyzer.print_eda_summary()

# COMMAND ----------

from copy import deepcopy
cols = weather1ColToKeep[len(weather1ColToKeep)//2:]
cols.append('DEP_DEL15')
analyze_df = deepcopy(flights_train_sample_pdf[cols])
analyzer = Analyze(analyze_df)
analyzer.print_eda_summary()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Data Wrangling and Cleaning

# COMMAND ----------

def clean_data(df):  
    
    final_cols_to_keep = ['YEAR','QUARTER','MONTH', 'DAY_OF_WEEK', 'DAY_OF_MONTH','OP_UNIQUE_CARRIER', 
       'ORIGIN', 'DEST', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'ACTUAL_ELAPSED_TIME', 
       'AIR_TIME','DISTANCE','CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY',
       'LATE_AIRCRAFT_DELAY','IN_FLIGHT_AIR_DELAY', 'WEATHER1_TMP_AIR_TEMP','WEATHER1_VIS_VARIABILITY_CODE', 
       'WEATHER1_CIG_CEILING_HEIGHT_DIMENSION','WEATHER1_DEW_POINT_TEMP','WEATHER1_CIG_CEILING_DETERMINATION_CODE',
       'WEATHER1_WND_DIRECTION_ANGLE','WEATHER1_WND_TYPE_CODE','WEATHER1_SLP_SEA_LEVEL_PRES','WEATHER1_WND_SPEED_RATE',
       'WEATHER1_CIG_CAVOK_CODE', 'WEATHER1_VIS_DISTANCE_DIMENSION','WEATHER2_TMP_AIR_TEMP','WEATHER2_VIS_VARIABILITY_CODE',
       'WEATHER2_CIG_CEILING_HEIGHT_DIMENSION','WEATHER2_DEW_POINT_TEMP','WEATHER2_CIG_CEILING_DETERMINATION_CODE', 
       'WEATHER2_WND_DIRECTION_ANGLE','WEATHER2_WND_TYPE_CODE','WEATHER2_SLP_SEA_LEVEL_PRES',
       'WEATHER2_WND_SPEED_RATE','WEATHER2_CIG_CAVOK_CODE', 'WEATHER2_VIS_DISTANCE_DIMENSION'] 
    
    df = df.select(final_cols_to_keep)
    # cast fields to correct data type
    # StringType(), DoubleType(), IntegerType()
    df = df.withColumn('YEAR', df['YEAR'].cast(StringType()))
    df = df.withColumn('QUARTER', df['QUARTER'].cast(StringType()))
    df = df.withColumn('MONTH', df['MONTH'].cast(StringType()))
    df = df.withColumn('DAY_OF_WEEK', df['DAY_OF_WEEK'].cast(StringType()))
    df = df.withColumn('DAY_OF_MONTH', df['DAY_OF_MONTH'].cast(StringType()))
    df = df.withColumn('OP_UNIQUE_CARRIER', df['OP_UNIQUE_CARRIER'].cast(StringType()))
    df = df.withColumn('ORIGIN', df['ORIGIN'].cast(StringType()))
    df = df.withColumn('DEST', df['DEST'].cast(StringType()))
    df = df.withColumn('DEP_DELAY', df['DEP_DELAY'].cast(IntegerType()))
    df = df.withColumn('DEP_DELAY_NEW', df['DEP_DELAY_NEW'].cast(IntegerType()))
    df = df.withColumn('DEP_DEL15', df['DEP_DEL15'].cast(IntegerType()))
    df = df.withColumn('ACTUAL_ELAPSED_TIME', df['ACTUAL_ELAPSED_TIME'].cast(IntegerType()))
    df = df.withColumn('AIR_TIME', df['AIR_TIME'].cast(IntegerType()))
    df = df.withColumn('DISTANCE', df['DISTANCE'].cast(IntegerType()))
    df = df.withColumn('CARRIER_DELAY', df['CARRIER_DELAY'].cast(IntegerType()))
    df = df.withColumn('WEATHER_DELAY', df['WEATHER_DELAY'].cast(IntegerType()))
    df = df.withColumn('NAS_DELAY', df['NAS_DELAY'].cast(IntegerType()))
    df = df.withColumn('SECURITY_DELAY', df['SECURITY_DELAY'].cast(IntegerType()))
    df = df.withColumn('LATE_AIRCRAFT_DELAY', df['LATE_AIRCRAFT_DELAY'].cast(IntegerType()))
    df = df.withColumn('IN_FLIGHT_AIR_DELAY', df['IN_FLIGHT_AIR_DELAY'].cast(IntegerType()))
    df = df.withColumn('WEATHER1_TMP_AIR_TEMP', df['WEATHER1_TMP_AIR_TEMP'].cast(IntegerType()))
    df = df.withColumn('WEATHER1_VIS_VARIABILITY_CODE', df['WEATHER1_VIS_VARIABILITY_CODE'].cast(StringType()))
    df = df.withColumn('WEATHER1_CIG_CEILING_HEIGHT_DIMENSION', df['WEATHER1_CIG_CEILING_HEIGHT_DIMENSION'].cast(IntegerType()))
    df = df.withColumn('WEATHER1_DEW_POINT_TEMP', df['WEATHER1_DEW_POINT_TEMP'].cast(IntegerType()))
    df = df.withColumn('WEATHER1_CIG_CEILING_DETERMINATION_CODE', df['WEATHER1_CIG_CEILING_DETERMINATION_CODE'].cast(StringType()))
    df = df.withColumn('WEATHER1_WND_DIRECTION_ANGLE', df['WEATHER1_WND_DIRECTION_ANGLE'].cast(IntegerType()))
    df = df.withColumn('WEATHER1_WND_TYPE_CODE', df['WEATHER1_WND_TYPE_CODE'].cast(StringType()))
    df = df.withColumn('WEATHER1_SLP_SEA_LEVEL_PRES', df['WEATHER1_SLP_SEA_LEVEL_PRES'].cast(IntegerType()))
    df = df.withColumn('WEATHER1_WND_SPEED_RATE', df['WEATHER1_WND_SPEED_RATE'].cast(IntegerType()))
    df = df.withColumn('WEATHER1_CIG_CAVOK_CODE', df['WEATHER1_CIG_CAVOK_CODE'].cast(StringType()))
    df = df.withColumn('WEATHER1_VIS_DISTANCE_DIMENSION', df['WEATHER1_VIS_DISTANCE_DIMENSION'].cast(IntegerType()))
    df = df.withColumn('WEATHER2_TMP_AIR_TEMP', df['WEATHER1_TMP_AIR_TEMP'].cast(IntegerType()))
    df = df.withColumn('WEATHER2_VIS_VARIABILITY_CODE', df['WEATHER1_VIS_VARIABILITY_CODE'].cast(StringType()))
    df = df.withColumn('WEATHER2_CIG_CEILING_HEIGHT_DIMENSION', df['WEATHER1_CIG_CEILING_HEIGHT_DIMENSION'].cast(IntegerType()))
    df = df.withColumn('WEATHER2_DEW_POINT_TEMP', df['WEATHER1_DEW_POINT_TEMP'].cast(IntegerType()))
    df = df.withColumn('WEATHER2_CIG_CEILING_DETERMINATION_CODE', df['WEATHER1_CIG_CEILING_DETERMINATION_CODE'].cast(StringType()))
    df = df.withColumn('WEATHER2_WND_DIRECTION_ANGLE', df['WEATHER1_WND_DIRECTION_ANGLE'].cast(IntegerType()))
    df = df.withColumn('WEATHER2_WND_TYPE_CODE', df['WEATHER1_WND_TYPE_CODE'].cast(StringType()))
    df = df.withColumn('WEATHER2_SLP_SEA_LEVEL_PRES', df['WEATHER1_SLP_SEA_LEVEL_PRES'].cast(IntegerType()))
    df = df.withColumn('WEATHER2_WND_SPEED_RATE', df['WEATHER1_WND_SPEED_RATE'].cast(IntegerType()))
    df = df.withColumn('WEATHER2_CIG_CAVOK_CODE', df['WEATHER1_CIG_CAVOK_CODE'].cast(StringType()))
    df = df.withColumn('WEATHER2_VIS_DISTANCE_DIMENSION', df['WEATHER1_VIS_DISTANCE_DIMENSION'].cast(IntegerType()))
    
    return df
   

    

# COMMAND ----------

# MAGIC %md # Missing Data

# COMMAND ----------

def correct_missing_codes(df):
    missing_value_ref = {'WEATHER1_TMP_AIR_TEMP': 9999,
                       'WEATHER1_CIG_CEILING_HEIGHT_DIMENSION': 99999,
                       'WEATHER1_DEW_POINT_TEMP': 9999,
                       'WEATHER1_WND_DIRECTION_ANGLE': 999,
                       'WEATHER1_SLP_SEA_LEVEL_PRES': 99999,
                       'WEATHER1_WND_SPEED_RATE': 9999,
                       'WEATHER1_VIS_DISTANCE_DIMENSION': 999999,
                       'WEATHER2_TMP_AIR_TEMP': 9999,
                       'WEATHER2_CIG_CEILING_HEIGHT_DIMENSION': 99999,
                       'WEATHER2_DEW_POINT_TEMP': 9999,
                       'WEATHER2_WND_DIRECTION_ANGLE': 999,
                       'WEATHER2_SLP_SEA_LEVEL_PRES': 99999,
                       'WEATHER2_WND_SPEED_RATE': 9999,
                       'WEATHER2_VIS_DISTANCE_DIMENSION': 999999}
    
    for col in missing_value_ref.keys():
        df = df.withColumn(col, f.when(df[col] == missing_value_ref[col], f.lit(None)).otherwise(df[col]))
    
    return df

def impute_missing_values(df):
  missing_count_list = []
  for c in df.columns:
      if df.where(f.col(c).isNull()).count() > 0:
          tup = (c,int(df.where(f.col(c).isNull()).count()))
          missing_count_list.append(tup)

  missing_column_list = [x[0] for x in missing_count_list]
  missing_df = df.select(missing_column_list)

  missing_cat_columns = [item[0] for item in missing_df.dtypes if item[1].startswith('string') and item != "YEAR"]  # string 
  print("\nCategorical Columns with missing data:", missing_cat_columns)

  missing_num_columns = [item[0] for item in missing_df.dtypes if item[1].startswith('int') | item[1].startswith('double')] # will select name of column with integer or double data type
  print("\nNumerical Columns with missing data:", missing_num_columns)
  
  # Fill the missing categorical values with the most frequent category 
  for x in missing_cat_columns:                  
    mode = df.groupBy(x).count().sort(f.col("count").desc()).collect()
    if mode:
      #print(x, mode[0][0]) #print name of columns and it's most categories 
      df = df.withColumn(x, f.when(df[x].isNull(), f.lit(mode[0][0])).otherwise(df[x]))

  # Fill the missing numerical values with the average of each #column
  for i in missing_num_columns:
    mean_value = df.select(f.round(f.mean(i))).collect()
    mean_value = df.select(f.mean(i).cast(IntegerType())).collect()
    
    if mean_value:
        #print(i, mean_value[0][0]) 
        df = df.withColumn(i, f.when(df[i].isNull(), mean_value[0][0]).otherwise(df[i]))
  
  return df

# COMMAND ----------

#cleaned_data = clean_data(flights_and_weather_combined_processed.sample(0.01, False)) # clean and cast our data to appropriate data types
cleaned_data = clean_data(flights_and_weather_combined_processed) # clean and cast our data to appropriate data types
cols_to_drop = [ 'DEP_DELAY', 'DEP_DELAY_NEW', 'ACTUAL_ELAPSED_TIME', 
       'AIR_TIME','CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY',
       'LATE_AIRCRAFT_DELAY','IN_FLIGHT_AIR_DELAY']
cleaned_data = cleaned_data.drop(*cols_to_drop)
cleaned_data = cleaned_data.createOrReplaceTempView('cleaned_data')
cleaned_data = spark.sql('SELECT * FROM cleaned_data')
dbutils.fs.rm('/airline_delays/kevin/DLRS/flights_baseline/processed', recurse=True)
cleaned_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save('/airline_delays/kevin/DLRS/flights_baseline/processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_and_weather_combined_processed_baseline;
# MAGIC 
# MAGIC CREATE TABLE flights_and_weather_combined_processed_baseline
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/kevin/DLRS/flights_baseline/processed"

# COMMAND ----------

cleaned_data_with_missing_values_corrected = correct_missing_codes(spark.sql("SELECT * FROM flights_and_weather_combined_processed_baseline"))
data_ready_for_model = impute_missing_values(cleaned_data_with_missing_values_corrected)
data_ready_for_model.createOrReplaceTempView('data_ready_for_model')

# COMMAND ----------

# MAGIC %md # Pipelining

# COMMAND ----------

# MAGIC %md
# MAGIC First we need to drop columns we can't use at inference time.

# COMMAND ----------

display(data_ready_for_model)

# COMMAND ----------

# MAGIC %md ## Split data into train, validation, test

# COMMAND ----------

train_data = spark.sql("SELECT * FROM data_ready_for_model WHERE YEAR IN (2015, 2016, 2017)").drop('YEAR')
validation_data = spark.sql("SELECT * FROM data_ready_for_model WHERE YEAR = 2018;").drop('YEAR')
test_data = spark.sql("SELECT * FROM data_ready_for_model WHERE YEAR = 2019;").drop('YEAR')

train_data.createOrReplaceTempView('train_data')
validation_data.createOrReplaceTempView('validation_data')
test_data.createOrReplaceTempView('test_data')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Routines to encode and scale our features
# MAGIC 
# MAGIC We encode and scale only on train data and set up the transformation pipeline with those stages so it may be applied on validate and test data later

# COMMAND ----------

# MAGIC %md ### Encoding Function

# COMMAND ----------

# Use the OneHotEncoderEstimator to convert categorical features into one-hot vectors
# Use VectorAssembler to combine vector of one-hots and the numerical features
# Append the process into the stages array to reproduce

def create_encoding_stages(data, label_name):
  
  cat_cols = [item[0] for item in data.dtypes if item[1].startswith('string')]
  numeric_cols = [item[0] for item in data.dtypes if item[1].startswith('int') | item[1].startswith('double')]
  numeric_cols.remove(label_name)
  
  # One-Hot Encode Categorical Columns in the vector
  string_indexer = StringIndexer(inputCols=cat_cols, outputCols=[c + '_idx' for c in cat_cols] , handleInvalid = 'keep') 
  encoder = OneHotEncoder(inputCols= [c + '_idx' for c in cat_cols], outputCols = [c + '_enc' for c in cat_cols])

  # Deal with Numeric Features
  vector_num_assembler = VectorAssembler(inputCols = numeric_cols, outputCol = 'unscaled_features', handleInvalid = 'keep')
  scaler = StandardScaler(inputCol = 'unscaled_features', outputCol = 'scaled_features', withStd = True, withMean = True)
  
  final_cols = [c + '_enc' for c in cat_cols]
  final_cols.append('scaled_features')
  final_assembler = VectorAssembler(inputCols = final_cols, outputCol = 'features', handleInvalid = 'keep')

  label_string_indexer = StringIndexer(inputCol = label_name, outputCol = 'label')

  stages = [string_indexer, encoder, vector_num_assembler, scaler, final_assembler, label_string_indexer]
  
  return stages

# COMMAND ----------

# MAGIC %md ### Routines to evaluate our model performance
# MAGIC 
# MAGIC We train our models and evaluate performance throughout the pipeline

# COMMAND ----------

## This function will calculate evaluation metrics based on predictions
def evaluation_metrics(predictions, model_name):
  predictions = predictions.createOrReplaceTempView('predictions')
  display(predictions)
  true_positives = spark.sql("SELECT COUNT(*) FROM predictions WHERE predictions.label = 1 AND predictions.prediction = 1").collect()[0][0]
  true_negatives = spark.sql("SELECT COUNT(*) FROM predictions WHERE predictions.label = 0 AND predictions.prediction = 0").collect()[0][0]
  false_positives = spark.sql("SELECT COUNT(*) FROM predictions WHERE predictions.label = 0 AND predictions.prediction = 1").collect()[0][0]
  false_negatives = spark.sql("SELECT COUNT(*) FROM predictions WHERE predictions.label = 1 AND predictions.prediction = 0").collect()[0][0]
  
  # Now we should compute our main statistics
  accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
  recall = (true_positives) / (true_positives + false_negatives)
  precision = (true_positives) / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
  f1_score = (2 * recall * precision) / (recall + precision) if (recall + precision) != 0 else 0

  print("The accuracy is: %s" % np.round(accuracy, 4))
  print("The recall is: %s" % np.round(recall, 4))
  print("The precision is: %s" % np.round(precision, 4))
  print("The f1_score is: %s" % np.round(f1_score, 4))
  df_cols = ['Model','accuracy', 'precision', 'recall', 'f1_score', 'true_positives', 'true_negatives', 'false_positives', 'false_negatives']
  metrics_dataframe = pd.DataFrame(columns = df_cols)
  result_dict = {'Model': model_name,
                 'accuracy': accuracy,
                 'precision': precision, 
                 'recall': recall, 
                 'f1_score': f1_score, 
                 'true_positives': true_positives, 
                 'true_negatives': true_negatives, 
                 'false_positives': false_positives, 
                 'false_negatives': false_negatives}
  metrics_dataframe = metrics_dataframe.append(result_dict, ignore_index=True)
  
  return metrics_dataframe

def confusion_matrix(model_results,index):
  confusion_matrix = pd.DataFrame(columns = ['_','Predicted Delay', 'Predicted No Delay'])
  conf_matrix_delay = {'_': 'Actual Delay', 'Predicted Delay': model_results['true_positives'][index], 'Predicted No Delay': model_results['false_negatives'][index] }
  conf_matrix_no_delay = {'_': 'Actual  No Delay', 'Predicted Delay': model_results['false_positives'][index], 'Predicted No Delay': model_results['true_negatives'][index] }
  confusion_matrix = confusion_matrix.append(conf_matrix_delay,ignore_index=True)
  confusion_matrix = confusion_matrix.append(conf_matrix_no_delay,ignore_index=True)
  return confusion_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 1: Baseline Evaluation
# MAGIC 
# MAGIC Set up and run the pipeline for the baseline model

# COMMAND ----------

# create an encoding pipeline based on information from our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data,'DEP_DEL15')).fit(train_data)

# apply the transformations to our train data
transformed_train_data = encoding_pipeline.transform(train_data)['features', 'label']


# train a model on our transformed train data
startTime = time.time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 30, regParam = 0.001)
model = lr.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes")
                             

# COMMAND ----------

train_metrics = evaluation_metrics(train_preds, "Logistic Regression on training data")
display(train_metrics)

# COMMAND ----------

# MAGIC %md #### Evaluate against validation

# COMMAND ----------

# apply the encoding transformations from our pipeline to the validation data
transformed_validation_data = encoding_pipeline.transform(validation_data)['features', 'label']

# run the fitted model on the transformed validation data
validation_preds = model.transform(transformed_validation_data)

# COMMAND ----------

# display our evaluation metrics
validation_metrics = evaluation_metrics(validation_preds, "Logistic Regression on validation data")
display(validation_metrics)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Feature Selection and Engineering for next model

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

# MAGIC %md ## PageRank Features

# COMMAND ----------

joinedDf = spark.sql("SELECT * from joinedDf")
display(joinedDf) 

# COMMAND ----------

airlineGraph = {'nodes': joinedDf.select('ORIGIN', 'DEST').rdd.flatMap(list).distinct().collect(), 
                'edges': joinedDf.select('ORIGIN', 'DEST').rdd.map(tuple).collect()}

directedGraph = nx.DiGraph()
directedGraph.add_nodes_from(airlineGraph['nodes'])
directedGraph.add_edges_from(airlineGraph['edges'])

pageRank = nx.pagerank(directedGraph, alpha = 0.85)
pandasPageRank = pd.DataFrame(pageRank.items(), columns = ['Station', 'PageRank'])
pandasPageRank = spark.createDataFrame(pandasPageRank)
pandasPageRank.createOrReplaceTempView("pandasPageRank")
joinedDf.createOrReplaceTempView("joinedDf")
# Now we want to separate the pagerank for the stations based on destination and origin
joinedDf = spark.sql("SELECT * from joinedDf LEFT JOIN pandasPageRank ON joinedDf.ORIGIN == pandasPageRank.Station").drop('Station')
joinedDf = joinedDf.withColumnRenamed('PageRank', 'PAGERANK_ORIGIN')
joinedDf.createOrReplaceTempView("joinedDf")
# Repeat for Dest
joinedDf = spark.sql("SELECT * from joinedDf LEFT JOIN pandasPageRank ON joinedDf.DEST == pandasPageRank.Station").drop('Station')
joinedDf = joinedDf.withColumnRenamed('PageRank', 'PAGERANK_DEST')
joinedDf.createOrReplaceTempView('joinedDf')
display(joinedDf)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Possible Weighting NEXT TIME SAYAN TODO

# COMMAND ----------

