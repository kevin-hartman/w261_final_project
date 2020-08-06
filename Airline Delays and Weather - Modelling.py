# Databricks notebook source
# MAGIC %md # W261 Final Project - Airline Delays and Weather Modeling Notebook
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Summer 2020`__
# MAGIC ### Team 1:
# MAGIC * Sayan Das
# MAGIC * Kevin Hartman
# MAGIC * Hersh Solanki
# MAGIC * Nick Sylva

# COMMAND ----------

# MAGIC %md # Table of Contents
# MAGIC ### 6. Pre-Model Data Prep
# MAGIC #### a. Configure access to post-processed data from the silver data pipeline
# MAGIC #### b. Split into Train/Validation/Test
# MAGIC #### c. Apply specific cleanup required before modelling (e.g. imputation)
# MAGIC #### d. Save the model-ready datasets in silver
# MAGIC ### 7. Model Exploration (Pipeline)
# MAGIC #### a. Feature Selection
# MAGIC #### b. Feature Engineering
# MAGIC #### c. Transformations (Encoding & Scaling)
# MAGIC #### d. Evaluation
# MAGIC ### 8. Model Selection and Tuning

# COMMAND ----------

# MAGIC %md **Note that this notebook consists of our work modeling three main algorithms: Logistic Regression, Random Forest, and Gradient-Boosted Trees.**

# COMMAND ----------

# MAGIC %md ## Notebook Imports

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
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder, PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

%matplotlib inline
sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md ### 6a. Configure access to data pipeline staging areas 

# COMMAND ----------

username = "kevin"
dbutils.widgets.text("username", username)
spark.sql(f"CREATE DATABASE IF NOT EXISTS airline_delays_{username}")
spark.sql(f"USE airline_delays_{username}")

flights_and_weather_pipeline_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_pipeline/"

spark.conf.set("spark.sql.shuffle.partitions", 8)

# COMMAND ----------

# MAGIC %md
# MAGIC #### If running this notebook for the first time, continue with section 6. Otherwise, jump to section 7.

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Pre-Model Data Prep

# COMMAND ----------

# MAGIC %md ## 6b. Split data into train, validation, test

# COMMAND ----------

train_data = spark.sql("SELECT * FROM flights_and_weather_pipeline_processed WHERE YEAR IN (2015, 2016, 2017)").drop('YEAR')
validation_data = spark.sql("SELECT * FROM flights_and_weather_pipeline_processed WHERE YEAR = 2018").drop('YEAR')
test_data = spark.sql("SELECT * FROM flights_and_weather_pipeline_processed WHERE YEAR = 2019").drop('YEAR')

# COMMAND ----------

# MAGIC %md ## 6c. Apply specific cleanup required before modeling (e.g. imputation)

# COMMAND ----------

# MAGIC %md ### Prepare an imputation dictionary to use to replace missing values

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

# MAGIC %md Dictionary should be populated from ONLY the train set to prevent leakage into validation and test sets.

# COMMAND ----------

impute_dict = make_imputation_dict(train_data)

# COMMAND ----------

# MAGIC %md Impute our missing values using the mean and modes found in our impute dictionary

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

# MAGIC %md Impute our training data

# COMMAND ----------

train_data = impute_missing_values(train_data, impute_dict)

# COMMAND ----------

# MAGIC %md Let's save our train data that has been imputed to the Delta Lake.

# COMMAND ----------

flights_and_weather_train_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_train/"
dbutils.fs.rm(flights_and_weather_train_loc + 'processed', recurse=True)
train_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_train_loc + 'processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_and_weather_train_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_and_weather_train_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_and_weather_train/processed"

# COMMAND ----------

# MAGIC %md Next let's impute our validation data based on values stored in our dictionary (this is unseen data)

# COMMAND ----------

validation_data = impute_missing_values(validation_data, impute_dict)

# COMMAND ----------

# MAGIC %md Let's save our validation data that has been imputed to the Delta Lake.

# COMMAND ----------

flights_and_weather_validation_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_validation/"
dbutils.fs.rm(flights_and_weather_validation_loc + 'processed', recurse=True)
validation_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_validation_loc + 'processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_and_weather_validation_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_and_weather_validation_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_and_weather_validation/processed"

# COMMAND ----------

# MAGIC %md Next let's impute our test data based on values stored in our dictionary (this is unseen data)

# COMMAND ----------

test_data = impute_missing_values(test_data, impute_dict)

# COMMAND ----------

# MAGIC %md Let's save our test data that has been imputed to the Delta Lake.

# COMMAND ----------

flights_and_weather_test_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_test/"
dbutils.fs.rm(flights_and_weather_test_loc + 'processed', recurse=True)
test_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_test_loc + 'processed')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS flights_and_weather_test_processed;
# MAGIC 
# MAGIC CREATE TABLE flights_and_weather_test_processed
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/flights_and_weather_test/processed"

# COMMAND ----------

# MAGIC %md ### Replace all empty strings

# COMMAND ----------

def replace_empty_strings(df):
  cat_columns = [item[0] for item in df.dtypes if item[1].startswith('string')]  # string
    
  # Fill the empty string values with 'Empty'
  for x in cat_columns:                  
     df = df.withColumn(x, f.when(df[x] == '', f.lit("Empty")).otherwise(df[x]))
    
  return df

# COMMAND ----------

train_data = replace_empty_strings(train_data)
validation_data = replace_empty_strings(validation_data)
test_data = replace_empty_strings(test_data)

# COMMAND ----------

train_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_train_loc + 'processed')
validation_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_validation_loc + 'processed')
test_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_test_loc + 'processed')

# COMMAND ----------

# MAGIC %md ### Drop Numeric Columns that are all Null

# COMMAND ----------

def get_cols_to_drop(impute_dict, numeric_cols):
  colsToDrop = []
  numeric_cols = set(numeric_cols)
  for key, value in impute_dict.items():
    if value == None and key in numeric_cols:
      colsToDrop.append(key)
  return colsToDrop

numeric_cols = [item[0] for item in train_data.dtypes if item[1] in ['int', 'double']]
cols_to_drop = get_cols_to_drop(impute_dict, numeric_cols)

#cols_to_drop = ['ORIGIN_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'ORIGIN_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'ORIGIN_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION', 'ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION', 'ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE', 'ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE', 'ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'DEST_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'DEST_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION', 'DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION', 'DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE', 'DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE', 'DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION']

# COMMAND ----------

train_data = train_data.drop(*cols_to_drop)
validation_data = validation_data.drop(*cols_to_drop)
test_data = test_data.drop(*cols_to_drop)

# COMMAND ----------

# MAGIC %md ## 6d. Save datasets into silver

# COMMAND ----------

train_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_train_loc + 'processed')
validation_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_validation_loc + 'processed')
test_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_test_loc + 'processed')

# COMMAND ----------

# MAGIC %md ### Drop Delay Columns (revisit if we want to use a linear regression model to predict the actual time delayed)

# COMMAND ----------

cols_to_drop = ['DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP', 'TAIL_NUM', 'ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'FL_DATE', 'ORIGIN_WEATHER_KEY', 'DEST_WEATHER_KEY']

# COMMAND ----------

train_data = train_data.drop(*cols_to_drop)
validation_data = validation_data.drop(*cols_to_drop)
test_data = test_data.drop(*cols_to_drop)

# COMMAND ----------

train_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_train_loc + 'processed')
validation_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_validation_loc + 'processed')
test_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_test_loc + 'processed')

# COMMAND ----------

print(flights_and_weather_train_loc + 'processed')

# COMMAND ----------

# MAGIC %md # 7. Model Pipeline for Exploration

# COMMAND ----------

# MAGIC %md Let's read in our data from silver for modeling.

# COMMAND ----------

train_data = spark.sql("select * from flights_and_weather_train_processed")
validation_data = spark.sql("select * from flights_and_weather_validation_processed")
test_data = spark.sql("select * from flights_and_weather_test_processed")

cols_to_drop = ['DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP', 'TAIL_NUM', 'ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'FL_DATE', 'ORIGIN_WEATHER_KEY', 'DEST_WEATHER_KEY', 'ORIGIN_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'ORIGIN_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'ORIGIN_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION', 'ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION', 'ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE', 'ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE', 'ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'DEST_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'DEST_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION', 'DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION', 'DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE', 'DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE', 'DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION']

train_data = train_data.drop(*cols_to_drop)
validation_data = validation_data.drop(*cols_to_drop)
test_data = test_data.drop(*cols_to_drop)

# COMMAND ----------

# MAGIC %md ### Routines to encode and scale our features
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
  cat_cols.remove(label_name)
  
  # One-Hot Encode Categorical Columns in the vector
  cat_pipeline = Pipeline(stages=[
    StringIndexer(inputCols=cat_cols, outputCols=[c + '_idx' for c in cat_cols] , handleInvalid = 'keep') ,
    OneHotEncoder(inputCols= [c + '_idx' for c in cat_cols], outputCols = [c + '_enc' for c in cat_cols]),
    VectorAssembler(inputCols=[c + '_enc' for c in cat_cols], outputCol="cat_features")
   ])  
  
  # Deal with Numeric Features
  num_pipeline = Pipeline(stages=[
    VectorAssembler(inputCols = numeric_cols, outputCol = 'unscaled_features', handleInvalid = 'keep'),
    StandardScaler(inputCol = 'unscaled_features', outputCol = 'scaled_features', withStd = True, withMean = True)
  ])
   
  feature_assembler = Pipeline(stages=[
    cat_pipeline,
    num_pipeline,
    VectorAssembler(inputCols = ["cat_features", "scaled_features"], outputCol = 'features', handleInvalid = 'keep')
  ])

  label_string_indexer = StringIndexer(inputCol = label_name, outputCol = 'label')

  stages = [feature_assembler] + [label_string_indexer]
  
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

def AUC(model, predictions):
  from pyspark.ml.evaluation import BinaryClassificationEvaluator
  import matplotlib.pyplot as plt
  evaluator = BinaryClassificationEvaluator()
  evaluation = evaluator.evaluate(predictions)
  print("evaluation (area under ROC): %f" % evaluation)

def ROC(model, predictions):
  plt.figure(figsize=(10,10))
  plt.plot([0, 1], [0, 1], 'r--')
  plt.plot(model.summary.roc.select('FPR').collect(),
         model.summary.roc.select('TPR').collect())
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.show()

def showPR(model_results):
  pr = model_results.toPandas()
  ax.plot(pr['recall'], pr['precision'])
  ax.set_xlabel('Precision')
  ax.set_ylabel('Recall')
  ax.set_title('Precision Recall')

# COMMAND ----------

# MAGIC %md # Model 1: Baseline Evaluation (Logistic Regression)
# MAGIC 
# MAGIC Set up and run the pipeline for the baseline model (using select highest correlated features)

# COMMAND ----------

correlated_cols = ['ORIGIN_WND_SPEED_RATE', 'ORIGIN_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY', 'ORIGIN_SNOW_DEPTH_DIMENSION',
 'ORIGIN_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_DIRECTION_ANGLE',
 'ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION', 'ORIGIN_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION',
 'ORIGIN_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION',
 'DEST_WND_SPEED_RATE', 'DEST_AA2_LIQUID_PRECIPITATION_PERIOD_QUANTITY', 'DEST_AA3_LIQUID_PRECIPITATION_DEPTH_DIMENSION',
 'DEST_SNOW_DEPTH_DIMENSION', 'DEST_AL1_SNOW_ACCUMULATION_DEPTH_DIMENSION',
 'DEST_GA2_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION',  'DEST_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION',
 'DEST_SKY_CONDITION_OBSERVATION_LOWEST_CLOUD_BASE_HEIGHT_DIMENSION', 'DEP_DEL15']

# COMMAND ----------

# create an encoding pipeline based on information from our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data[correlated_cols],'DEP_DEL15'))
encoding_pipeline = encoding_pipeline.fit(train_data[correlated_cols])
# apply the transformations to our train data
transformed_train_data = encoding_pipeline.transform(train_data[correlated_cols])['features', 'label']

# COMMAND ----------

# train a model on our transformed train data
startTime = time.time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 30, regParam = 0.001, elasticNetParam = 0.25, standardization = False)
model = lr.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes")                    

# COMMAND ----------

train_metrics = evaluation_metrics(train_preds, "Logistic Regression on training data")
display(train_metrics)

# COMMAND ----------

AUC(model, train_preds)

# COMMAND ----------

ROC(model, train_preds)

# COMMAND ----------

# MAGIC %md #### Evaluate against validation

# COMMAND ----------

# apply the encoding transformations from our pipeline to the validation data
transformed_validation_data = encoding_pipeline.transform(validation_data[correlated_cols])['features', 'label']

# run the fitted model on the transformed validation data
validation_preds = model.transform(transformed_validation_data)

# display our evaluation metrics
validation_metrics = evaluation_metrics(validation_preds, "Logistic Regression on validation data")
display(validation_metrics)

# COMMAND ----------

# MAGIC %md # Parameter Search

# COMMAND ----------

# transformed_train_data_sample = transformed_train_data.sample(False, 0.0001)
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def parameter_search(model, featuresCol, labelCol, dictionary, data):
  mod = model(featuresCol = featuresCol, labelCol = labelCol)
  pipeline = Pipeline(stages=[mod])
  
  paramGrid = ParamGridBuilder()
  for i in range(len(dictionary)):
    key = list(dictionary.keys())[i]
    value = list(dictionary.values())[i]        
    paramGrid = paramGrid.addGrid(getattr(mod,key) , value)
  
  paramGrid = paramGrid.build()

  crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=2) 

  cvModel = crossval.fit(data)
  return(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])

# COMMAND ----------

parameters = {'maxIter': [1, 30], 'regParam' : [0.01, 0.001], 'elasticNetParam' : [0.25, 0.5, 0.1]}
parameter_search(LogisticRegression, 'features', 'label', parameters, transformed_train_data.sample(False, 0.005))

# COMMAND ----------

# MAGIC %md
# MAGIC     # Results
# MAGIC     {Param(parent='LogisticRegression_6a0dd178a09d', name='maxIter', doc='max number of iterations (>= 0).'): 30,
# MAGIC      Param(parent='LogisticRegression_6a0dd178a09d', name='regParam', doc='regularization parameter (>= 0).'): 0.01,
# MAGIC      Param(parent='LogisticRegression_6a0dd178a09d', name='elasticNetParam',
# MAGIC      doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.5}

# COMMAND ----------

# MAGIC %md
# MAGIC ### LR Model using results from parameter search

# COMMAND ----------

# Using results from our parameter search
startTime = time.time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 30, regParam = 0.01, elasticNetParam = 0.5, standardization = False)
model = lr.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes")
train_metrics = evaluation_metrics(train_preds, "Logistic Regression on training data")
display(train_metrics)

# COMMAND ----------

AUC(model, train_preds)
ROC(model, train_preds)

# COMMAND ----------

# run the fitted model on the transformed validation data
validation_preds = model.transform(transformed_validation_data)
# display our evaluation metrics
validation_metrics = evaluation_metrics(validation_preds, "Logistic Regression on validation data")
display(validation_metrics)

# COMMAND ----------

# MAGIC %md Our precision is much worse with these parameters. The 5% sample is not enough to conduct a proper CV. We will review a full CV when our final model is selected later.

# COMMAND ----------

# MAGIC %md # Feature Selection and Engineering for next models

# COMMAND ----------

# MAGIC %md ## PageRank Features (PR is based on train data only)

# COMMAND ----------

#PageRank
flights_and_weather_train_processed = spark.sql("SELECT * from flights_and_weather_train_processed")
airlineGraph = {'nodes': flights_and_weather_train_processed.select('ORIGIN', 'DEST').rdd.flatMap(list).distinct().collect(), 
                'edges': flights_and_weather_train_processed.select('ORIGIN', 'DEST').rdd.map(tuple).collect()}

directedGraph = nx.DiGraph()
directedGraph.add_nodes_from(airlineGraph['nodes'])
directedGraph.add_edges_from(airlineGraph['edges'])

pageRank = nx.pagerank(directedGraph, alpha = 0.85)
pandasPageRank = pd.DataFrame(pageRank.items(), columns = ['Station', 'PageRank'])
pandasPageRank = spark.createDataFrame(pandasPageRank)

# COMMAND ----------

pagerank_origin = pandasPageRank.withColumnRenamed('Station','ORIGIN_IATA').withColumnRenamed('PageRank','ORIGIN_PAGERANK')
pagerank_dest= pandasPageRank.withColumnRenamed('Station','DEST_IATA').withColumnRenamed('PageRank','DEST_PAGERANK')

# COMMAND ----------

pagerank_loc = f"/airline_delays/{username}/DLRS/pagerank/"

dbutils.fs.rm(pagerank_loc + 'origin', recurse=True)
dbutils.fs.rm(pagerank_loc + 'dest', recurse=True)

# COMMAND ----------

pagerank_origin.write.option('mergeSchema', True).mode('overwrite').format('delta').save(pagerank_loc + 'origin')
pagerank_dest.write.option('mergeSchema', True).mode('overwrite').format('delta').save(pagerank_loc + 'dest')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS pagerank_origin;
# MAGIC 
# MAGIC CREATE TABLE pagerank_origin
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/pagerank/origin"

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS pagerank_dest;
# MAGIC 
# MAGIC CREATE TABLE pagerank_dest
# MAGIC USING DELTA
# MAGIC LOCATION "/airline_delays/$username/DLRS/pagerank/dest"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rolling Average Features

# COMMAND ----------

train_data = spark.sql('SELECT * FROM flights_and_weather_train_ra_processed')
validation_data = spark.sql('SELECT * FROM flights_and_weather_validation_ra_processed')
test_data = spark.sql('SELECT * FROM flights_and_weather_test_ra_processed')

# COMMAND ----------

train_data = replace_empty_strings(train_data)
validation_data = replace_empty_strings(validation_data)
test_data = replace_empty_strings(test_data)

# COMMAND ----------

flights_and_weather_train_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_ra_train/"
flights_and_weather_validation_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_ra_validation/"
flights_and_weather_test_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_ra_test/"

# COMMAND ----------

train_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_train_loc + 'processed')
validation_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_validation_loc + 'processed')
test_data.write.option('mergeSchema', True).mode('overwrite').format('delta').save(flights_and_weather_test_loc + 'processed')

# COMMAND ----------

cols_to_drop = ['DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP', 'TAIL_NUM', 'ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'FL_DATE', 'ORIGIN_WEATHER_KEY', 'DEST_WEATHER_KEY', 'ORIGIN_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'ORIGIN_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'ORIGIN_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION', 'ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION', 'ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE', 'ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE', 'ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'DEST_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'DEST_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION', 'DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION', 'DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE', 'DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE', 'DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION']

train_data = train_data.drop(*cols_to_drop)
validation_data = validation_data.drop(*cols_to_drop)
test_data = test_data.drop(*cols_to_drop)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Model Attempts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient Boosted Trees

# COMMAND ----------

# MAGIC %md #### GBT on limited correlated columns

# COMMAND ----------

startTime = time.time()
gbt = GBTClassifier(labelCol = "label", featuresCol = "features", maxIter = 10)
gbt_model = gbt.fit(transformed_train_data)
gbt_train_preds = gbt_model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Gradient-Boosted Tree model is: {(endTime - startTime) / (60)} minutes")
gbt_train_metrics = evaluation_metrics(gbt_train_preds, "Gradient-Boosted Trees on training data")
display(gbt_train_metrics)

# COMMAND ----------

AUC(model, train_preds)
ROC(model, train_preds)

# COMMAND ----------

gbt_validation_preds = gbt_model.transform(transformed_validation_data)
gbt_validation_metrics = evaluation_metrics(gbt_validation_preds, "Gradient-Boosted Trees on validation data")
display(gbt_validation_metrics)

# COMMAND ----------

AUC(model, train_preds)
ROC(model, train_preds)

# COMMAND ----------

# MAGIC %md #### GBT on full set of columns

# COMMAND ----------

# create an encoding pipeline based on information from our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data,'DEP_DEL15'))
encoding_pipeline = encoding_pipeline.fit(train_data)
# apply the transformations to our train data
transformed_train_data = encoding_pipeline.transform(train_data)['features', 'label']


# COMMAND ----------

startTime = time.time()
gbt = GBTClassifier(labelCol = "label", featuresCol = "features", maxIter = 10)
gbt_model = gbt.fit(transformed_train_data)
gbt_train_preds = gbt_model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Gradient-Boosted Tree model is: {(endTime - startTime) / (60)} minutes")
gbt_train_metrics = evaluation_metrics(gbt_train_preds, "Gradient-Boosted Trees on training data")
display(gbt_train_metrics)

# COMMAND ----------

# apply the encoding transformations from our pipeline to the validation data
transformed_validation_data = encoding_pipeline.transform(validation_data)['features', 'label']

gbt_validation_preds = gbt_model.transform(transformed_validation_data)
gbt_validation_metrics = evaluation_metrics(gbt_validation_preds, "Gradient-Boosted Trees on validation data")
display(gbt_validation_metrics)

# COMMAND ----------

# MAGIC %md #### Gradient-Boosted Trees with PR

# COMMAND ----------

# MAGIC %md Running this example on the limited set of highly correlated columns

# COMMAND ----------

join_statement = " JOIN pagerank_origin op ON op.ORIGIN_IATA = f.ORIGIN JOIN pagerank_dest dp ON dp.DEST_IATA = f.DEST"

train_data_pr = spark.sql("SELECT * FROM flights_and_weather_train_processed f" + join_statement).drop(*cols_to_drop)
validation_data_pr = spark.sql("SELECT * FROM flights_and_weather_validation_processed f" + join_statement).drop(*cols_to_drop)
test_data_pr = spark.sql("SELECT * FROM flights_and_weather_test_processed f" + join_statement).drop(*cols_to_drop)

# COMMAND ----------

# create an encoding pipeline based on information from our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data_pr[correlated_cols],'DEP_DEL15')).fit(train_data_pr[correlated_cols])

# apply the transformations to our train data
transformed_train_data = encoding_pipeline.transform(train_data_pr[correlated_cols])['features', 'label']



# COMMAND ----------

startTime = time.time()
gbt = GBTClassifier(labelCol = "label", featuresCol = "features", maxIter = 10)
gbt_model = gbt.fit(transformed_train_data)
train_preds = gbt_model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Gradient-Boosted Tree model is: {(endTime - startTime) / (60)} minutes")
train_metrics = evaluation_metrics(train_preds, "Gradient-Boosted Trees on training data")
display(train_metrics)

# COMMAND ----------

# apply the encoding transformations from our pipeline to the validation data
transformed_validation_data = encoding_pipeline.transform(validation_data_pr[correlated_cols])['features', 'label']

#transformed_validation_data = encoding_pipeline.transform(validation_data)['features', 'label']
validation_preds = gbt_model.transform(transformed_validation_data)
validation_metrics = evaluation_metrics(validation_preds, "Gradient-Boosted Trees on validation data")
display(validation_metrics)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Random Forest 

# COMMAND ----------

# MAGIC %md Note that we ran this model in another notebook and the output is not available here. Our results are summarized in the overview notebook.

# COMMAND ----------

# create an encoding pipeline based on information from our training data
# encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data,'DEP_DEL15')).fit(train_data)

# apply the transformations to our train data
# transformed_train_data = encoding_pipeline.transform(train_data)['features', 'label']


# train a model on our transformed train data
startTime = time.time()
rf = RandomForestClassifier(labelCol = "label", featuresCol = "features", numTrees = 45, 
                           featureSubsetStrategy = "auto", impurity = "gini", subsamplingRate = 0.6, maxDepth = 25, maxBins = 16)
model = rf.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Random Forest model is: {(endTime - startTime) / (60)} minutes")
                             

# COMMAND ----------

train_metrics = evaluation_metrics(train_preds, "Random Forest on training data")
display(train_metrics)

# COMMAND ----------

#transformed_validation_data = encoding_pipeline.transform(validation_data)['features', 'label']
validation_preds = model.transform(transformed_validation_data)
validation_metrics = evaluation_metrics(validation_preds, "Random Forest on validation data")
display(validation_metrics)

# COMMAND ----------

# MAGIC %md ## PageRank Random Forest

# COMMAND ----------

#encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data,'DEP_DEL15')).fit(train_data)

# apply the transformations to our train data
#transformed_train_data = encoding_pipeline.transform(train_data)['features', 'label']

# COMMAND ----------

# train a model on our transformed train data
startTime = time.time()
rf = RandomForestClassifier(labelCol = "label", featuresCol = "features", numTrees = 45, 
                           featureSubsetStrategy = "auto", impurity = "gini", subsamplingRate = 0.6, maxDepth = 25, maxBins = 16)
model = rf.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Random Forest model is: {(endTime - startTime) / (60)} minutes")
                             

# COMMAND ----------

train_metrics = evaluation_metrics(train_preds, "Random Forest on training data")
display(train_metrics)

# COMMAND ----------

# MAGIC %md ### Validation Evaluation

# COMMAND ----------

#transformed_validation_data = encoding_pipeline.transform(validation_data)['features', 'label']
validation_preds = model.transform(transformed_validation_data)
validation_metrics = evaluation_metrics(validation_preds, "Random Forest on validation data")
display(validation_metrics)

# COMMAND ----------

# MAGIC %md # 8. Model Selection and Tuning

# COMMAND ----------

# MAGIC %md **We selected the Logistic Regression model for further fine-tuning for reasons presented in the summary notebook.**

# COMMAND ----------

# MAGIC %md ## PageRank Logistic Regression Model

# COMMAND ----------

join_statement = " JOIN pagerank_origin op ON op.ORIGIN_IATA = f.ORIGIN JOIN pagerank_dest dp ON dp.DEST_IATA = f.DEST"

train_data_pr = spark.sql("SELECT * FROM flights_and_weather_train_processed f" + join_statement).drop(*cols_to_drop)
validation_data_pr = spark.sql("SELECT * FROM flights_and_weather_validation_processed f" + join_statement).drop(*cols_to_drop)
test_data_pr = spark.sql("SELECT * FROM flights_and_weather_test_processed f" + join_statement).drop(*cols_to_drop)

# COMMAND ----------

# create an encoding pipeline based on information from our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data_pr,'DEP_DEL15')).fit(train_data_pr)

# apply the transformations to our train data
transformed_train_data = encoding_pipeline.transform(train_data_pr)['features', 'label']


# train a model on our transformed train data
startTime = time.time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 30, regParam = 0.001)
model = lr.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes")
                        

# COMMAND ----------

# MAGIC %md ### Evaluate our training results

# COMMAND ----------

train_metrics = evaluation_metrics(train_preds, "Logistic Regression for PageRank Train")
display(train_metrics)

# COMMAND ----------

# MAGIC %md ### Evaluate our valuation results

# COMMAND ----------

# apply the encoding transformations from our pipeline to the validation data
transformed_validation_data = encoding_pipeline.transform(validation_data_pr)['features', 'label']

# run the fitted model on the transformed validation data
validation_preds = model.transform(transformed_validation_data)

# display our evaluation metrics
validation_metrics = evaluation_metrics(validation_preds, "Logistic Regression for PageRank Validation")
display(validation_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression RA's Full 586 Columns

# COMMAND ----------

# create an encoding pipeline based on information from our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data,'DEP_DEL15'))
encoding_pipeline = encoding_pipeline.fit(train_data)
# apply the transformations to our train data
transformed_train_data = encoding_pipeline.transform(train_data)['features', 'label']

# COMMAND ----------

# train a model on our transformed train data
startTime = time.time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 30, regParam = 0.001, elasticNetParam = 0.25, standardization = False)
model = lr.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes")       

# COMMAND ----------

# MAGIC %md ### Let's evaluate our training results 

# COMMAND ----------

train_metrics = evaluation_metrics(train_preds, "Logistic Regression on training data")
display(train_metrics)

# COMMAND ----------

AUC(model, train_preds)

# COMMAND ----------

ROC(model, train_preds)

# COMMAND ----------

# MAGIC %md ### Train and Evaluate on Validation

# COMMAND ----------

# apply the encoding transformations from our pipeline to the validation data
transformed_validation_data = encoding_pipeline.transform(validation_data)['features', 'label']

# run the fitted model on the transformed validation data
validation_preds = model.transform(transformed_validation_data)

# display our evaluation metrics
validation_metrics = evaluation_metrics(validation_preds, "Logistic Regression on validation data")
display(validation_metrics)

# COMMAND ----------

AUC(model, validation_preds)

# COMMAND ----------

ROC(model, validation_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression on Rolling Average Engineered features, PCA

# COMMAND ----------

pca = PCA(k=100, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(transformed_train_data)
pca_transformed_train_data = pca_model.transform(transformed_train_data)

# COMMAND ----------

# train a model on our transformed train data
startTime = time.time()
lr_pca = LogisticRegression(featuresCol = 'pca_features', labelCol = 'label', maxIter = 30, regParam = 0.001, elasticNetParam = 0.25, standardization = False)
lr_pca_model = lr_pca.fit(pca_transformed_train_data)
lr_pca_train_preds = model.transform(pca_transformed_train_data)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes")   

# COMMAND ----------

# MAGIC %md ### Let's evaluate our training data

# COMMAND ----------

lr_pca_train_metrics = evaluation_metrics(lr_pca_train_preds, "Logistic Regression on training data")
display(lr_pca_train_metrics)

# COMMAND ----------

#transform validation data
pca_transformed_validation_data = pca_model.transform(transformed_validation_data)

# COMMAND ----------

# run the fitted model on the transformed validation data
lr_pca_validation_preds = lr_pca_model.transform(pca_transformed_validation_data)

# COMMAND ----------

# MAGIC %md ### Let's evaluate our validation data

# COMMAND ----------

# display our evaluation metrics
lr_pca_validation_metrics = evaluation_metrics(lr_pca_validation_preds, "Logistic Regression on validation data")
display(lr_pca_validation_metrics)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression on Rolling Average Engineered features, with highly correlated columns

# COMMAND ----------



# COMMAND ----------

cor_picked_cols = ['ORIGIN_WND_SPEED_RATE', 'ORIGIN_WIND_GUST_OBSERVATION_SPEED_RATE', 'DEST_WND_SPEED_RATE', 'ORIGIN_WND_DIRECTION_ANGLE_3_RA', 'ORIGIN_WND_SPEED_RATE_3_RA',
 'ORIGIN_CIG_CEILING_HEIGHT_DIMENSION_3_RA', 'ORIGIN_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION_3_RA', 'ORIGIN_GD4_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA',
 'ORIGIN_WIND_GUST_OBSERVATION_SPEED_RATE_3_RA', 'DEST_WND_SPEED_RATE_3_RA', 'DEST_AA1_LIQUID_PRECIPITATION_PERIOD_QUANTITY_3_RA', 'DEST_RUNWAY_VISUAL_RANGE_OBSERVATION_VISIBILITY_DIMENSION_3_RA',
 'DEST_GA3_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION_3_RA', 'DEST_GD1_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION_3_RA', 'DEST_WIND_GUST_OBSERVATION_SPEED_RATE_3_RA', 'DEP_DEL15']

# COMMAND ----------

ra_train_data_2 = spark.sql('SELECT * FROM flights_and_weather_train_ra_processed')[cor_picked_cols]
ra_validation_data_2 = spark.sql('SELECT * FROM flights_and_weather_validation_ra_processed')[cor_picked_cols]
ra_test_data_2 = spark.sql('SELECT * FROM flights_and_weather_test_ra_processed')[cor_picked_cols]

# COMMAND ----------

# create an encoding pipeline based on information from our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(ra_train_data_2,'DEP_DEL15'))
encoding_pipeline = encoding_pipeline.fit(ra_train_data_2)
# apply the transformations to our train data
transformed_train_data_2 = encoding_pipeline.transform(ra_train_data_2)['features', 'label']

# COMMAND ----------

# train a model on our transformed train data
startTime = time.time()
lr_2 = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 30, regParam = 0.001, elasticNetParam = 0.25, standardization = False)
lr_model2 = lr_2.fit(transformed_train_data_2)
lr_2_train_preds = lr_model2.transform(transformed_train_data_2)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes")       

# COMMAND ----------

# MAGIC %md ### Let's evaluate our training data

# COMMAND ----------

lr_2_train_metrics = evaluation_metrics(lr_2_train_preds, "Logistic Regression on training data")
display(lr_2_train_metrics)

# COMMAND ----------

# MAGIC %md ### Let's evaluate our validation data

# COMMAND ----------

# apply the encoding transformations from our pipeline to the validation data
transformed_validation_data_2 = encoding_pipeline.transform(ra_validation_data_2)['features', 'label']

# COMMAND ----------

# run the fitted model on the transformed validation data
lr_2_validation_preds = lr_model2.transform(transformed_validation_data_2)

# COMMAND ----------

# display our evaluation metrics
lr_2_validation_metrics = evaluation_metrics(lr_2_validation_preds, "Logistic Regression on validation data")
display(lr_2_validation_metrics)

# COMMAND ----------

# MAGIC %md ## Running best model on Test

# COMMAND ----------

# MAGIC %md ## Logistic Regression RA's Full 586 Columns

# COMMAND ----------

train_data = spark.sql('SELECT * FROM flights_and_weather_train_ra_processed')
validation_data = spark.sql('SELECT * FROM flights_and_weather_validation_ra_processed')
test_data = spark.sql('SELECT * FROM flights_and_weather_test_ra_processed')

cols_to_drop = ['DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP', 'TAIL_NUM', 'ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'FL_DATE', 'ORIGIN_WEATHER_KEY', 'DEST_WEATHER_KEY', 'ORIGIN_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'ORIGIN_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'ORIGIN_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'ORIGIN_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION', 'ORIGIN_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION', 'ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE', 'ORIGIN_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE', 'ORIGIN_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'ORIGIN_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_AL2_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'DEST_AL3_SNOW_ACCUMULATION_PERIOD_QUANTITY', 'DEST_AL3_SNOW_ACCUMULATION_DEPTH_DIMENSION', 'DEST_GA6_SKY_COVER_LAYER_BASE_HEIGHT_DIMENSION', 'DEST_GD5_SKY_COVER_SUMMATION_STATE_HEIGHT_DIMENSION', 'DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_UPPER_RANGE_ATTRIBUTE', 'DEST_SKY_CONDITION_OBSERVATION_BASE_HEIGHT_LOWER_RANGE_ATTRIBUTE', 'DEST_GG1_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG2_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG3_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION', 'DEST_GG4_BELOW_STATION_CLOUD_LAYER_TOP_HEIGHT_DIMENSION']

train_data = train_data.drop(*cols_to_drop)
validation_data = validation_data.drop(*cols_to_drop)
test_data = test_data.drop(*cols_to_drop)

# COMMAND ----------

# create an encoding pipeline based on information from our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(train_data,'DEP_DEL15'))
encoding_pipeline = encoding_pipeline.fit(train_data)
transformed_train_data = encoding_pipeline.transform(train_data)

# COMMAND ----------

# train a model on our transformed train data
startTime = time.time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 30, regParam = 0.001, elasticNetParam = 0.25, standardization = False)
model = lr.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes")       

# COMMAND ----------

train_metrics = evaluation_metrics(train_preds, "Logistic Regression on training data")
display(train_metrics)

# COMMAND ----------

# apply the encoding transformations from our pipeline to the validation data
transformed_validation_data = encoding_pipeline.transform(validation_data)['features', 'label']

# run the fitted model on the transformed validation data
validation_preds = model.transform(transformed_validation_data)

# display our evaluation metrics
validation_metrics = evaluation_metrics(validation_preds, "Logistic Regression on validation data")
display(validation_metrics)

# COMMAND ----------

# apply the encoding transformations from our pipeline to the test data
transformed_test_data = encoding_pipeline.transform(test_data)['features', 'label']

# run the fitted model on the transformed test data
test_preds = model.transform(transformed_test_data)

# display our evaluation metrics
test_metrics = evaluation_metrics(test_preds, "Logistic Regression on test data")
display(test_metrics)

# COMMAND ----------

AUC(model, test_preds)
ROC(model, test_preds)

# COMMAND ----------



# COMMAND ----------

