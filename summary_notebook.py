# Databricks notebook source
# MAGIC %md # W261 Final Project - Airline Delays and Weather - Project Summary
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Summer 2020`__
# MAGIC ### Team 1:
# MAGIC * Sayan Das
# MAGIC * Kevin Hartman
# MAGIC * Hersh Solanki
# MAGIC * Nick Sylva

# COMMAND ----------

# MAGIC %md # Referenced Notebooks
# MAGIC ## EDA and Feature Engineering Notebook
# MAGIC * <Insert Link Here>  
# MAGIC ## Modeling Notebook
# MAGIC * <Insert Link Here>  

# COMMAND ----------

# MAGIC %md # Table of Contents
# MAGIC ## 1. Introduction/Question Formulation
# MAGIC ## 2. EDA & Discussion of Challenges
# MAGIC ## 3. Feature Engineering
# MAGIC ## 4. Algorithm Exploration
# MAGIC ## 5. Algorithm Implementation
# MAGIC ## 6. Conclusions
# MAGIC ## 7. Application of Course Concepts

# COMMAND ----------

# MAGIC %md # Introduction/Question Formulation 

# COMMAND ----------

# MAGIC %md ## Objective
# MAGIC 
# MAGIC Our goal is to understand flight delays given information about the flight, and surrounding weather information. Flight delays are common, however, then can be a result of various amount of factors. We want to identify what these features may be, and create a model that can accurately predict whether the flight will be delayed by atleast 15 minutes, or not. We attempt to use multiple models, as well as hyperparameter turning to get our final result.
# MAGIC 
# MAGIC ## Testing Approach 
# MAGIC 
# MAGIC We decided to partition our dataset by years, where our training dataset consisted of data from 2015-2017, our validation data is from 2018, and our testing data is from 2019. The reason we split the data like this is because we don't want to to use future data to predict past data.  Furthermore, it is essential that all of the testing data is not sampled from any data that is in the future, for otherwise the model would not be practically useful.  
# MAGIC 
# MAGIC Note for the evaluation metric that is used, it appears that other literature on this topic have used the accuracy metric as the ideal metric to gauge model performance (Ye, 2020, Abdulwahab, 2020).  Therefore, our team has also decided to adopt accuracy as the de facto metric of comparison to determine which model is performing the best.  In addition, while it is important to minimize both the number of false positives and false negatives, our group has decided to prioritize minimizing the number of false positives, as we do not want to tell an individual that there is a delay when there actually is not a delay, as that could cause the individual to miss the flight which is the worst possible scenario.  Indeed, even if there a large number of false negatives, which implies that we tell the individual that there is a delay even though there is not a delay, then this outcome does not have as a negative impact on the user compared to them potentially missing their flight.  
# MAGIC 
# MAGIC ## Baseline Model
# MAGIC 
# MAGIC For our baseline model, we decided to use a logistic regression model which just predicts on the binary variable of a delay greater than 15 minutes (DEP_DEL15).  This raw model was able to produce an accuracy score of 0.8154 on the validation data, but it unfortunately predicted a fairly large number of false positives, namely 4053.  This is large with respect to other baseline model of random forest, which produced a similar accuracy score of 0.8156.  However, it produced a much smaller number of false positives, namely 426.
# MAGIC 
# MAGIC Overall, in order to be practically useful, our model should have an accuracy score that exceeds 0.8 with a miniscule number of false positives.  If there are a large number of false positives, then the veracity of our model is incredibly questionable.  Moreover, a distinguishing factor between the varying performances among models will be the number of predictions.  For instance, models can have high accuracy scores, but they may not generate a large number of predictions in the first place.  If the model is unable to create a prediction, then indeed the practicality of the model is quite dubious.  
# MAGIC 
# MAGIC ## Limitations
# MAGIC 
# MAGIC Some of the limitations of our model include our model not predicting on different severities of delay.  From a user perspective, it is more beneficial have different levels of delay based on how long their will be a delay.  By only predicting whether there is a delay or not, it is difficult for the individual to truly manage their schedule to accomodate for the flight delay.  This distinction between different magnitudes of delay will especially have prominent impacts on airports that have a lot of traffic.

# COMMAND ----------

# MAGIC %md # EDA & Discussion of Challenges

# COMMAND ----------

# MAGIC %md # Feature Engineering

# COMMAND ----------

# MAGIC %md # Algorithm Exploration

# COMMAND ----------

# MAGIC %md # Algorithm Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt
import time
import numpy as np 
import pandas as pd
import seaborn as sns
from pytz import timezone 
from datetime import  datetime, timedelta 
import os
from delta.tables import DeltaTable


# Model Imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder, PCA
from pyspark.ml.classification import LogisticRegression

%matplotlib inline
sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up Data Access

# COMMAND ----------

username = "kevin"
dbutils.widgets.text("username", username)
spark.sql(f"CREATE DATABASE IF NOT EXISTS airline_delays_{username}")
spark.sql(f"USE airline_delays_{username}")

flights_and_weather_pipeline_loc = f"/airline_delays/{username}/DLRS/flights_and_weather_pipeline/"

spark.conf.set("spark.sql.shuffle.partitions", 8)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull in Data

# COMMAND ----------

data = spark.sql('SELECT * FROM flights_and_weather_pipeline_processed')
toy_data_sample = data.sample(False, 0.00001, seed = 42)

# COMMAND ----------

# split data into train/test/validation
toy_data_sample_train = toy_data_sample.where('YEAR IN (2015, 2016, 2017)')
toy_data_sample_validation = toy_data_sample.where('YEAR = 2018')
toy_data_sample_test = toy_data_sample.where('YEAR = 2019')

# COMMAND ----------

#subset the data on a few select columns for the sake of our toy example
toy_data_sample_train = toy_data_sample_train.select(['DEP_DEL15','ORIGIN_WND_SPEED_RATE',\
                                                     'ORIGIN_CIG_CEILING_HEIGHT_DIMENSION',\
                                                     'ORIGIN_VIS_DISTANCE_DIMENSION',\
                                                     'ORIGIN_TMP_AIR_TEMP','ORIGIN_SLP_SEA_LEVEL_PRES'])
toy_data_sample_validation = toy_data_sample_validation.select(['DEP_DEL15','ORIGIN_WND_SPEED_RATE',\
                                                     'ORIGIN_CIG_CEILING_HEIGHT_DIMENSION',\
                                                     'ORIGIN_VIS_DISTANCE_DIMENSION',\
                                                     'ORIGIN_TMP_AIR_TEMP','ORIGIN_SLP_SEA_LEVEL_PRES'])
toy_data_sample_test = toy_data_sample_test.select(['DEP_DEL15','ORIGIN_WND_SPEED_RATE',\
                                                     'ORIGIN_CIG_CEILING_HEIGHT_DIMENSION',\
                                                     'ORIGIN_VIS_DISTANCE_DIMENSION',\
                                                     'ORIGIN_TMP_AIR_TEMP','ORIGIN_SLP_SEA_LEVEL_PRES'])

# COMMAND ----------

display(toy_data_sample_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handle Missing Values
# MAGIC There are a few null values, so we need to impute them. We developed a couple routines for this process.

# COMMAND ----------

def make_imputation_dict(df):
  '''
  This function creates a dictionary containing the mean or mode 
  for numerical and categorical features, respectively.
  Returns: dict
  '''
  impute_dict = {}
  
  #all categorical columns are of string datatype
  cat_cols = [item[0] for item in df.dtypes if item[1].startswith('string')]  
  # all numerical columns are of integer or double datatypes
  num_cols = [item[0] for item in df.dtypes if item[1].startswith('int') | item[1].startswith('double')] 
  
  #iterate over categorical columns and calculate mode
  for x in cat_cols:                  
    mode = df.groupBy(x).count().sort(f.col("count").desc()).collect()
    impute_dict[x] = mode[0][0]

  # iterate over numerical columns and caluclate means
  for i in num_cols:
    mean_value = df.select(f.mean(i).cast(DoubleType())).collect()
    impute_dict[i] = mean_value[0][0]
    
  return impute_dict

# COMMAND ----------

def impute_missing_values(df, impute_dict):
  '''
  This function uses the imputation dictionary created by the 
  make_imputation_dict function to replace missing values.
  '''
  #Build a list of columns that have missing values and their counts
  missing_count_list = []
  for c in df.columns:
      if df.where(f.col(c).isNull()).count() > 0:
          tup = (c,int(df.where(f.col(c).isNull()).count()))
          missing_count_list.append(tup)

  missing_column_list = [x[0] for x in missing_count_list]
  
  #subselect df on missing vlues
  missing_df = df.select(missing_column_list)

  # break apart missing columns into categorical and numeric. Report which of each type are missing,
  missing_cat_columns = [item[0] for item in missing_df.dtypes if item[1].startswith('string')]
  print("\nCategorical Columns with missing data:", missing_cat_columns)

  missing_num_columns = [item[0] for item in missing_df.dtypes if item[1].startswith('int') | item[1].startswith('double')]
  print("\nNumerical Columns with missing data:", missing_num_columns)
  
  # Fill the missing categorical values with the most frequent category (mode)
  for x in missing_cat_columns:                  
    mode = impute_dict[x]
    if mode:
      df = df.withColumn(x, f.when(df[x].isNull(), f.lit(mode)).otherwise(df[x]))
    else:
      df = df.withColumn(x, f.when(df[x].isNull(), 'None').otherwise(df[x]))

  # Fill the missing numerical values with the average of each #column
  for i in missing_num_columns:
    mean_value = impute_dict[i]
    if mean_value:
        df = df.withColumn(i, f.when(df[i].isNull(), mean_value).otherwise(df[i]))
    else:
        df = df.withColumn(i, f.when(df[i].isNull(), 0).otherwise(df[i]))
        
  return df

# COMMAND ----------

#create the imputation dictionary, note, avoiding selecting the target field, DEP_DEL15
impute_dict = make_imputation_dict(toy_data_sample_train.select(['ORIGIN_WND_SPEED_RATE',\
                                                     'ORIGIN_CIG_CEILING_HEIGHT_DIMENSION',\
                                                     'ORIGIN_VIS_DISTANCE_DIMENSION',\
                                                     'ORIGIN_TMP_AIR_TEMP','ORIGIN_SLP_SEA_LEVEL_PRES']))

# COMMAND ----------

#impute the datasets. Note, we use the same imputation dictionary created from training data
toy_data_sample_train = impute_missing_values(toy_data_sample_train, impute_dict)
toy_data_sample_validation = impute_missing_values(toy_data_sample_validation, impute_dict)
toy_data_sample_test = impute_missing_values(toy_data_sample_test, impute_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Set Up
# MAGIC In order to feed our model, we need to vectorize our data. We use Spark 3.0's Pipeline construct to do that.

# COMMAND ----------

# Use the OneHotEncoderEstimator to convert categorical features into one-hot vectors
# Use VectorAssembler to combine vector of one-hots and the numerical features
# Append the process into the stages array to reproduce

def create_encoding_stages(data, label_name):
  
  cat_cols = [item[0] for item in data.dtypes if item[1].startswith('string')]
  numeric_cols = [item[0] for item in data.dtypes if item[1].startswith('int') | item[1].startswith('double')]
  cat_cols.remove(label_name)
  
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

# create an encoding pipeline based on our training data
encoding_pipeline = Pipeline(stages = create_encoding_stages(toy_data_sample_train,'DEP_DEL15'))
encoding_pipeline = encoding_pipeline.fit(toy_data_sample_train)

# COMMAND ----------

# apply our Pipeline transformations to our datasets
transformed_train_data = encoding_pipeline.transform(toy_data_sample_train)['features', 'label']
transformed_validation_data = encoding_pipeline.transform(toy_data_sample_validation)['features', 'label']
transformed_test_data = encoding_pipeline.transform(toy_data_sample_test)['features', 'label']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Algorithm Implementation
# MAGIC Our best performance consistently came from logistic regression. The toy dataset we are using has five predictor variables and a target variable.  
# MAGIC **Predictor Variables:**
# MAGIC  * Origin Wind Speed Rate (integer, \\(x\_1\\)): Current wind speed at the origin airport of a flight. 
# MAGIC  * Origin Ceiling Height (integer, \\(x\_2\\)): Cloud ceiling height at the origin airport of a flight.
# MAGIC  * Origin Visibility Distance (integer, \\(x\_3\\)): Visibility distince at the origin airport of a flight.
# MAGIC  * Origin Air Temperature (integer, \\(x\_4\\)): Air Temperature at the origin airport of a flight.
# MAGIC  * Origin Barometric Pressure (integer, \\(x\_5\\)): Barometric Pressure relative to sea level at the origin airport of a flight.  
# MAGIC   
# MAGIC **Target Variable:**
# MAGIC  * Departure Delay Indicator (binary, \\(y\\)): Indicates whether or not a flight departure was delayed 15 mins or more.
# MAGIC  
# MAGIC  # FILL OUT MORE MATH STUFF HERE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Our Model

# COMMAND ----------

# train a model on our transformed train data
startTime = time.time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 100, regParam = 0.001, standardization = False)
model = lr.fit(transformed_train_data)
train_preds = model.transform(transformed_train_data)
endTime = time.time()
print(f"The training time of the Logistic Regression model is: {(endTime - startTime) / (60)} minutes") 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

def evaluation_metrics(predictions, model_name, metrics_dataframe):
  '''
  This function evaluations our model predictions and reports standard metrics.
  '''
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

# COMMAND ----------

df_cols = ['Model','accuracy', 'precision', 'recall', 'f1_score', 'true_positives', 'true_negatives', 'false_positives', 'false_negatives']
metrics_dataframe = pd.DataFrame(columns = df_cols)

# COMMAND ----------

metrics_dataframe = evaluation_metrics(train_preds, "Logistic Regression on Training Data", metrics_dataframe)

# COMMAND ----------

display(metrics_dataframe)

# COMMAND ----------

#run the model and evaluate metrics on the validation data
validation_preds = model.transform(transformed_validation_data)
metrics_dataframe = evaluation_metrics(validation_preds, "Logistic Regression on Validation Data", metrics_dataframe)

# COMMAND ----------

display(metrics_dataframe)

# COMMAND ----------

#run the model and evaluate metrics on the test data
test_preds = model.transform(transformed_test_data)
metrics_dataframe = evaluation_metrics(test_preds, "Logistic Regression on Test Data", metrics_dataframe)

# COMMAND ----------

display(metrics_dataframe)

# COMMAND ----------

# MAGIC %md ## Analysis of Implementation
# MAGIC @Nick please answer the following when done:
# MAGIC Use this toy example to explain the math behind the algorithm that you will perform. Apply your algorithm to the training dataset and evaluate your results on the test set. 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC Link to comparison chart: 'insert path to Github of Model Results Doc'
# MAGIC 
# MAGIC Note that the above chart contains all of our trials of the three main models tested: Logistic Regression, Random Forests, and Gradient-Boosted Trees.  We highlighted our best performing model from each section in yellow for easy visibility, but the table does a great job of showing the different permutations of models tested along with their associated results.
# MAGIC 
# MAGIC After an absurd amount of testing of our three main models, our group had a multitude of revelations.  Firstly, we realized that it is quite difficult to train the Random Forest model consistently due to the inefficiency of the algorithm, especially compared to the other algorithms of Logistic Regression and Gradient-Boosted Trees.  Overall, our group made the following observations (all relative to the Validation Dataset Performance):
# MAGIC 
# MAGIC * The actual accuracy results of all three models were quite comparable.
# MAGIC * Random Forest generated a minimal number of predictions (<1000) for both True and False Positives.
# MAGIC * Logistic Regression trained on our 586 selected features generated the largest number of True Positive (10427) and False Positive (6319) predictions, while still maintaining a top accuracy score (0.8168) and a respectable precision score of (0.6227).  
# MAGIC * Gradient-Boosted Trees trained on our 495 selected features (does not include Weather Rolling Average features) and PageRank also generated a significant number of  True Positive (19113) and False Positive (16382) predictions.  
# MAGIC * Logistic Regression trains incredibly fast when only having a limit number of features.  
# MAGIC 
# MAGIC Scalability wise, we encountered issues when training our Random Forest model.  We were only able to fully train our Random Forest model when using our initial Baseline selected features (35 Total).  When trying to train Random Forest with our 495 or 586 features, our cluster simply crashed.  In fact, even performing PCA on our features was unsuccesful: there were still too many columns.  In order to solve this problem, we increased the compute power of our cluster and also tried making it more memory optimized, but Random Forest still crashed.  For all of the models, we ran cross-validation to identify the optimal hyperparameters, so we would not have to consistently run each model arbitrarily over and over again.  This was beneficial in optimizing our training time.  Regarding model run time for 35 features (as Random Forest failed on our 495 and 586 features), our group noticed the following for validation: 
# MAGIC 
# MAGIC * Logistic Regression took around 2 minutes
# MAGIC * Random Forest took around 12 minutes
# MAGIC * Gradient Boosted Trees took around 4 minutes
# MAGIC 
# MAGIC In order to further optimize our training time, we increased the compute power of our clusters.  Furthermore, we also attempted to train Gradient-Boost Trees on GPUs to further speed up our training time, but we ran into problems with the worker nodes timing out due to heartbeat issues.  Based on the dataset and problem, it is important for training time to be incredibly fast.  Users need to be informed of flight delays ASAP, so longer training times, such as with Random Forest, would not be practical for our users.  It is important for our predictions to be as close to real-time as possible, for data can change quite frequently.  For example, our weather data can change quite frequently, and therefore we would ideally want a robust and speedily trained model that is impervious to numerous changes.  This model will need to be retrained when there are significant changes in dynamic variables such as the weather, e.g if the wind speed changed from 5 mph to 40 mph due to a tornado (an incredibly contrived example).  Overall, however, the model will not need to be retrained on an incredibly consistent basis, but primarily when there are significant changes to the dynamic portion of our features, which will most likely be primarily attributed to the weather data.  

# COMMAND ----------

# MAGIC %md # Application of Course Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC Our final project incorporated a large variety of course concepts.  For example, one key concept that our course taught was scalability.  This concept was especially apparent when modeling.  For instance, it is much easier to model using Logistic Regression compared to Random Forests, for from a scalability perspective, Random Forest models are computationally much more expensive.  Our group found this especially true when adding more and more features to our models; our Random Forest model frequently crashed as we added numerous features (over 100), whereas our Logistic Regression model ran just fine.  Note that these incidents occurred regardless of how much Computation power we provided to our clusters.  In fact, our group even named a cluster "op_cluster" (where op stands for overpowered), a cluster that was maxed out in everything possible within our AWS vCPU limit, and our Random Forest model STILL crashed.  Furthermore, it was quite impressive to see how I/O scaled in our different Databricks clusters.  Spark was able to handle the ridiculously large number of read and write operations quite well, where our main out-of-memory errors came when modeling as mentioned before.  
# MAGIC 
# MAGIC Additionally, our group applied the course concept of Graph Algorithms.  Indeed, we created a PageRank feature that essentially applied an inverse page rank algorithm based on the out degree of nodes, where more nodes (aka more routes) imply more weight to the airport, which implies a higher chance of delay. This feature was derived by creating a nearest adjacency matrix that is a hash map where the key is the origin airport and the value is the destination airports and counts that reflect the number of times visited from the origin. This constructs a map of which airports were visited most frequently and which airports have the most departure traffic (out degree of nodes).  Upon the completion of Homework 5, our group was inspired to apply our understanding of PageRank by using the concepts of nodes and edges with airports and routes.  Although the model improvement with the addition of the feature was marginal, the addition of PageRank decreased the number of False Positives relative to our Baseline Logistic Regression model, albeit at the expense of the number of True Positives.  However, as mentioned at the beginning, our group's focus was on decreasing the number of false positives, as we deemed this to be a more substantially negative impact to the user.
# MAGIC 
# MAGIC Moreover, our group also applied the concept of Data Systems and Pipelines.  In fact, one of our group's earliest objectives was to establish a working modeling pipeline.  This was particuarly noticable in the halfway presentation checkpoint, where we noticed that we appeared to be the only group to establish a modeling pipeline end-to-end.  While it was a big struggle debugging the many vectorization and encoding issues late at night, it was incredibly worth it when testing out a myriad of models.  Our group transformed data by one-hot encoding categorical data, and we normalized numerical data using a standard scaler. These transformation steps were staged in our pipeline so they can be repeated for validation and test evaluation.  Additionally, we imputed null values in both categorical and numerical columns.  
# MAGIC 
# MAGIC Finally, we also applied the class concept of model optimization.  While many rookie Data Scientists are in a rat race of adding on as many features as possible, our group meticulously selected our features; for example, we handpicked only key dimensions from the massive weather dataset.  In addition to this, we applied Principal Component Analysis to our features to ensure we were not using anything that did not contribute any meaningful value.  Also, our group applied cross-validation to our models to determine the optimal hyperparameters that will yield the best results.  Moreover, we made sure to really focus on tuning and optimizing more scalable models that were optimal given our time constraint; for instance, we spent quite a bit of time optimizing our Logistic Regression Model compared to our Random Forest model, as the Random Forest model takes a ridiculously large amount of time to run and yields comparable if not inferior performance.
# MAGIC 
# MAGIC In conclusion, these are only some of the course concepts our group applied while working on this project.  It was truly a pleasure to work as a team on such an arduous task.  Our group has only one regret: we wish we had more time to work on it.  We are very proud of our work, and we are very thankful to have taken this class.  Thank you very much.
