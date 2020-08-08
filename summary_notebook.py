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
# MAGIC * https://dbc-50712828-d793.cloud.databricks.com/?o=4611511999589276#notebook/2431210803799670/command/1672490400676146
# MAGIC 
# MAGIC ## Modeling Notebook
# MAGIC * https://dbc-b08f19ef-aaab.cloud.databricks.com/?o=8795677657115827#notebook/4222972772477374/command/760349354865761

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

# MAGIC %md 
# MAGIC ## Objective
# MAGIC Our goal is to understand flight departure delays given information about the flight and weather conditions at the origin and destination. Flight delays are common, however, their exact cause is a result of many factors. We want to identify what these features may be and create a model that can accurately predict whether or not a flight will be delayed by atleast 15 minutes. We attempt to use multiple models, hyperparameter turning, and cross-validation to get our final result.
# MAGIC 
# MAGIC ## Testing Approach 
# MAGIC 
# MAGIC ### Data Partitioning
# MAGIC We partitioned our dataset by years, where our training dataset consisted of data from 2015-2017, our validation data is from 2018, and our testing data is from 2019. The reason we split the data like this is because we don't want to to use future data to predict past data.  Furthermore, it is essential that any features used are sampled from past data. Otherwise, the model will  not be practically useful.  
# MAGIC 
# MAGIC ### Evaluation Metrics
# MAGIC Literature on this topic has used accuracy as the ideal metric to gauge model performance (Examples: Ye, 2020; Abdulwahab, 2020).  Following the lead of the literature, our team intitially decided to adopt accuracy as the de facto metric for comparing performance between models. However, when considering the business case of running an airport or airline, we decided that precision and the false positive rate are the most important metrics. Minimizing the number of false positives is important because we do not want to tell an individual that there is a delay when there actually is not a delay, as that could cause the individual to miss the flight. Even if there a large number of false negatives, which implies that we tell the individual that there is a delay even though there is not a delay, then this outcome does not have as big of a negative impact on the user compared to potentially missing their flight. In an effort to guide model development, we also consider additional standard metrics including, Recall, F1 Score, and the Area Under the Receiver-Operator Characteristic Curve (AUROC).   
# MAGIC 
# MAGIC ## Baseline Model
# MAGIC For our baseline model we decided to use logistic regression, which predicts on the binary variable of a delay greater than 15 minutes (DEP_DEL15). This model utilized 35 features from the flights and weather data and produced an accuracy score of 0.8154 on the validation data, but it unfortunately predicted a fairly large number of false positives, 4,053. 
# MAGIC 
# MAGIC Overall, in order to be practically useful, our model should have an accuracy score that exceeds 0.8 while minimizing false positives.  If there are a large number of false positives, then the veracity of our model is questionable.  Moreover, a distinguishing factor for model performance will be the number of predicted delays.  For instance, models can have high accuracy scores due to the imbalance of the classes (far fewer flights are delayed than on time), but they may not predict delays in the first place.  If the model is unable to predict a delay, then the practicality of the model is dubious.  
# MAGIC 
# MAGIC ## Limitations
# MAGIC 
# MAGIC Some of the limitations of our model is that the model does not predict the magnitude of the delay.  From a user perspective, it is beneficial to understand the magnitude of the delay.  By only predicting whether or not there is a delay, it is difficult for the individual to manage their schedule to accomodate for the flight delay.  This distinction between different magnitudes of delay will especially have prominent impacts on airports that have a lot of traffic.

# COMMAND ----------

# MAGIC %md # EDA & Discussion of Challenges

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE:** Please note that a deep dive of our EDA can be found in our EDA and Feature Engineering Notebook here: https://dbc-50712828-d793.cloud.databricks.com/?o=4611511999589276#notebook/2431210803799670/command/1672490400676146
# MAGIC 
# MAGIC ### EDA  
# MAGIC At a very high level, the goals of our EDA consist of: identifying missing data in both the airlines and weather datasets, joining the weather dataset with the airlines dataset and creating haverstine distance and other functions to map weather stations with airports, and identifying correlated features for selection and engineering.  
# MAGIC 
# MAGIC We conducted preliminary EDA on key variables of interest, namely those most correlated with our outcome variable. We imported and modified an EDA function one of the team members had written in W207 to analyze the airline data, giving us the different categories, or the mean/median if it’s a continuous variable. We then analyzed the historical departures and arrivals or flights, as well as the amount of time that was made up/lost in the air. Next, we looked at the weather data and did a similar analysis to our airline data. 
# MAGIC 
# MAGIC ### Challenges
# MAGIC One of the key challenges identified is joining the flights and weather datasets. In order to accomplish this, we created a compound key based on the closest weather station to each airport and the time of weather observations in the weather dataset. We use two different compound keys, one for each of the origin and destination airports. Adjusting the timezone for measurements and flight times is necessary in order to achieve a consistent join.
# MAGIC 
# MAGIC To narrow our focus to only the most important features and combat multicollinearity, we created heat maps to visualize which features were correlated heavily with each other. Moreover, we also computed correlation scores for each feature with respect to the our Departure Delay variable to create a model focused on only the most significant features. 
# MAGIC 
# MAGIC There are numerous missing values in both datasets. Specifics can be found in our deep dive notebook linked above.    
# MAGIC 
# MAGIC After completing these aforementioned tasks, one of the key challenges our group faced was with regards to imputation.  It was absolutely imperative that we converted variables to their appropriate type, and furthermore, that we imputed variables using the mean, mode, or NULL value accordingly based on the values that exist.  This was particularly challenging, for many of the fields in the weather data had unique encodings to represent NULL values that had to be decoded manually.  

# COMMAND ----------

# MAGIC %md # Feature Engineering
# MAGIC 
# MAGIC Please note that a deep dive of our Feature Engineering can be found in our EDA and Feature Engineering Notebook here: https://dbc-50712828-d793.cloud.databricks.com/?o=4611511999589276#notebook/2431210803799670/command/1672490400676146
# MAGIC 
# MAGIC Some of the proposed features that our group worked on include the following:
# MAGIC * **ORIGIN_PAGERANK/DEST_PAGERANK**: This feature applies PageRank based on a graph of airports generated from the training set of the flight data, where more nodes (aka more routes) imply more weight to the airport, which implies a higher chance of delay. This feature is derived by creating an adjacency matrix that is a hash map where the key is the origin airport and the value is the destination airports and counts that reflect the number of times visited from the origin. This constructs a map of which airports were visited most frequently and which airports have the most traffic.
# MAGIC * **LATE_ARRIVAL_DELAY**: Each flight has a TAIL_NUM indicating which plane flew a route. We can leverage paritioning and window functions to order flights first by their TAIL_NUM and then by the timestamp of the flight. Then, if we are looking at flight \\(i\\), we use a lag function to look at flight \\(i-1\\) and attach its UTC arrival time and whether or not it was delayed by 15 minutes or more on arrival. Using these two pieces of data, we then compare the arrival time to the inference time (2 hours prior to the flight) and see if we could feasibly have that knowledge when trying to predict a delay. 
# MAGIC Two other features were discussed but ultimately not implemented:.
# MAGIC * **IN_FLIGHT_AIR_DELAY**: quantifies the amount of time that is either made up or lost in the air. This feature is derived from the underlying data by taking the difference between arrival and departure delay. This feature was not used because it relies on future data. We cannot know how much time a flight will lose or make up in the air before the flight happens.
# MAGIC * **SEASONAL_DELAY**: This feature will use the date of departure to identify whether the flight delay could be exacerbated due to the seasonal cycle (e.g holiday travels). This feature will be derived from the underlying data by analyzing the flight delay variables with respect to the date of travel. This feature was not used because other time variables included, such as MONTH and DAY_OF_MONTH contain the same information that this feature would.
# MAGIC 
# MAGIC We computed 3-hr rolling averages of all numerical weather features. Here are a few examples because there are 45 that were ultimately included in our models: 
# MAGIC * ORIGIN/DEST_AVG_WND_SPEED: moving average of observed wind speed over over 3 hrs. 
# MAGIC * ORIGIN/DEST_AVG_CIG_HEIGHT_DIMENSION: moving average ceiling height over 3 hrs.
# MAGIC * ORIGIN/DEST_AVG_PRECIP_RATE: moving average precipitation over 3 hrs.
# MAGIC * ORIGIN/DEST_AVG_VIS_OBS_DISTANCE_DIMENSION: moving average visibility distance observation over 3 hrs.
# MAGIC * ORIGIN/DEST_AVG_SNOW_DEPTH: moving average snow depth over 3 hrs.  
# MAGIC 
# MAGIC In order to compute these features, we embarked on a significant research and engineering effort in order to unpack the features and determine whether a feature is categorical or numeric. The justification for creating these weather features is that when determining whether a flight should be delayed or not in advance, it may be advantageous to know the average weather conditions in addition to the point in time weather conditions. 
# MAGIC 
# MAGIC After engineering all of these features and including many of the other flight and weather features, our biggest model included a total of 576 features. At that scale, feature selection becomes very important. We utilized a number of techniques to reduce the number of features in our models including L1 and L2 regularization, and PCA. We also attempted to run a few models with a very small subset of features that were the most highly correlated with our outcome variable. 

# COMMAND ----------

# MAGIC %md # Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC Please note that a deep dive of our Algorithm Exploration can be found in our modeling notebook here: https://dbc-b08f19ef-aaab.cloud.databricks.com/?o=8795677657115827#notebook/4222972772477374/command/760349354865761
# MAGIC 
# MAGIC Our group applied three algorithms to our training set: Logistic Regression, Random Forest, and Gradient-Boosted Trees.  Our group expected Logistic Regression to train the fastest, Random Forest to take the longest, and Gradient-Boosted Trees to generate the best results.  From past experience with modeling in various Kaggle challenges and W207, our group had very high hopes for Gradient-Boosted Trees.  Some of the trade-offs that our group experienced was that while Random Forest initially had the highest accuracy scores, it took an incredibly long time to train, as expected.  Results wise, Random Forest yielded a miniscule number of predictions for both true and false positives.  Gradient-Boosted Trees, while it predicted a lot of delays it had a fairly low accuracy and precision score relative to the other models.  Most importantly, it generated a large number of false positives, the opposite of one of our main directives.
# MAGIC 
# MAGIC Ultimately, all three algorithms yielded fairly similar accuracy and precision scores.  Due to the fast training time and fairly large number of true positive predictions with a relatively smaller number of false positive predictions, we focused on optimizing the Logistic Regression model.  When fine-tuning the Random Forest model, we ran into significant computational limitations that were not resolved by scaling up or out. Due to the lackluster performance of the Gradient-Boosted Trees model, we decided it was not worth the time or compute to fine-tune it.  The baseline results for our three algorithms are summarized as follows:
# MAGIC 
# MAGIC * Baseline Logistic Regression Training
# MAGIC   * Accuracy: 0.8203790056456892
# MAGIC   * Precision: 0.5221751127262938
# MAGIC   * True Positives: 7759
# MAGIC   * False Positives: 7100
# MAGIC * Baseline Logistic Regression Validation
# MAGIC   * Accuracy: 0.8154005246740127
# MAGIC   * Precision: 0.4392639734366353
# MAGIC   * True Positives: 3175
# MAGIC   * False Positives: 4053
# MAGIC * Baseline Random Forest Training
# MAGIC   * Accuracy: 0.8213231793813259
# MAGIC   * Precision: 0.909340248341894
# MAGIC   * True Positives: 18235
# MAGIC   * False Positives: 1818
# MAGIC * Baseline Random Forest Validation
# MAGIC   * Accuracy: 0.8155832561642371
# MAGIC   * Precision: 0.6616362192216044
# MAGIC   * True Positives: 833
# MAGIC   * False Positives: 426
# MAGIC * Baseline Gradient-Boosted Trees Training
# MAGIC   * Accuracy: 0.8204792469618283
# MAGIC   * Precision: 0.6080229757272558
# MAGIC   * True Positives: 6563
# MAGIC   * False Positives: 4231
# MAGIC * Baseline Gradient-Boosted Trees Validation
# MAGIC   * Accuracy: 0.8154005246740127
# MAGIC   * Precision: 0.4392639734366353
# MAGIC   * True Positives: 3175
# MAGIC   * False Positives: 4053
# MAGIC 
# MAGIC Please note that the specific results for each run is found below in our comparions chart mentioned in the Conclusions section.  

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
# MAGIC ## Algorithm Choice
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
# MAGIC In general, logistic regression is used to calculate the log-odds that a binary variable equals 1:  
# MAGIC \\(l(y=1) = log\frac{p}{1-p} =\beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_3 + \beta_4x_4 + \beta_5x_5\\)  
# MAGIC where \\(p = P(y=1)\\)
# MAGIC Based on this, we can use the coefficients (\\(\beta_i\\)) to understand the contribution to the log-odds that each predictor has on the target. In other words, we can estimate the change in log-odds that a flight will be delayed with each unit increase in one of our predictor variables, such as wind speed.   
# MAGIC   
# MAGIC **Model Fitting**  
# MAGIC Logistic Regression is most commonly fit using maximum likelihood estimation. We express the log-likelihood of a given set of parameters \\(\beta\\) as:  
# MAGIC $$l(\beta) = \sum_{i=1}^{N}\\{y_i logp(x_i;\beta) + (1-y_i)log(1-p(x_i;\beta))\\}$$
# MAGIC $$l(\beta) = \sum_{i=1}^{N}\\{y_i\beta^{T}x_i - log(1+e^{\beta^{T}x_i})\\}$$
# MAGIC where \\(N\\) is the number of examples used to train the model.  
# MAGIC   
# MAGIC Because our best model included 576 features, regularization became necessary. Using k-fold cross-validation and grid parameter search on a sample of our training data, we decided to use both L1 and L2 regularization in our model. Thus our equation to maximize becomes:  
# MAGIC $$l(\beta) = \sum_{i=1}^{N}\\{y_i(\beta_0 + \beta^{T}x_i) - log(1+e^{\beta_0+\beta^{T}x_i})\\} -\lambda\sum_{j=1}^{p}|\beta_j| - \gamma \sum_{j=1}^{p}(\beta_j)^{2}$$
# MAGIC where \\(p\\) is the number of parameters in \\(\beta\\), \\(\lambda\\) is the L1 regularization parameter, and \\(\gamma\\) is the L2 regularization parameter. In the code below, the total amount of regularization (\\(R = \lambda + \gamma\\)) is expressed as `regParam` (\\(R\\)) and `elasticNetParam` (\\(\eta\\)) balances between the two. When \\(\eta\\) is 0, then only L2 regularization is used, and when \\(\eta\\) is 1, then only L1 regularization is used. Values between the two will split regularization types accordingly: \\(R = (1-\eta)\lambda + \eta\gamma\\)
# MAGIC   
# MAGIC The fitting process takes place iteratively. We specify the maximum number of iterations using the `maxIter` parameter in the code below.
# MAGIC   
# MAGIC **Other Considerations**
# MAGIC We train our model on the training set and evaluate on both the training and validation sets for tuning. Finally, after tuning we evaluate on the held-out test set. Note, the results below are on a very small sample of data and features for our toy example.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Our Model

# COMMAND ----------

# train a model on our transformed train data
startTime = time.time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 100, regParam = 0.001, elasticNetParam = 0.25, standardization = False)
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

# MAGIC %md 
# MAGIC ## Model Interpretation  
# MAGIC Now that we have trained and evaluated our model, we can dive into its coefficients and interpret it.

# COMMAND ----------

#examine coefficients
print('Coefficients: ')
print(f'Intercept, beta_0 = {model.interceptVector[0]}')
print(model.coefficientMatrix)

# COMMAND ----------

# MAGIC %md
# MAGIC Our model coefficients are as follows:  
# MAGIC \\(\beta\_{0} = -1.1600842\\): is our intercept, meaning that if all \\(x\_i\\) are 0, then the odds of \\(y = 1\\), a flight being delayed is: \\(e^{-1.1600842}\\). This equates to and odds ratio of 1 to 0.31.  
# MAGIC   
# MAGIC \\(\beta\_{1} = 0.5092805 \\): increasing \\(x\_{1}\\) by 1 increases the log-odds by 0.5092805. So if \\(x\_{1}\\) increases by 1, the odds that \\(y = 1\\) increase by a factor of \\(e^{0.5092805}\\). Putting this in the terms of the feature unit, increasing the wind speed by 1 mph increases the odds of a delay by 1.664.  
# MAGIC   
# MAGIC \\(\beta\_2 = -0.25834607\\): increasing \\(x\_{2}\\) by 1 decreases the log-odds by 0.25834607. So if \\(x\_{2}\\) increases by 1, the odds that \\(y = 1\\) increase by a factor of \\(e^{-0.25834607}\\). Putting this in the terms of the feature unit, increasing the ceiling height by 1 foot decreases the odds of a delay by 0.772.   
# MAGIC   
# MAGIC \\(\beta\_3 = -0.07973095\\): increasing \\(x\_{3}\\) by 1 decreases the log-odds by 0.07973095. So if \\(x\_{3}\\) increases by 1, the odds that \\(y = 1\\) increase by a factor of \\(e^{-0.07973095}\\). Putting this in the terms of the feature unit and probability, increasing the visibility by 1 foot increases the probability of a delay by 0.92.  
# MAGIC   
# MAGIC \\(\beta\_4 = 0.21659978\\): increasing \\(x\_{4}\\) by 1 increases the log-odds by 0.21659978. So if \\(x\_{4}\\) increases by 1, the odds that \\(y = 1\\) increase by a factor of \\(e^{0.21659978}\\). Putting this in the terms of the feature unit, increasing the temperature by 1 degree F increases the odds of a delay by 1.224.  
# MAGIC   
# MAGIC \\(\beta\_5 = 0.17401567\\): increasing \\(x\_{5}\\) by 1 increases the log-odds by 0.17401567. So if \\(x\_{5}\\) increases by 1, the odds that \\(y = 1\\) increase by a factor of \\(e^{0.17401567}\\). Putting this in the terms of the feature unit, increasing the barometric pressure by by 1 Pa increases the odds of a delay by 1.190.

# COMMAND ----------

# MAGIC %md # Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Evaluation
# MAGIC The chart below depicts the performance of our models on training and validation sets.  
# MAGIC ![Performance Chart](https://raw.githubusercontent.com/kevin-hartman/w261_final_project/master/images/prec_acc_f1.png?token=AKEHNPDZBRJHMPD2UZI2BP27GV3XS)  
# MAGIC A more detailed table of every model run is located here: https://docs.google.com/document/d/1g_uq1LQDFygjZRN8gdLffJVn2puRS-Gqg50IaHL5GgQ/edit
# MAGIC 
# MAGIC Our extensive modeling efforts yielded the following conclusions:
# MAGIC * All of our models had very low F1 scores, < 0.035, indicating that the positive class was rarely predicted.  
# MAGIC * The accuracy results of all models were comparable and roughly reflected the percentage of flights that were not delayed.
# MAGIC * Of the models that managed to predict the positive class, Logistic Regression trained on our 586 features generated the largest number of True Positives (10427, while still maintaining a top accuracy score (0.8168) and the best precision score 0.6227 for all models. 
# MAGIC * Random Forest models exhibited the greatest bias towards the negative class, frequently resulting in fewer than 1000 positive class predictions on the validation set.  
# MAGIC * Gradient-Boosted Trees trained on our 495 selected features (does not include Weather Rolling Average features) and PageRank also generated the greatest number of positive class predictions, with 19,113 true positives and 16,382 false positives.  
# MAGIC * Logistic Regression trains incredibly fast when only having a limited number of features.  
# MAGIC   
# MAGIC ## Scalability
# MAGIC Scalability wise, we encountered issues when training our Random Forest model.  We were only able to fully train our Random Forest model when using our initial Baseline selected features (35 Total).  When trying to train Random Forest with our 495 or 586 features, our cluster simply crashed. We attempted to use PCA to reduce the featureset down to the first 100 principal components (95% of observed variance), but this too proved to be unsuccessful.  In order to solve this problem, we attempted to scale up the compute power and memory of our cluster, but Random Forest still crashed. For all of the models, we ran cross-validation to identify the optimal hyperparameters, so we would not have to consistently run each model arbitrarily over and over again.  This was beneficial in optimizing our training time. Regarding model run time for 35 features (as Random Forest failed on our 495 and 586 features), our group noticed the following for validation: 
# MAGIC 
# MAGIC * Logistic Regression took around 2 minutes
# MAGIC * Random Forest took around 12 minutes
# MAGIC * Gradient Boosted Trees took around 4 minutes
# MAGIC 
# MAGIC We also attempted to implement XGBoost on GPUs to further speed up our training time and perhaps achieve better model performance, but we ran into problems with the worker nodes timing out due to heartbeat issues.  
# MAGIC   
# MAGIC Based on the dataset and problem, it is not totally important for training times to be incredibly fast. Because weather patterns are geographically linked and exhibit patterns across long periods of time, it is not necessary to retrain the model frequently. A practical retraining frequency would be quarterly, or if stakeholders wanted to be a little bit more conservative, monthly. Because of the infrequent necessity to retrain the model, training time becomes less important unless there are significant budget constraints for computer power that would benefit from faster training times. It is important to note that inference speed is critical for this problem. Users need to be informed of flight delays as soon as possible because issues with prior flights and changes in weather conditions can develop rapidly.
# MAGIC 
# MAGIC ## Division of Labor / Focus of Responsibility
# MAGIC * Sayan and Hersh focused on the modeling efforts and presentation preparation / graphics
# MAGIC * Nick and Kevin concentrated on Feature Engineering and Data Pipeline orchestration, with late contribution to Modeling
# MAGIC     
# MAGIC ## Lessons learned
# MAGIC  We have come to a new appreciation that 90% of Data Science is spent in data munging/cleaning, EDA, feature engineering, and pipeline orchestration. While we had a workable, fully-joined data set at the midway point, we continued with the same division of labor/responsibilities in the latter half of the project, believing that well-engineered features and a streamlined process would yield dividends to our downstream modeling efforts. Unfortunately, not enough focus was spent on modeling while our feature engineering and pipeline processing efforts continued to be developed. This was hard lesson learned from the team. Had we another week to improve our modeling efforts we believe we would have seen better results. Nonetheless, we come away having learned a lot, not just about Spark, but also on setting and managing expectations on the team objective for success.

# COMMAND ----------

# MAGIC %md # Application of Course Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC Our final project incorporated many of the concepts we learned in this course. Here are a few that we would like to highlight.  
# MAGIC   
# MAGIC #### Scalability
# MAGIC Scalability was crucial for modeling. For example, it is much easier to model using Logistic Regression compared to Random Forests. Random Forest models are much more computationally expensive than Logisitic Regressiom. This observation became apparent when adding more features to our models. As we added more features our Random Forest models frequently crashed due to memory and network shuffle constraints. In contrast, additional features increased the training times of Logistic Regression models, but memory and network shuffle issues were not encountered. We encountered issues with Random Forest models with many features even when scaling our cluster to the maximums allowed by our AWS accounts with respect to vCPUs and memory. Databricks, AWS, and Spark allowed us to track the performance and health of our clusters along the way. We were particularly impressed with how  well I/O scaled in our different Databricks clusters.
# MAGIC   
# MAGIC #### Graph Algorithms
# MAGIC Flights naturally form a graph from which we can extract information relevant to solving our problem. Further, thinking about flights as a graph led to a lot of discussion about feature engineering. We created a PageRank feature that effectively captured the traffic inbound to each airport. We hypothesized that this would help us because it would provide information about potential traffic congestion at airports that might cause delays. This feature was derived by creating anadjacency matrix that is a hash map where the key is the origin airport and the value is the destination airports and counts that reflect the number of times visited from the origin. 
# MAGIC 
# MAGIC #### Data Systems and Pipelines
# MAGIC One of our group's earliest objectives was to establish a working feature engineering and modeling pipeline. We leveraged two key pieces of technology:
# MAGIC 1. Delta Lakes: Delta Lake technology allowed us to have a repeatable, checkpointed pipeline for our feature engineering efforts. Each of our data transformations could be replayed or accessed ad hoc. The latter was critical in scenarios where bugs led to engineered features with unexpected values.
# MAGIC 2. MLFlow Pipeline: Setting up an MLFlow Pipeline allows for us to apply many of our feature transformations on different datasets (e.g., train, validation, test) with minimal repeated code. Some of the transformations we used include: one-hot encoding categorical data, normalized numerical data using a standard scaler, and vectorized our datasets so that our models could use them. A later addition included FeatureHashing for larger categorical variables.
# MAGIC 
# MAGIC #### Model Optimization
# MAGIC We took a two-pronged approach to optimizing our models: feature reduction and hyperparameter tuning. We used Principal Component Analysis and Regularization (L1 and L2) in order to automatically select the features that would have the most predictive power. We used k-fold cross-validation to select the optimal hyperparameters for each model. One key thing we learned is that sometimes the optimal hyperparameters found during cross-validation on a sample dataset were untenable when scaling our model. One example of this is the number of trees for a Random Forest. We also investigated Feature Hashing and addressing class imbalance by considering class weights in late modelling activity in the last few days. While we saw a noticible improvement to our F1 scores, our work is not yet complete as our precision went down. As an exercise for us, after this report is submitted, our plans are to improve our understanding of the models and continue to refine and tune them.

# COMMAND ----------

