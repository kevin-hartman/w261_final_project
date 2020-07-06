# Databricks notebook source
# MAGIC %md # Airline delays 
# MAGIC ## Bureau of Transportation Statistics
# MAGIC https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236   
# MAGIC https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC 
# MAGIC 2015 - 2019

# COMMAND ----------

# MAGIC %md ### Additional sources
# MAGIC This might be useful in matching station codes to airports:
# MAGIC 1. http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html
# MAGIC 2. https://www.world-airport-codes.com/

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
sqlContext = SQLContext(sc)

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data"))

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/*.parquet")
#display(airlines.sample(False, 0.00001))

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

f'{airlines.count():,}'

# COMMAND ----------

airlines_sample = airlines.where('(ORIGIN = "ORD" OR ORIGIN = "ATL") AND QUARTER = 1 and YEAR = 2015').sample(False, .10, seed = 42)

# COMMAND ----------

display(airlines_sample)

# COMMAND ----------

airlines_sample.count()

# COMMAND ----------

airlines_sample.dtypes

# COMMAND ----------

def get_dtype(df,colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]

# COMMAND ----------

get_dtype(airlines_sample, 'ORIGIN')

# COMMAND ----------

airlines_sample.columns

# COMMAND ----------

airlines_sample['ORIGIN']

# COMMAND ----------



# Custom-made class to assist with EDA on this dataset
# The code is generalizable. However, specific decisions on plot types were made because
# all our features are categorical
class Analyze:
    def __init__(self, df):
        self.df = df.toPandas()
    
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
            #if col == 'MachineIdentifier': continue
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

analyzer = Analyze(airlines_sample)
analyzer.print_eda_summary()

# COMMAND ----------



# COMMAND ----------

sns.set(rc={'figure.figsize':(100,100)})
sns.heatmap(airlines_sample.toPandas().corr(), cmap='RdBu_r', annot=True, center=0.0)

# COMMAND ----------

sns.set(rc={'figure.figsize':(10,10)})

# COMMAND ----------

airlines_sample.where('DEP_DELAY < 0').count() / airlines_sample.count() # This statistic explains that 47% of flights depart earlier

# COMMAND ----------

airlines_sample.where('DEP_DELAY == 0').count() / airlines_sample.count()  # This statistic explains that 6.9% of flights depart EXACTLY on time

# COMMAND ----------

# MAGIC %md ### The cells below are displaying histograms that analyze departures that are on time or early

# COMMAND ----------

bins, counts = airlines_sample.select('DEP_DELAY').where('DEP_DELAY <= 0').rdd.flatMap(lambda x: x).histogram(100)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = airlines_sample.select('DEP_DELAY').where('DEP_DELAY <= 0 AND DEP_DELAY > -25').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md ### The cells below are displaying histograms that analyze departures that are delayed

# COMMAND ----------

bins, counts = airlines_sample.select('DEP_DELAY').where('DEP_DELAY > 0').rdd.flatMap(lambda x: x).histogram(100)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = airlines_sample.select('DEP_DELAY').where('DEP_DELAY > 0 AND DEP_DELAY < 300').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = airlines_sample.select('DEP_DELAY').where('DEP_DELAY > -25 AND DEP_DELAY < 50').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md **Analyzing the plot above, it is apparent that the distribution is right-skewed, implying that there is a heavy amount of data that is delayed and shifting the distribution towards the right, so therefore the median departure delay is higher than the mean.  Intuitively, this makes sense, for it is more likely that a flight will depart a day later compared to a flight departing a day earlier.  Moreover, we can see that much of the data revolves around flights that depart early or on time, and it is possible that the data is from airports that are smaller with less load; this would explain how the flights would be more likely to depart at an earlier time.  Further analysis of the locations of the actual airports and the distribution of these airports is necessary.**

# COMMAND ----------

# MAGIC %md # TODO
# MAGIC > Visualize the locations of the airport and the amount of traffic that is coming using a US map  
# MAGIC https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time
# MAGIC > Scatterplot comparing the distance between the airports and the in flight air delay. 

# COMMAND ----------

# MAGIC %md ### Next, we will look into visualizing arrival delay.  However, we should note that arrival delay also encompasses any delay from the departure delay.  Therefore, we must first ensure that we create a new column that accounts for this discrepancy.

# COMMAND ----------

from pyspark.sql.functions import lit
# airlines['ARR_DELAY_CONTRIB'] = airlines['ARR_DELAY'] - airlines['DEP_DELAY'] # contribution of the arrival delay ONLY
# df.withColumn("ARR_DELAY_CONTRIB",col("salary")* -1)
# airlines.select('ARR_DELAY','DEP_DELAY', (airlines.ARR_DELAY - airlines.DEP_DELAY).alias('ARR_')).show()
airlines_sample = airlines_sample.withColumn('IN_FLIGHT_AIR_DELAY', lit(airlines_sample['ARR_DELAY'] - airlines_sample['DEP_DELAY'])) # this column is the time difference between arrival and departure and does not include total flight delay
airlines_sample.select('IN_FLIGHT_AIR_DELAY').show()

# COMMAND ----------

bins, counts = airlines_sample.select('IN_FLIGHT_AIR_DELAY').where('IN_FLIGHT_AIR_DELAY > -50 AND IN_FLIGHT_AIR_DELAY < 50').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md **We can see that there is a normal distribution that is centered around -5; this indicates that the flight makes up 5 minutes of time after departing from the airport.  In general, this is implying that flights are making up time in the air time.  Further analysis should look into analyzing the amount of time made up in the air based on distance to see if flights make up more delay time with longer flight distances.**

# COMMAND ----------

# MAGIC %md # Weather
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data"))

# COMMAND ----------

weather = spark.read.option("header", "true")\
                    .parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/*.parquet")

f'{weather.count():,}'

# COMMAND ----------

weather.printSchema()

# COMMAND ----------

display(weather)

# COMMAND ----------

# subset to 1Q2015
weather_subset = weather.where('DATE >= TO_DATE("01/01/2015", "MM/dd/yyyy") AND DATE <= TO_DATE("03/31/2015", "MM/dd/yyyy")')
weather_subset.count()

# COMMAND ----------

# exploring station id lengths for join
weather_w_length = weather.withColumn("STATION_LENGTH", f.length("STATION"))
display(weather_w_length.agg(f.min("STATION_LENGTH")))

# COMMAND ----------

#Padding IDs
weather_subset = weather_subset.withColumn("STATION_PAD", f.lpad(weather_subset.STATION, 11, '0'))
#display(weather_subset.select(f.lpad(weather_subset.STATION, 11, '0').alias('STATION_PAD')))
display(weather_subset)

# COMMAND ----------

# MAGIC %md # Stations

# COMMAND ----------

stations = spark.read.option("header", "true").csv("dbfs:/mnt/mids-w261/data/DEMO8/gsod/stations.csv.gz")
stations = stations.withColumnRenamed("name","station_name")

# COMMAND ----------

display(stations)

# COMMAND ----------

stations.printSchema()

# COMMAND ----------

display(stations.where(f.col('station_name').contains('BIG BEAR')))

# COMMAND ----------

#concatenate usaf and wban
stations = stations.withColumn("USAF_WBAN", f.concat(f.col("usaf"), f.col("wban")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample functions for selection station info

# COMMAND ----------

stations.select('name').distinct().count()

# COMMAND ----------

display(stations.select('name').distinct())

# COMMAND ----------

weather.select('NAME').distinct().count()

# COMMAND ----------

display(weather.select('name').distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Joining Weather to Station Data
# MAGIC Note this is for exploratory purposes. The most efficient way to do this will be to first identify the station associated with each airport, add a column for that to the flights data, and then join directly flights to weather. Note this will require a composite key because we care about both **time** and **location**. Note the cell after the initial join where the joined table is displayed with a filter will take a long time to load.

# COMMAND ----------

joined_weather = weather_subset.join(f.broadcast(stations), weather_subset.STATION_PAD == stations.USAF_WBAN, 'left')

# COMMAND ----------

joined_weather_cache = joined_weather.cache()

# COMMAND ----------

display(joined_weather_cache)

# COMMAND ----------

display(joined_weather_cache.where(f.col('NAME').contains('BIG BEAR')).select(['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION','SOURCE','NAME']).distinct())

# COMMAND ----------

no_id_match = joined_weather_cache.where('USAF_WBAN IS NULL').cache()

# COMMAND ----------

display(joined_weather_cache.select(["STATION_PAD", "NAME", "station_name", "usaf", "wban"]).where("usaf = 999999").distinct())

# COMMAND ----------

display(no_id_match.select(["STATION_PAD", "STATION", "NAME", "LATITUDE", "LONGITUDE", "SOURCE"]).distinct())
#no_id_match.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Airports
# MAGIC Data from Global Airport Database: https://www.partow.net/miscellaneous/airportdatabase/index.html#Downloads  
# MAGIC **Make sure to use decimal degree fields for NN**

# COMMAND ----------

# you will need to upllad this to s3 on your own.
display(dbutils.fs.ls("dbfs:/global_airports"))

# COMMAND ----------

airports = spark.read.option("header", "false").csv("dbfs:/global_airports/GlobalAirportDatabase.txt", sep = ":")

# COMMAND ----------

display(airports)

# COMMAND ----------

# rename the columns
airports = airports.select(f.col("_c0").alias("ICAO Code"),
                           f.col("_c1").alias("IATA Code"),
                           f.col("_c2").alias("Airport Name"),
                           f.col("_c3").alias("City/Town"),
                           f.col("_c4").alias("Country"),
                           f.col("_c5").alias("Latitude Degrees"),
                           f.col("_c6").alias("Latitude Minutes"),
                           f.col("_c7").alias("Latitude Seconds"),
                           f.col("_c8").alias("Latitude Direction"),
                           f.col("_c9").alias("Longitude Degrees"),
                           f.col("_c10").alias("Longitude Minutes"),
                           f.col("_c11").alias("Longitude Seconds"),
                           f.col("_c12").alias("Longitude Direction"),
                           f.col("_c13").alias("Altitude"),
                           f.col("_c14").alias("Latitude Decimal Degrees"),
                           f.col("_c15").alias("Longitude Decimal Degrees"),
                          )

# COMMAND ----------

display(airports)

# COMMAND ----------

airports = airports.where(f.col('Country').contains('USA'))

# COMMAND ----------

display(airports.where(f.col("Latitude Decimal Degrees") == 0))

# COMMAND ----------

# are there airports with missing lat/long that have corresponding complete entries
# ABQ fine, ABY (Albany), BQK (Brunswick) totally bad, HAR (Harrisburg) fine with code MDT, SFB (Sanford) totally bad
display(airports.where(f.col('City/Town').contains('SANFORD')))

# COMMAND ----------

# MAGIC %md
# MAGIC # New Airports Dataset

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/openflights"))

# COMMAND ----------

airports_new = spark.read.option("header", "false").csv("dbfs:/openflights/airports.csv", sep = ",")

# COMMAND ----------

display(airports_new)

# COMMAND ----------

# rename the columns
airports_new = airports_new.select(f.col("_c0").alias("Airport ID"),
                           f.col("_c1").alias("Name"),
                           f.col("_c2").alias("City"),
                           f.col("_c3").alias("Country"),
                           f.col("_c4").alias("IATA"),
                           f.col("_c5").alias("ICAO"),
                           f.col("_c6").alias("Latitude"),
                           f.col("_c7").alias("Longitude"),
                           f.col("_c8").alias("Altitude"),
                           f.col("_c9").alias("Timezone"),
                           f.col("_c10").alias("DST"),
                           f.col("_c11").alias("LTz database time zone"),
                           f.col("_c12").alias("Type"),
                           f.col("_c13").alias("Source")
                          )

# COMMAND ----------

display(airports_new)

# COMMAND ----------

airports_new = airports_new.where('Country = "United States"')

# COMMAND ----------

airports_new.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Finding nearest station to each airport

# COMMAND ----------

from math import radians, cos, sin, asin, sqrt

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

#34.14694	-97.1225
haversine(34.14694, -97.1225, 34.24694, -97.3225)

# COMMAND ----------

# for every airport
  # create data structure to hold weather station/distance pairs
  # for every weather station
    # calculate haversine distance
    # append to data structure
    
  # key for min(distances between airport and weather station)

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
            station_id = station_list[-1]
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
airports_rdd = airports_new.rdd
stations_rdd = stations.rdd

# COMMAND ----------

airports_rdd.take(1)

# COMMAND ----------

closest_stations = find_closest_station(airports_rdd,stations_rdd).cache()

# COMMAND ----------

closest_stations.collect()

# COMMAND ----------

airports_stations = sqlContext.createDataFrame(closest_stations)
airports_stations = airports_stations.withColumn("nearest_station_id",f.col("_2")["_1"]).withColumn("nearest_station_dist",f.col("_2")["_2"])
airports_stations =airports_stations.drop("_2")
airports_stations = airports_stations.withColumnRenamed("_1", "IATA")
display(airports_stations)

# COMMAND ----------

display(airports_new.where(f.col('IATA').contains('\\N')))


# COMMAND ----------

display(stations.where(f.col('USAF_WBAN').contains('72054300167')))

# COMMAND ----------

display(stations.where(f.col()))

# COMMAND ----------

# MAGIC %md
# MAGIC # Split the train/validation/test sets and normalize the data

# COMMAND ----------

# data distribution across RDD partitions is not idempotent, and could be rearranged or updated during the query execution, thus affecting the output of the randomSplit method
# to resolve the issue, we can repartition, or apply an aggregate function, or we can cache (https://kb.databricks.com/data/random-split-behavior.html)
# also add a unique ID (monotonically_increasing_id)
flightsCache = airlines.withColumn("id", f.monotonically_increasing_id()).cache()

# COMMAND ----------

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


# COMMAND ----------

df_train, df_test, df_validate = generate_train_test_validate_sets(train_ratio=.8, test_ratio=.1, df=flightsCache, label='DEP_DEL15', join_on="id", seed=42)

# COMMAND ----------

print(df_train.count())

# COMMAND ----------

print(df_test.count())

# COMMAND ----------

print(df_validate.count())

# COMMAND ----------

print(flightsCache.count())

# COMMAND ----------

