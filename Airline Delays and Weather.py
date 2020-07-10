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

# MAGIC %md # Flights EDA
# MAGIC 
# MAGIC Schema for flights: https://annettegreiner.com/vizcomm/OnTime_readme.html

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

# MAGIC %md # Weather EDA
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532
# MAGIC 
# MAGIC 
# MAGIC Schema for weather: https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf

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
# weather_subset = weather.where('DATE >= TO_DATE("01/01/2015", "MM/dd/yyyy") AND DATE <= TO_DATE("03/31/2015", "MM/dd/yyyy")')  # this takes only the first quarter of weather data
weather_subset = weather 
weather_subset.count() 

# COMMAND ----------

# exploring station id lengths for join
weather_w_length = weather.withColumn("STATION_LENGTH", f.length("STATION"))
display(weather_w_length.agg(f.min("STATION_LENGTH")))

# COMMAND ----------

#Padding IDs
weather_subset = weather_subset.withColumn("STATION_PAD", f.lpad(weather_subset.STATION, 11, '0'))
#display(weather_subset.select(f.lpad(weather_subset.STATION, 11, '0').alias('STATION_PAD')))
# display(weather_subset)

# COMMAND ----------

# WND Fields [direction angle, quality code, type code, speed rate, speed quality code]
split_col = f.split(weather_subset['WND'], ',')
weather_subset = weather_subset.withColumn('WND_Direction_Angle', split_col.getItem(0).cast('int')) # numerical
weather_subset = weather_subset.withColumn('WND_Quality_Code', split_col.getItem(1).cast('int')) # categorical
weather_subset = weather_subset.withColumn('WND_Type_Code', split_col.getItem(2).cast('string')) # categorical
weather_subset = weather_subset.withColumn('WND_Speed_Rate', split_col.getItem(3).cast('int')) # categorical
weather_subset = weather_subset.withColumn('WND_Speed_Quality_Code', split_col.getItem(4).cast('int')) # numerical

# CIG Fields
split_col = f.split(weather_subset['CIG'], ',')
weather_subset = weather_subset.withColumn('CIG_Ceiling_Height_Dimension', split_col.getItem(0).cast('int')) # numerical 
weather_subset = weather_subset.withColumn('CIG_Ceiling_Quality_Code', split_col.getItem(1).cast('int')) # categorical
weather_subset = weather_subset.withColumn('CIG_Ceiling_Determination_Code', split_col.getItem(2).cast('string')) # categorical 
weather_subset = weather_subset.withColumn('CIG_CAVOK_code', split_col.getItem(3).cast('string')) # categorical/binary

# VIS Fields
split_col = f.split(weather_subset['VIS'], ',')
weather_subset = weather_subset.withColumn('VIS_Distance_Dimension', split_col.getItem(0).cast('int')) # numerical
weather_subset = weather_subset.withColumn('VIS_Distance_Quality_Code', split_col.getItem(1).cast('int')) # categorical
weather_subset = weather_subset.withColumn('VIS_Variability_Code', split_col.getItem(2).cast('string')) # categorical/binary
weather_subset = weather_subset.withColumn('VIS_Quality_Variability_Code', split_col.getItem(3).cast('int')) # categorical

# TMP Fields
split_col = f.split(weather_subset['TMP'], ',')
weather_subset = weather_subset.withColumn('TMP_Air_Temp', split_col.getItem(0).cast('int')) # numerical
weather_subset = weather_subset.withColumn('TMP_Air_Temp_Quality_Code', split_col.getItem(1).cast('string')) # categorical

# DEW Fields
split_col = f.split(weather_subset['DEW'], ',')
weather_subset = weather_subset.withColumn('DEW_Point_Temp', split_col.getItem(0).cast('int')) # numerical
weather_subset = weather_subset.withColumn('DEW_Point_Quality_Code', split_col.getItem(1).cast('string')) # categorical

# SLP Fields
split_col = f.split(weather_subset['SLP'], ',')
weather_subset = weather_subset.withColumn('SLP_Sea_Level_Pres', split_col.getItem(0).cast('int')) # numerical
weather_subset = weather_subset.withColumn('SLP_Sea_Level_Pres_Quality_Code', split_col.getItem(1).cast('int')) # categorical

# Now that the data is split apart, we can transform each column

# COMMAND ----------

weather_distinct_ids = weather_subset.select(f.lpad(weather_subset.STATION, 11, '0').alias('STATION_PAD')).distinct()

# COMMAND ----------

display(weather_distinct_ids)

# COMMAND ----------



# COMMAND ----------

display(weather_subset)

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

# MAGIC %md ### Now let's take a look at how much of this data is missing

# COMMAND ----------

# TMP_Air_Tempbins, counts = weather_subset.where('WND_Categorical_Variables' <= 113').select('WND_Speed_Rate').rdd.flatMap(lambda x: x).histogram(20)
# plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------



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

# MAGIC %md **Now we will do more joins where we take the distinct weather id's and join it with the station dataset**

# COMMAND ----------

joined_weather_distinct_ids = weather_distinct_ids.join(f.broadcast(stations), weather_distinct_ids.STATION_PAD == stations.USAF_WBAN, 'left')
joined_weather_distinct_ids = joined_weather_distinct_ids.where("STATION_PAD IS NOT NULL")

# COMMAND ----------

display(joined_weather_distinct_ids)

# COMMAND ----------

joined_weather_joined_weather_distinct_ids.where("STATION_PAD IS NOT NULL")

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
# MAGIC Data from OpenFlights: https://openflights.org/data.html 

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/openflights"))

# COMMAND ----------

airports = spark.read.option("header", "false").csv("dbfs:/openflights/airports.csv", sep = ",")

# COMMAND ----------

display(airports)

# COMMAND ----------

# rename the columns
airports = airports.select(f.col("_c0").alias("Airport ID"),
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

display(airports)

# COMMAND ----------

airports = airports.where('Country = "United States" OR Country = "Puerto Rico" OR Country = "Virgin Islands"')

# COMMAND ----------

airports.count()

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

display(joined_weather_distinct_ids)

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
            if not station_list[8] or not station_list[7]:
                continue
            station_lon = float(station_list[8])
            station_lat = float(station_list[7])
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
airports_rdd = airports.rdd
stations_rdd = joined_weather_distinct_ids.rdd

# COMMAND ----------

airports.take(1)

# COMMAND ----------

closest_stations = find_closest_station(airports_rdd,stations_rdd).cache()

# COMMAND ----------

closest_stations.collect()

# COMMAND ----------

airports_stations = sqlContext.createDataFrame(closest_stations)
airports_stations = airports_stations.withColumn("nearest_station_id",f.col("_2")["_1"]).withColumn("nearest_station_dist",f.col("_2")["_2"])
airports_stations =airports_stations.drop("_2")
airports_stations_origin = airports_stations.withColumnRenamed("_1", "IATA")
airports_stations_dest = airports_stations_origin

airports_stations_origin = airports_stations_origin.withColumnRenamed("IATA", "IATA_ORIGIN")
airports_stations_origin = airports_stations_origin.withColumnRenamed("nearest_station_id", "nearest_station_id_ORIGIN")
airports_stations_origin = airports_stations_origin.withColumnRenamed("nearest_station_dist", "nearest_station_dist_ORIGIN")

airports_stations_dest = airports_stations_dest.withColumnRenamed("IATA", "IATA_DEST")
airports_stations_dest = airports_stations_dest.withColumnRenamed("nearest_station_id", "nearest_station_id_DEST")
airports_stations_dest = airports_stations_dest.withColumnRenamed("nearest_station_dist", "nearest_station_id_DEST")
display(airports_stations_origin)

# COMMAND ----------

display(airports_stations_dest)

# COMMAND ----------

display(airports.where(f.col('IATA').contains('\\N')))


# COMMAND ----------

display(stations.where(f.col('USAF_WBAN').contains('72054300167')))

# COMMAND ----------

# MAGIC %md ## Now we will add in the nearest weather stations to each flight

# COMMAND ----------

display(airlines_sample)

# COMMAND ----------

joined_flights_stations = airlines_sample.join(f.broadcast(airports_stations_origin), airlines_sample.ORIGIN == airports_stations_origin.IATA_ORIGIN, 'left')
joined_flights_stations = joined_flights_stations.join(f.broadcast(airports_stations_dest), joined_flights_stations.DEST == airports_stations_dest.IATA_DEST, 'left')

# COMMAND ----------

display(joined_flights_stations)

# COMMAND ----------

# MAGIC %md **Prior to creating the composite keys, we need to make adjustments to our time variables**

# COMMAND ----------

from pytz import timezone
display(airports)


# COMMAND ----------

# MAGIC %md ## Now we will create composite keys based on the station id, flight and weather measurement time from the joined weather station dataset

# COMMAND ----------

# In Weather: StationID_MeasurementMonth_MeasurementDay_MeasurementHour
from pyspark.sql.functions import udf
def get_flight_hour(flight_time):
    flight_time = str(flight_time)
    hour = ''
    if len(flight_time) == 3:
      hour = flight_time[0]
    elif len(flight_time) == 4:
      hour = flight_time[:2]
    return hour
  
# spark.udf.register("get_flight_hour", get_flight_hour)
get_flight_hour = udf(get_flight_hour)
get_two_hour_adjusted_flight_hour = udf(get_two_hour_adjusted_flight_hour)
# joined_flight_stations = joined_flights_stations.withColumn("FLIGHT_HOUR", f.lit(lambda x: get_flight_hour(x["CRS_DEP_TIME"])))
# display(joined_flight_stations.select("CRS_DEP_TIME", get_flight_hour("CRS_DEP_TIME")))

joined_flight_stations = joined_flights_stations.withColumn("ORIGIN_Weather_Key",\
                                                                      f.concat(joined_flights_stations["nearest_station_id_ORIGIN"],\
                                                                               joined_flights_stations["YEAR"],\
                                                                               joined_flights_stations["MONTH"],\
                                                                               joined_flights_stations["DAY_OF_MONTH"],\
                                                                               get_two_hour_adjusted_flight_hour(joined_flights_stations["CRS_DEP_TIME"])))

joined_flight_stations = joined_flights_stations.withColumn("DEST_Weather_Key",\
                                                                      f.concat(joined_flights_stations["nearest_station_id_DEST"],\
                                                                               joined_flights_stations["YEAR"],\
                                                                               joined_flights_stations["MONTH"],\
                                                                               joined_flights_stations["DAY_OF_MONTH"],\
                                                                               get_two_hour_adjusted_flight_hour(joined_flights_stations["CRS_DEP_TIME"])))

# COMMAND ----------

display(joined_flight_stations)

# COMMAND ----------

# In Weather: StationID_MeasurementMonth_MeasurementDay_MeasurementHour
def get_two_hour_adjusted_flight_hour(flight_time):
    flight_time = str(flight_time)
    hour = None
    if len(flight_time) == 3:
      hour = int(flight_time[0])
    elif len(flight_time) == 4:
      hour = int(flight_time[:2])
    else:
      return hour
    hour = hour - 2
    if hour == -1:
      hour = 23
    elif hour == -2:
      hour = 22
    return str(hour)

# COMMAND ----------

# MAGIC %md
# MAGIC # Split the train/validation/test sets and normalize the data

# COMMAND ----------

# data distribution across RDD partitions is not idempotent, and could be rearranged or updated during the query execution, thus affecting the output of the randomSplit method
# to resolve the issue, we can repartition, or apply an aggregate function, or we can cache (https://kb.databricks.com/data/random-split-behavior.html)
# also add a unique ID (monotonically_increasing_id)
flightsCache = airlines_sample.withColumn("id", f.monotonically_increasing_id()).cache()

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

# review categorical and numerical features:
cat_cols = [item[0] for item in df_train.dtypes if item[1].startswith('string')] 
print(str(len(cat_cols)) + '  categorical features')
num_cols = [item[0] for item in df_train.dtypes if item[1].startswith('int') | item[1].startswith('double')][1:]
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

missing_recs = info_missing_table(flightsCache.toPandas())
missing_recs

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import lit
from pyspark.sql.functions import col,sum

def count_missings(df):
    miss_counts = list()
    for c in df.columns:
        if df.where(col(c).isNull()).count() > 0:
           tup = (c,int(df.where(col(c).isNull()).count()))
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
from pyspark.sql.functions import rank,sum,col

cleaned_airlines_sample = flightsCache

df_Nomiss=cleaned_airlines_sample.na.drop()
for x in catcolums_miss:                  
  mode=df_Nomiss.groupBy(x).count().sort(col("count").desc()).collect()
  if mode:
    print(x, mode[0][0]) #print name of columns and it's most categories 
    cleaned_airlines_sample = cleaned_airlines_sample.na.fill({x:mode[0][0]})
    
# fill the missing numerical values with the average of each #column
from pyspark.sql.functions import mean, round
for i in numcolumns_miss:
  meanvalue = cleaned_airlines_sample.select(round(mean(i))).collect()
  if meanvalue:
    print(i, meanvalue[0][0]) 
    cleaned_airlines_sample=cleaned_airlines_sample.na.fill({i:meanvalue[0][0]})

# COMMAND ----------



# COMMAND ----------

# Use the OneHotEncoderEstimator to convert categorical features into one-hot vectors
# Use VectorAssembler to combine vector of one-hots and the numerical features
# Append the process into the stages array to reproduce
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

stages = []
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

