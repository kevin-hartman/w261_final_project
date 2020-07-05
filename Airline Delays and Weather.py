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
sqlContext = SQLContext(sc)


import pandas as pd
import seaborn as sns

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
sns.heatmap(df_train.corr(), cmap='RdBu_r', annot=True, center=0.0)

# COMMAND ----------



# COMMAND ----------

airlines.where('DEP_DELAY < 0').count() / airlines.count() # This statistic explains that 58.7% of flights depart earlier

# COMMAND ----------

airlines.where('DEP_DELAY == 0').count() / airlines.count()  # This statistic explains that 5.2% of flights depart EXACTLY on time

# COMMAND ----------

type(airlines)

# COMMAND ----------

# MAGIC %md ### The cells below are displaying histograms that analyze departures that are on time or early

# COMMAND ----------

bins, counts = airlines.select('DEP_DELAY').where('DEP_DELAY <= 0').rdd.flatMap(lambda x: x).histogram(100)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = airlines.select('DEP_DELAY').where('DEP_DELAY <= 0 AND DEP_DELAY > -25').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

# MAGIC %md ### The cells below are displaying histograms that analyze departures that are delayed

# COMMAND ----------

bins, counts = airlines.select('DEP_DELAY').where('DEP_DELAY > 0').rdd.flatMap(lambda x: x).histogram(100)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = airlines.select('DEP_DELAY').where('DEP_DELAY > 0 AND DEP_DELAY < 300').rdd.flatMap(lambda x: x).histogram(50)
plt.hist(bins[:-1], bins=bins, weights=counts)

# COMMAND ----------

bins, counts = airlines.select('DEP_DELAY').where('DEP_DELAY > -25 AND DEP_DELAY < 50').rdd.flatMap(lambda x: x).histogram(50)
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
airlines = airlines.withColumn('IN_FLIGHT_AIR_DELAY', lit(airlines['ARR_DELAY'] - airlines['DEP_DELAY'])) # this column is the time difference between arrival and departure and does not include total flight delay
airlines.select('IN_FLIGHT_AIR_DELAY').show()

# COMMAND ----------

bins, counts = airlines.select('IN_FLIGHT_AIR_DELAY').where('IN_FLIGHT_AIR_DELAY > -50 AND IN_FLIGHT_AIR_DELAY < 50').rdd.flatMap(lambda x: x).histogram(50)
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

display(airports.where(f.col('Country').contains('USA')))

# COMMAND ----------

