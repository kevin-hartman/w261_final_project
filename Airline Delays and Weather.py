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


# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data"))

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/*.parquet")
display(airlines.sample(False, 0.00001))

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

f'{airlines.count():,}'

# COMMAND ----------

display(airlines.describe())

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

# MAGIC %md # Weather
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data"))

# COMMAND ----------

weather = spark.read.option("header", "true")\
                    .parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/weather_data/*.parquet")

f'{weather.count():,}'

# COMMAND ----------

display(weather.where('DATE =="DATE"'))

# COMMAND ----------

display(weather.describe())

# COMMAND ----------

# MAGIC %md # Stations

# COMMAND ----------

stations = spark.read.option("header", "true").csv("dbfs:/mnt/mids-w261/data/DEMO8/gsod/stations.csv.gz")

# COMMAND ----------

display(stations)

# COMMAND ----------

from pyspark.sql import functions as f
stations.where(f.col('name').contains('JAN MAYEN NOR NAVY'))

# COMMAND ----------

stations.select('name').distinct().count()

# COMMAND ----------

display(stations.select('name').distinct())

# COMMAND ----------

weather.select('NAME').distinct().count()

# COMMAND ----------

display(weather.select('name').distinct())

# COMMAND ----------

# MAGIC %md # Re-downloading Weather Data

# COMMAND ----------

# MAGIC %md
# MAGIC > NOTE: You don't ahve to download the data, is already available in the `dbfs:/mnt/mids-w261/data/datasets_final_project/new_weather_parquet/weather.parquet`
# MAGIC > It's just an exercise to show you how can you parallelize the task of downloading data when the size is considerable

# COMMAND ----------

from bs4 import BeautifulSoup
import requests
import copy

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, ShortType, StringType, DoubleType, IntegerType, NullType, LongType, TimestampType


# COMMAND ----------

# Website is organized as baseURL/year/<some_file_number>.csv
baseURL = 'https://www.ncei.noaa.gov/data/global-hourly/access/'

# COMMAND ----------

def scrap_html(year):
  results = requests.get(f'{bURL.value}{year}/')
  soup = BeautifulSoup(results.text, 'lxml')
  
  for tag in soup.html.body.table.findChildren():
      try:
          if tag.name == 'tr' and '.csv' in tag.td.a.text:
              
              # Yield year and file name
              yield (year,tag.td.a.text)
      except:
          pass

# We'll scrap 5 html pages in parallel to get all the file names
bURL = sc.broadcast(baseURL)
filesRDD = sc.parallelize(range(2015,2020))\
              .flatMap(scrap_html)\
              .repartition(586)\
              .cache()

filesRDD.getNumPartitions(), filesRDD.take(10)

# COMMAND ----------

# We get all the headers from all files to capture the most information as possible

def parse_headers(file):
  
  year, filename = file
  
  try:
    # This request-get gets a single csv file content as a string
    blob = requests.get(f'https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{filename}')
  
  except:
    pass
  
  header = blob.text.split('\n')[0].replace('"','').split(',')
  
  for col in header:
    
    yield col
  

headers = filesRDD.flatMap(parse_headers)\
                  .distinct()\
                  .collect()

len(headers)

# COMMAND ----------

dict_ = {'STATION': '', 'DATE': '', 'SOURCE': '', 'LATITUDE': '', 'LONGITUDE': '',
            'ELEVATION': '', 'NAME': '', 'REPORT_TYPE': '', 'CALL_SIGN': '', 'QUALITY_CONTROL': '',
            'WND': '', 'CIG': '', 'VIS': '', 'TMP': '', 'DEW': '', 'SLP': '', 'AW1': '', 
            'GA1': '', 'GA2': '', 'GA3': '', 'GA4': '', 'GE1': '', 'GF1': '', 'KA1': '',
            'KA2': '', 'MA1': '', 'MD1': '', 'MW1': '', 'MW2': '', 'OC1': '', 'OD1': '',
            'OD2': '', 'REM': '', 'EQD': '', 'AW2': '', 'AX4': '', 'GD1': '', 'AW5': '',
            'GN1': '', 'AJ1': '', 'AW3': '', 'MK1': '', 'KA4': '', 'GG3': '', 'AN1': '',
            'RH1': '', 'AU5': '', 'HL1': '', 'OB1': '', 'AT8': '', 'AW7': '', 'AZ1': '',
            'CH1': '', 'RH3': '', 'GK1': '', 'IB1': '', 'AX1': '', 'CT1': '', 'AK1': '',
            'CN2': '', 'OE1': '', 'MW5': '', 'AO1': '', 'KA3': '', 'AA3': '', 'CR1': '',
            'CF2': '', 'KB2': '', 'GM1': '', 'AT5': '', 'AY2': '', 'MW6': '', 'MG1': '',
            'AH6': '', 'AU2': '', 'GD2': '', 'AW4': '', 'MF1': '', 'AA1': '', 'AH2': '',
            'AH3': '', 'OE3': '', 'AT6': '', 'AL2': '', 'AL3': '', 'AX5': '', 'IB2': '',
            'AI3': '', 'CV3': '', 'WA1': '', 'GH1': '', 'KF1': '', 'CU2': '', 'CT3': '',
            'SA1': '', 'AU1': '', 'KD2': '', 'AI5': '', 'GO1': '', 'GD3': '', 'CG3': '',
            'AI1': '', 'AL1': '', 'AW6': '', 'MW4': '', 'AX6': '', 'CV1': '', 'ME1': '',
            'KC2': '', 'CN1': '', 'UA1': '', 'GD5': '', 'UG2': '', 'AT3': '', 'AT4': '',
            'GJ1': '', 'MV1': '', 'GA5': '', 'CT2': '', 'CG2': '', 'ED1': '', 'AE1': '',
            'CO1': '', 'KE1': '', 'KB1': '', 'AI4': '', 'MW3': '', 'KG2': '', 'AA2': '',
            'AX2': '', 'AY1': '', 'RH2': '', 'OE2': '', 'CU3': '', 'MH1': '', 'AM1': '',
            'AU4': '', 'GA6': '', 'KG1': '', 'AU3': '', 'AT7': '', 'KD1': '', 'GL1': '',
            'IA1': '', 'GG2': '', 'OD3': '', 'UG1': '', 'CB1': '', 'AI6': '', 'CI1': '',
            'CV2': '', 'AZ2': '', 'AD1': '', 'AH1': '', 'WD1': '', 'AA4': '', 'KC1': '',
            'IA2': '', 'CF3': '', 'AI2': '', 'AT1': '', 'GD4': '', 'AX3': '', 'AH4': '',
            'KB3': '', 'CU1': '', 'CN4': '', 'AT2': '', 'CG1': '', 'CF1': '', 'GG1': '',
            'MV2': '', 'CW1': '', 'GG4': '', 'AB1': '', 'AH5': '', 'CN3': '' }

# COMMAND ----------

def parse(file):
  
  try:
    
    # This request-get gets a single csv file content as a string
    blob = requests.get(f'https://www.ncei.noaa.gov/data/global-hourly/access/{file[0]}/{file[1]}')
  
  except:
    
    # We will keep a report in case website unavailable, or file not found
    yield ('#ZZZ#', file[0], file[1], 'URL Not Available', None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None
           )
           
  else:
    
    header = blob.text.split('\n')[0].replace('"', '').split(',')
    
    # For every successful csv read, we parse as usual
    for line in blob.text.split('\n')[1:]:

      row = ''
      count_qt = 0

      for i in line:
        if i == '"':
          count_qt += 1
        elif i == ',' and count_qt % 2 == 0:
          row += ':|:'
        else:
          row += i
      
      row = row.split(':|:')
      
      if len(header) == len(row):
        
        # We'll usa a dictionary to make sure we place the data in the right order
        payload = copy.deepcopy(dict_b.value)

        for colname, item in zip(header, row):

          payload[colname] = str(item)

        # Yield only the lines that had 177 elements after splitting
        if len(payload.values()) == 177:
          yield tuple(list(payload.values()))

      else:
        
        # We will keep a report in case website unavailable, or file not found
        yield ('#ZZZ#', file[0], file[1], 'corrupted row', row, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None
               )

# What we parallelize in this case are the file names, since the baseURL is the same, 
# we just pass the year and file name to out helper func

dict_b = sc.broadcast(dict_)

badData = []

try:
  dataRDD.unpersist()
except:
  pass

for year in range(2015,2020):
  dataRDD = filesRDD.filter(lambda x: x[0] == year)\
                    .flatMap(parse)\
                    .cache()
  
  badData += dataRDD.filter(lambda x: x[0] == '#ZZZ#')\
                   .map(lambda x: (x[0], x[1], x[2], x[3], x[4]))\
                   .collect()
  
  print(year, len(badData))
  
  # Convert to Dataframe
  df = dataRDD.filter(lambda x: x[0] != '#ZZZ#')\
              .toDF(list(dict_.keys()))
  
  df = df.withColumn('DATE', df['DATE'].cast(TimestampType())) 
  df = df.withColumn('SOURCE', df['SOURCE'].cast(ShortType()))
  df = df.withColumn('LATITUDE', df['LATITUDE'].cast(DoubleType()))
  df = df.withColumn('LONGITUDE', df['LONGITUDE'].cast(DoubleType()))
  df = df.withColumn('ELEVATION', df['ELEVATION'].cast(DoubleType()))
  
  df.write.mode('overwrite').parquet(f'dbfs:/mnt/mids-w261/data/datasets_final_project/new_weather_parquet_177/weather{year}.parquet')
  
  dataRDD.unpersist()

# COMMAND ----------

# Release cache after writing
for (id, rdd) in sc._jsc.getPersistentRDDs().items():
  rdd.unpersist()
try:
  weather_parquet.unpersist()
except:
  pass

try:
  df.unpersist()
except:
  pass

# COMMAND ----------

# After downnloading ALL the files successfully, I read now from parquet partitions now sitting on the S3 bucket mount
weather_parquet = spark.read.option("header", "true")\
                      .parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/new_weather_parquet_177/weather20*a.parquet")

# COMMAND ----------

display(weather_parquet)

# COMMAND ----------

# Count should be in the 630 Million range
weather_parquet.count()

# COMMAND ----------

weather_parquet.printSchema()

# COMMAND ----------

# Sanity checks, making sure I have data from every month of each year I downloaded data from
display(weather_parquet.select(f.year('DATE'), f.month('DATE')).distinct())

# COMMAND ----------

for col in weather_parquet.columns:
  print(col,weather_parquet.select(col).distinct().count())

# COMMAND ----------

weather_parquet.select('GG4').distinct().show()

# COMMAND ----------

weather_parquet.select('MV2').distinct().show()

# COMMAND ----------

weather_parquet.select('AX3').distinct().show()

# COMMAND ----------

weather_parquet.select('GL1').distinct().show()

# COMMAND ----------

weather_parquet.select('CO1').distinct().show()

# COMMAND ----------

weather_parquet.select('MV1').distinct().show()

# COMMAND ----------

weather_parquet.select('GD5').distinct().show()

# COMMAND ----------

weather_parquet.select('AX6').distinct().show()

# COMMAND ----------

weather_parquet.select('AW6').distinct().show()

# COMMAND ----------

weather_parquet.select('AL3').distinct().show()

# COMMAND ----------

weather_parquet.select('AX5').distinct().show()

# COMMAND ----------

weather_parquet.select('MW6').distinct().show()

# COMMAND ----------

weather_parquet.select('MW5').distinct().show()

# COMMAND ----------

weather_parquet.select('GE1').distinct().show()

# COMMAND ----------

weather_parquet.select('AX4').distinct().show()

# COMMAND ----------

weather_parquet.select('AW7').distinct().show()

# COMMAND ----------

