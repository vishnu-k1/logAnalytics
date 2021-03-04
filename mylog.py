from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, StringType
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, concat_ws, udf, collect_list, input_file_name
from pyspark.ml.clustering import KMeans

import numpy as np
import collections
import re
import pandas as pd
import matplotlib.pyplot as pl

loggingLevel = 'Info'
TStamp = '05 May 2020 18:38:04'
ThreadNum = '0'
Msg = 'NA'
Serial = '0'
logSchema = StructType([ \
    StructField("loggingLevel", StringType(), True), \
    StructField("TimeStamp", StringType(), True), \
    StructField("Thread", StringType(), True), \
    StructField("Message", StringType(), True),\
    StructField("Serial", StringType(), True)])

def parseLine(line):
    global loggingLevel
    global TStamp
    global ThreadNum
    global Serial

    if re.match(".*Thread-[0-9]+.*",line):
     fields = line.split('-')
     temp = fields[0].split(' ')
     loggingLevel = temp[0].replace(" ", "")
     TStamp = fields[0].replace(loggingLevel, "")
     ThreadNum = fields[2].split(' ')[0].replace(" ", "")
     Msg = fields[2].replace(ThreadNum, "")

     if re.match(".*\w+:[0-9]+:[a-z]+.*",Msg):
      num1 = re.search("\w+:[0-9]+:[a-z]+", Msg)
      Serial = num1[0].split(':')[1]
    else:
     Msg = line
    if len(Msg) < 1 and loggingLevel == 'INFO':
     Serial = 0
    return (loggingLevel,TStamp,ThreadNum,Msg,Serial)

def clean_text(c):
  c = str.lower(c)
  c = re.sub(r"&lt|&gt|;|div","",c)
  #c = split(c, "\\s+") tokenization...
  return c

conf = SparkConf().setMaster("local").setAppName("logIsolation")
sc = SparkContext(conf = conf)
file = sc.textFile("/home/vishnu/Desktop/data/bank/inv2/*")

parsed = file.map(parseLine)
sc2 = SQLContext(sc)
df= sc2.createDataFrame(parsed,logSchema)
df= df.withColumn("input_file", input_file_name())
#df.show(truncate=False)
txtCln = udf(clean_text)
df = df.select("Thread","Serial","Message").groupby("Thread","Serial").agg(concat_ws("--",collect_list("Message")).alias("MsgLine") )

df = df.withColumn("MsgLine",txtCln(df["Msgline"]))




tokenizer = Tokenizer(inputCol="MsgLine", outputCol="MsgLine2")
wordsData = tokenizer.transform(df)

remover = StopWordsRemover(inputCol='MsgLine2', outputCol='MsgLine3')
wordsData = remover.transform(wordsData)

TF = HashingTF(inputCol="MsgLine3", outputCol="Itf-MSG")
tfidfDf= TF.transform(wordsData)
#tfidfDf.show(truncate=False)

idf = IDF(inputCol="Itf-MSG", outputCol="Itf-MSG2")
idfd = idf.fit(tfidfDf)
tfidf = idfd.transform(tfidfDf)

kmeans = KMeans().setK(11).setSeed(1).setFeaturesCol('Itf-MSG2')
model = kmeans.fit(tfidf)
print(model.summary.trainingCost)
transformed = model.transform(tfidf)
transformed.show(truncate=False)
transformed.printSchema()
writetoFile = transformed.select("Thread","Serial","MsgLine","prediction")
writetoFile.repartition(1).write.csv('/home/vishnu/Desktop/out_cluster.csv')

sc.stop()

#wordsData.withColumn("words_clean", concat_ws(" - ",col("words_clean")))

#df_words.printSchema()
#wordsData.show(truncate=False)
# Calculate cost and plot
#cost = np.zeros(15)

#for k in range(2,15):
    #cost[k] = model.summary.trainingCost

# Plot the cost
#df_cost = pd.DataFrame(cost[2:])
#df_cost.columns = ["cost"]
#new_col = [2,3,4,5,6,7,8, 9,10,11,12,13,14]
#df_cost.insert(0, 'cluster', new_col)


#pl.plot(df_cost.cluster, df_cost.cost)
#pl.xlabel('Number of Clusters')
#pl.ylabel('Score')
#pl.title('Elbow Curve')
#pl.show()

#sql_results = sc2.sql("SELECT count(*),Serial FROM log WHERE group by Serial" )
#sql_results = sc2.sql("SELECT distinct(Serial),Thread FROM log where Serial = '0' order by TimeStamp" )
#sql_results.coalesce(1).write.csv('/home/vishnu/Desktop/out12.csv')
#wordsData.select("Message","words").createDataFrame()
#df2 = sc2.createDataFrame()

#tfidfDf.createOrReplaceTempView("log")
#df2.write.format("csv").save('/home/vishnu/Desktop/out2')



