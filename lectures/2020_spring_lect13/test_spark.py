from pyspark import SparkContext

#master "local" and appname = "wordcount"
sc = SparkContext(master="local", appName="wordcount")

# creating a RDD from pg2701.txt
text = sc.textFile('pg2701.txt')

print(text.getNumPartitions())

# Running this program display the first 10 entries in the RDD:
print("RDD:", text.take(10))
