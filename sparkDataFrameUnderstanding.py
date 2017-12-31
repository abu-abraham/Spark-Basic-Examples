from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark
from pyspark.sql.functions import *

#Refrence for more : https://www.nodalpoint.com/spark-dataframes-from-csv-files/ and http://www.techpoweredmath.com/spark-dataframes-mllib-tutorial/#.WkcDHnWWanw  

sc = pyspark.SparkContext()
sqlcontext = SQLContext(sc)
df = sqlcontext.read.load('rank.csv',format='com.databricks.spark.csv',header='true',inferSchema='true')
print df.count()
print df.dtypes
df.describe().dtypes
df.filter(df.Rank < 10).show()
df.select(max("Rank")).show() #pyspark.sql.functions REQUIRED

#spark-mllib uses RDDs.
#spark-ml uses DataFrames.     

df.registerTempTable("rank")
sqlcontext.sql("SELECT * FROM rank").show()

df = df.select('Rank','Mark','Gender')
df.describe(['Rank','Mark','Gender']).show()

