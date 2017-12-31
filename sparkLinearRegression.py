from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark
from pyspark.sql.functions import *
from pyspark.mllib.feature import StandardScaler

from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics

sc = pyspark.SparkContext()
sqlcontext = SQLContext(sc)
df = sqlcontext.read.load('rank.csv',format='com.databricks.spark.csv',header='true',inferSchema='true')
df = df.select('Rank','Mark','Gender')
features = df.rdd.map(lambda row: row[1:])

standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)

lab = df.rdd.map(lambda row: row[0])
transformedData = lab.zip(features_transform)

transformedData = transformedData.map(lambda row: LabeledPoint(row[0],[row[1]]))

trainingData, testingData = transformedData.randomSplit([.8,.2],seed=1234)
linearModel = LinearRegressionWithSGD.train(trainingData,1000,.2)


prediObserRDDout = testingData.map(lambda row: (float(linearModel.predict(row.features[0])),row.label))
metrics = RegressionMetrics(prediObserRDDout)

print metrics.rootMeanSquaredError

print linearModel.predict([7.49297445326,3.52055958053])

