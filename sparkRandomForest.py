from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark
import numpy as np
from pyspark.ml.linalg import Vectors

sc = pyspark.SparkContext()
sqlcontext = SQLContext(sc)

pandas_df = pd.read_csv('rank.csv')

pandas_df['split'] = np.random.randn(pandas_df.shape[0], 1)

msk = np.random.rand(len(pandas_df)) <= 0.7

train = pandas_df[msk]
test = pandas_df[~msk]


s_df = sqlcontext.createDataFrame(train)

trainingData=s_df.rdd.map(lambda x:(Vectors.dense(x[2:]), x[1])).toDF(["features", "label"])

featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(trainingData)


rf = RandomForestRegressor(featuresCol="indexedFeatures")

pipeline = Pipeline(stages=[featureIndexer, rf])

model = pipeline.fit(trainingData)


t_df = sqlcontext.createDataFrame(train)

testData=t_df.rdd.map(lambda x:(Vectors.dense(x[2:]), x[1])).toDF(["features", "label"])

predictions = model.transform(testData)

predictions.select("prediction", "label", "features").show(5)

evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# rfModel = model.stages[1]
# print(rfModel)