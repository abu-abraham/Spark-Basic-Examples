from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark
from pyspark.sql.functions import *
import pandas as pd

import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

sc = pyspark.SparkContext()
sqlcontext = SQLContext(sc)


pandas_df = pd.read_csv('rank.csv')

(trainingData, testData) = pandas_df.randomSplit([0.7, 0.3])

pandas_df['split'] = np.random.randn(pandas_df.shape[0], 1)

msk = np.random.rand(len(pandas_df)) <= 0.7

train = pandas_df[msk]
test = pandas_df[~msk]


s_df = sqlcontext.createDataFrame(train)


trainingData=s_df.rdd.map(lambda x:(Vectors.dense(x[2:3]), x[1])).toDF(["features", "label"])

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(trainingData)

dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

pipeline = Pipeline(stages=[featureIndexer, dt])

model = pipeline.fit(trainingData)


test_df = sqlcontext.createDataFrame(test)
testData=test_df.rdd.map(lambda x:(Vectors.dense(x[2:3]), x[1])).toDF(["features", "label"])
predictions = model.transform(testData)

predictions.select("prediction", "label", "features").show(5)
