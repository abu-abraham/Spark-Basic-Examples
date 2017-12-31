from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark
from pyspark.sql.functions import *
from pyspark.mllib.feature import StandardScaler

from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.mllib.evaluation import RegressionMetrics
import pandas as pd

import numpy as np

from pyspark.ml.linalg import Vectors

def showDetails(model):
    summary = model.summary
    print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    print("T Values: " + str(summary.tValues))
    print("P Values: " + str(summary.pValues))
    print("Dispersion: " + str(summary.dispersion))
    print("Null Deviance: " + str(summary.nullDeviance))
    print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    print("Deviance: " + str(summary.deviance))
    print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    print("AIC: " + str(summary.aic))
    print("Deviance Residuals: ")
    summary.residuals().show()

sc = pyspark.SparkContext()
sqlcontext = SQLContext(sc)


glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)
pandas_df = pd.read_csv('rank.csv')

pandas_df['split'] = np.random.randn(pandas_df.shape[0], 1)

msk = np.random.rand(len(pandas_df)) <= 0.7

train = pandas_df[msk]
test = pandas_df[~msk]


s_df = sqlcontext.createDataFrame(train)


# In spark.ml all features need to be vectors in a single column, usually named features

trainingData=s_df.rdd.map(lambda x:(Vectors.dense(x[2:]), x[1])).toDF(["features", "label"])
model = glr.fit(trainingData)

# Print the coefficients and intercept for generalized linear regression model
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

#showDetails(model)

test_df = sqlcontext.createDataFrame(test)

testData=test_df.rdd.map(lambda x:(Vectors.dense(x[2:]), x[1])).toDF(["features", "label"])
predictions = model.transform(testData)

# # Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

