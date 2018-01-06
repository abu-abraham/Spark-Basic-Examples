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
from pyspark.ml.feature import IndexToString, StringIndexer
import numpy as np
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


def logisiticRegression():
    sc = pyspark.SparkContext()
    sqlcontext = SQLContext(sc)

    glr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    pandas_df = pd.read_csv('mushrooms.csv')

    s_df = sqlcontext.createDataFrame(pandas_df)


    categorical_columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

    all_stages = [StringIndexer(inputCol=c, outputCol='stringindexed_' + c) for c in categorical_columns]

    pipeline = Pipeline(stages=all_stages)
    pipeline_mode = pipeline.fit(s_df)

    df_coded = pipeline_mode.transform(s_df)
    selected_columns = ['stringindexed_' + c for c in categorical_columns]
    df_coded = df_coded.select(selected_columns)


    (training_df, test_df) = df_coded.randomSplit([0.7, 0.3])



    trainingData=training_df.rdd.map(lambda x:(Vectors.dense(x[1:]), x[0])).toDF(["features", "label"])
    model = glr.fit(trainingData)

    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))

    testData=test_df.rdd.map(lambda x:(Vectors.dense(x[1:]), x[0])).toDF(["features", "label"])
    predictions = model.transform(testData)

    predictions.select("prediction", "label", "features").show(40,False)
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='f1') 

    print evaluator.getMetricName(), 'accuracy:', evaluator.evaluate(predictions)


def decisionTree():
    sc = pyspark.SparkContext()
    sqlcontext = SQLContext(sc)
    pandas_df = pd.read_csv('mushrooms.csv')

    s_df = sqlcontext.createDataFrame(pandas_df)

    categorical_columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

    all_stages = [StringIndexer(inputCol=c, outputCol='stringindexed_' + c) for c in categorical_columns]
    pipeline = Pipeline(stages=all_stages)
    pipeline_mode = pipeline.fit(s_df)

    df_coded = pipeline_mode.transform(s_df)
    selected_columns = ['stringindexed_' + c for c in categorical_columns]
    df_coded = df_coded.select(selected_columns)
    (training_df, test_df) = df_coded.randomSplit([0.8, 0.2])
    trainingData=training_df.rdd.map(lambda x:(Vectors.dense(x[1:]), x[0])).toDF(["features", "label"])

    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    model = dt.fit(trainingData)

    testData=test_df.rdd.map(lambda x:(Vectors.dense(x[1:]), x[0])).toDF(["features", "label"])
    predictions = model.transform(testData)

    predictions.select("prediction", "label", "features").show(500,False)

    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='f1') 

    print evaluator.getMetricName(), 'accuracy:', evaluator.evaluate(predictions)
    

