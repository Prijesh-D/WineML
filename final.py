

## import necessary libraries
import time
import sys
from pyspark import SparkContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel

##we need to get the file that is going to be used for testing the dataset, since the professor will give a different test file to test the model we will use Validation Dataset
fileToTest=sys.argv[1]

## runs locally with 1 core with the appname of Wine_Taste_Prediction##
spark = SparkSession.builder.master("local").appName("Wine_Taste_Prediction").getOrCreate()             
spark.sparkContext.setLogLevel('WARN')

## get the dataset from the csv that was inputted
testing_dataset = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv(fileToTest)

## prepares the feature columns that aren't quality. and transforms the training data that the random forest well then use to model
feature = [c for c in testing_dataset.columns if c != 'quality']
assembler_test = VectorAssembler(inputCols=feature, outputCol="features")
testing_transform = assembler_test.transform(testing_dataset)


## RandomForestClassificationModel is loads the model that is saved in the s3 bucket
model= RandomForestClassificationModel.load("s3://filesforprogrammingassignment2/wine_train_model.model")

##our evaluator is also loaded as well to get the metric name of f1 and its accuracy in the model.
modelEvaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
transformed_data = model.transform(testing_transform)
print(modelEvaluator.getMetricName(), 'accuracy :', modelEvaluator.evaluate(transformed_data))