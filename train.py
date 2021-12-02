
## import necessary libraries
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier



## runs locally with 4 cores with the appname of Wine_Taste_Prediction
spark = SparkSession.builder.master("local[4]").appName("Wine_Taste_Prediction").getOrCreate()

##sets the log levels because we don't want to have any uncessary cluterage
spark.sparkContext.setLogLevel('WARN')

##gets the data from the file located in the s3 bucket
training_dataset = spark.read.format('csv').options(delimiter=';',header='true', inferSchema='true').csv("s3://filesforprogrammingassignment2/TrainingDataset.csv")                                      

## prepares the feature columns that aren't quality. and transforms the training data that the random forest well then use to model
featureColumns = [c for c in training_dataset.columns if c != 'quality']
assembler_t = VectorAssembler(inputCols=featureColumns, outputCol="features")
training_transformed = assembler_t.transform(training_dataset)
training_transformed.cache()

## variable is initialized to model and it is saved to the s3 bucket.
random_forest = RandomForestClassifier(labelCol='""""quality"""""', featuresCol="features", numTrees=10)
model = random_forest.fit(training_transformed)

model.write().overwrite().save("s3://filesforprogrammingassignment2/wine_train_model.model")
print("Model is saved to the s3 bucket, you can now predict!")