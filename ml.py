from pyspark.sql import SparkSession
from datetime import datetime, date
import numpy as np
import warnings
from datetime import datetime
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

warnings. simplefilter(action='ignore', category=FutureWarning)

def main():


    # User inputs of state name, event and number of predicting years
    state_input = input("Enter a state (E.g: CA for California): " )
    print(state_input)
    event_input = input("Enter a disaster event: " )
    print(event_input)
    year_num = input("Enter number of year for forcasting: " )
    print(year_num)

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.option("header",True).csv("us_disaster_declarations.csv")
    

    # Filtering out the data based on user inputs
    filter_data = df.filter(df.state == state_input) 
    filter_data = filter_data.filter(filter_data.incident_type == event_input)
    filter_data = filter_data.withColumn("declaration_date", substring(filter_data.declaration_date,0,10))
    filter_data = filter_data.withColumn("declaration_date", unix_timestamp(col('declaration_date'),"yyyy-MM-dd"))
    filter_data = filter_data.select(col("declaration_date"))
    filter_data = filter_data.withColumn("occur", lit(1))
    first = filter_data.first()['declaration_date']
    last = filter_data.collect()[-1]['declaration_date']


    # Filling dates that the event didn't occur into the training data 
    one_data = np.array(filter_data.select("declaration_date").collect())
    zero_data = []
    columns = ['declaration_date', 'occur']

    while first < last:
        if first not in one_data:
            zero_data.append([first,0])
        first += 86400

    union_df = spark.createDataFrame(zero_data, columns)
    filter_data = filter_data.union(union_df)


    # Testing data of 365 days times the user input that used for forecasting
    testing = []
    for i in range(1,366*int(year_num)):
        last += 86400
        testing.append([last,-1])

    testing_data = spark.createDataFrame(testing, columns)
    trainingData = filter_data.rdd.map(lambda x:(Vectors.dense(x[0]), x[-1])).toDF(["declaration_date", "occur"])
    testingData = testing_data.rdd.map(lambda x:(Vectors.dense(x[0]), x[-1])).toDF(["declaration_date", "occur"])


    #Linear Regression (Regression)
    model = LinearRegression(featuresCol = 'declaration_date', labelCol = 'occur', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    fitModel = model.fit(trainingData)    
    predictions = fitModel.transform(testingData)
    plot = predictions.toPandas()
    testing_data = testing_data.toPandas()
    plt.plot(testing_data["declaration_date"],plot["prediction"])
    plt.ylabel('Posibility')
    plt.xlabel('Date')
    plt.title('Linear Regression (Regression)')
    plt.savefig('LinearR_Re.png')
    trainingSummary = fitModel.summary
    print("Linear Regression (Regression) RMSE: %f" % trainingSummary.rootMeanSquaredError)
    

    # Decision Tree (Regression)
    dt = DecisionTreeRegressor(labelCol="occur", featuresCol="declaration_date")
    dtmodel = dt.fit(trainingData)
    predictions = dtmodel.transform(testingData)
    plot = predictions.toPandas()
    plt.plot(testing_data["declaration_date"],plot["prediction"])
    plt.ylabel('Posibility')
    plt.xlabel('Date')
    plt.title('Decision Tree (Regression)')
    plt.savefig('DecisionT_Re.png')
    (train, test) = trainingData.randomSplit([0.7, 0.3])
    dt = DecisionTreeRegressor(labelCol="occur", featuresCol="declaration_date")
    model = dt.fit(train)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        labelCol="occur", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Decision Tree (Regression) RMSE: %g" % rmse)


    # Logistic Regression (Classification)
    lr = LogisticRegression(featuresCol = 'declaration_date', labelCol = 'occur', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testingData)
    plot = predictions.toPandas()
    plt.plot(testing_data["declaration_date"],plot["prediction"])
    plt.ylabel('Posibility')
    plt.xlabel('Date')
    plt.title('Logistic Regression (Classification)')
    plt.savefig('LogR_Cl.png')
    trainingSummary = lrModel.summary
    accuracy = trainingSummary.accuracy
    print("Logistic Regression (Classification) accuracy: " + str(accuracy))


    # Decision Tree (Classification)
    dt = DecisionTreeClassifier(labelCol="occur", featuresCol="declaration_date")
    dtmodel = dt.fit(trainingData)
    predictions = dtmodel.transform(testingData)
    plot = predictions.toPandas()
    plt.plot(testing_data["declaration_date"],plot["prediction"])
    plt.ylabel('Posibility')
    plt.xlabel('Date')
    plt.title('Decision Tree (Classification)')
    plt.savefig('DecisionT_Cl.png')
    (train, test) = trainingData.randomSplit([0.7, 0.3])
    model = dt.fit(train)
    predictions = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="occur", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Decision Tree (Classification) accuracy: " + str(accuracy))

    spark.stop()

if __name__ == '__main__':
    main()