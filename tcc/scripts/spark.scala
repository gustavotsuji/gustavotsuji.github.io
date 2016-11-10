val sqlContext = new org.apache.spark.sql.SQLContext(sc)

//val tableData = sqlContext.read.format("jdbc").option("driver", "org.sqlite.JDBC").option("url","jdbc:sqlite:/Users/gkendi/Desktop/DataScience/database.sqlite").option("dbtable", "loan").load()


import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

// Load and parse the data
val data = sc.textFile("/Users/gkendi/Desktop/DataScience/reduced_data.csv")
val parsedData = data.map(s => Vectors.dense(s.split(';').map(_.toDouble))).cache()

// Cluster the data into two classes using KMeans
val numClusters = 3
val numIterations = 20



val clusters = KMeans.train(parsedData, numClusters, numIterations)

// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)
println("Within Set Sum of Squared Errors = " + WSSSE)



import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.functions._

// load file and remove header

val data = sc.textFile("file:/Users/gkendi/Desktop/DataScience/reduced_data.csv")

val header = data.first

val rows = data.filter(l => l != header)

// define case class

case class CC1(
 id: String,
 funded_amnt: Double,
 funded_amnt_inv: Double,
 loan_amnt: Double,
 term_float: Double,
 int_rate_float: Double,
 installment: Double,
 annual_inc: Double,
 dti: Double,
 delinq_2yrs: Double,
 inq_last_6mths: Double,
 open_acc: Double,
 pub_rec: Double,
 revol_bal: Double,
 total_acc: Double,
 out_prncp: Double,
 out_prncp_inv: Double,
 total_pymnt: Double,
 total_pymnt_inv: Double,
 total_rec_prncp: Double,
 total_rec_int: Double,
 total_rec_late_fee: Double,
 recoveries: Double,
 collection_recovery_fee: Double,
 last_pymnt_amnt: Double)

// comma separator split

val allSplit = rows.map(line => line.split(";"))

// map parts to case class

val allData = allSplit.map( p => CC1(
p(0).toString,
p(1).toDouble,
p(2).trim.toDouble, 
p(3).trim.toDouble, 
p(4).trim.toDouble, 
p(5).trim.toDouble, 
p(6).trim.toDouble, 
p(7).trim.toDouble, 
p(8).trim.toDouble, 
p(9).trim.toDouble, 
p(10).trim.toDouble, 
p(11).trim.toDouble, 
p(12).trim.toDouble, 
p(13).trim.toDouble, 
p(14).trim.toDouble, 
p(15).trim.toDouble, 
p(16).trim.toDouble, 
p(17).trim.toDouble, 
p(18).trim.toDouble, 
p(19).trim.toDouble, 
p(20).trim.toDouble, 
p(21).trim.toDouble, 
p(22).trim.toDouble, 
p(23).trim.toDouble, 
p(24).trim.toDouble))

// convert rdd to dataframe

val allDF = allData.toDF()

// convert back to rdd and cache the data

val rowsRDD = allDF.rdd.map(r => (r.getString(0), r.getDouble(1), r.getDouble(2), r.getDouble(3), r.getDouble(4), r.getDouble(5), r.getDouble(6), r.getDouble(7), r.getDouble(8), r.getDouble(9), r.getDouble(10), r.getDouble(11), r.getDouble(12), r.getDouble(13), r.getDouble(14), r.getDouble(15), r.getDouble(16), r.getDouble(17), r.getDouble(18), r.getDouble(19), r.getDouble(20), r.getDouble(21), r.getDouble(22), r.getDouble(23), r.getDouble(24)))

rowsRDD.cache()

// convert data to RDD which will be passed to KMeans and cache the data. We are passing in RSI2, RSI_CLOSE_3, PERCENT_RANK_100, RSI_STREAK_2 and CRSI to KMeans. These are the attributes we want to use to assign the instance to a cluster

val vectors = allDF.rdd.map(r => Vectors.dense( r.getDouble(2), r.getDouble(3), r.getDouble(4), r.getDouble(5), r.getDouble(6), r.getDouble(7), r.getDouble(8), r.getDouble(9), r.getDouble(10), r.getDouble(11), r.getDouble(12), r.getDouble(13), r.getDouble(14), r.getDouble(15), r.getDouble(16), r.getDouble(17), r.getDouble(18), r.getDouble(19), r.getDouble(20), r.getDouble(21), r.getDouble(22), r.getDouble(23), r.getDouble(24) ))

vectors.cache()

//KMeans model with 2 clusters and 20 iterations

val kMeansModel = KMeans.train(vectors, 2, 20)

//Print the center of each cluster

kMeansModel.clusterCenters.foreach(println)

// Get the prediction from the model with the ID so we can link them back to other information

val predictions = rowsRDD.map{r => (r._1, kMeansModel.predict(Vectors.dense(r._6, r._7, r._8, r._9, r._10) ))}

// convert the rdd to a dataframe

val predDF = predictions.toDF("ID", "CLUSTER")


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)


