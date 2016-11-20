import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.functions._

// load file and remove header

val data = sc.textFile("file:/home/gustavo/DataScience/reduced_data.csv")

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
//0 a 25
// comma separator split

val allSplit = rows.map(line => line.split(";"))

// map parts to case class

val allData = allSplit.map( p => CC1(
p(0).toString,
p(1).trim.toDouble,
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

val rowsRDD = allDF.rdd.map(r => (r.getString(0), r.getDouble(1), r.getDouble(2), r.getDouble(3), r.getDouble(4), r.getDouble(5), r.getDouble(6), r.getDouble(7), r.getDouble(8), r.getDouble(9), r.getDouble(10), r.getDouble(11), r.getDouble(12), r.getDouble(13), r.getDouble(14), r.getDouble(15), r.getDouble(16), r.getDouble(17), r.getDouble(18), r.getDouble(19), r.getDouble(20), r.getDouble(21)))

rowsRDD.cache()

// convert data to RDD which will be passed to KMeans and cache the data. We are passing in RSI2, RSI_CLOSE_3, PERCENT_RANK_100, RSI_STREAK_2 and CRSI to KMeans. These are the attributes we want to use to assign the instance to a cluster

val vectors = allDF.rdd.map(r => Vectors.dense( r.getDouble(2), r.getDouble(3), r.getDouble(4), r.getDouble(5), r.getDouble(6), r.getDouble(7), r.getDouble(8), r.getDouble(9), r.getDouble(10), r.getDouble(11), r.getDouble(12), r.getDouble(13), r.getDouble(14), r.getDouble(15), r.getDouble(16), r.getDouble(17), r.getDouble(18), r.getDouble(19), r.getDouble(20), r.getDouble(21) ))

vectors.cache()

//KMeans model with 2 clusters and 20 iterations

val kMeansModel = KMeans.train(vectors, 3, 20)

//Print the center of each cluster

kMeansModel.clusterCenters.foreach(println)

// Get the prediction from the model with the ID so we can link them back to other information
val WSSSE = kMeansModel.computeCost(vectors)
println("Within Set Sum of Squared Errors = " + WSSSE)

val predictions = rowsRDD.map{r => (r._1, kMeansModel.predict(Vectors.dense(r._2, r._3, r._4, r._5, r._6, r._7, r._8, r._9, r._10, r._11, r._12, r._13, r._14, r._15, r._16, r._17, r._18, r._19, r._20, r._21) ))}

// convert the rdd to a dataframe

val predDF = predictions.toDF("ID", "CLUSTER")

var t = allDF.join(predDF, "id")

t.show()

//t.saveAsTextFile("predicted.csv")


import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

val data_no_norm = sc.textFile("file:/home/gustavo/DataScience/df.csv")

val header_no_norm = data_no_norm.first

val rows_no_norm = data_no_norm.filter(l => l != header_no_norm)

val allSplit_no_norm = rows_no_norm.map(line => line.split(";"))

// map parts to case class

val allData_no_norm = allSplit_no_norm.map( p => CC1(
p(0).toString,
p(1).trim.toDouble,
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

val allDF_no_norm = allData_no_norm.toDF()

var t_no_norm = allDF_no_norm.join(predDF, "id")


val labeledPoints = t_no_norm.rdd.map(r => LabeledPoint(r.getInt(25), Vectors.dense( r.getDouble(2), r.getDouble(3), r.getDouble(4), r.getDouble(5), r.getDouble(6), r.getDouble(7), r.getDouble(8), r.getDouble(9), r.getDouble(10), r.getDouble(11), r.getDouble(12), r.getDouble(13), r.getDouble(14), r.getDouble(15), r.getDouble(16), r.getDouble(17), r.getDouble(18), r.getDouble(19), r.getDouble(20), r.getDouble(21) )))

val splits = labeledPoints.randomSplit(Array(0.7, 0.3), seed = 5043l)

val trainingData = splits(0)
val testData = splits(1)

import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Gini

//val algorithm = Algo.Classification
//val impurity = Gini
//val maximumDepth = 3
//val treeCount = 20
//val featureSubsetStrategy = "auto"
//val seed = 5043

import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.RandomForest

// Train a RandomForest model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 3
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 20 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

//val model = RandomForest.trainClassifier(trainingData, new Strategy(algorithm, 
//  impurity, maximumDepth), treeCount, featureSubsetStrategy, seed)


val labeledPredictions = testData.map { labeledPoint =>
    val predictions = model.predict(labeledPoint.features)
    (labeledPoint.label, predictions)
}

import org.apache.spark.mllib.evaluation.MulticlassMetrics

val evaluationMetrics = new MulticlassMetrics(labeledPredictions.map(x => 
  (x._1, x._2)))


println("Confusion matrix:")
println(evaluationMetrics.confusionMatrix)

// Overall Statistics
val accuracy = evaluationMetrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = evaluationMetrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + evaluationMetrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + evaluationMetrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + evaluationMetrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + evaluationMetrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${evaluationMetrics.weightedPrecision}")
println(s"Weighted recall: ${evaluationMetrics.weightedRecall}")
println(s"Weighted F1 score: ${evaluationMetrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${evaluationMetrics.weightedFalsePositiveRate}")



import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}

val model2 = new LogisticRegressionWithLBFGS().setNumClasses(10).run(trainingData)

val labeledPredictions2 = testData.map { labeledPoint =>
    val predictions2 = model2.predict(labeledPoint.features)
    (labeledPoint.label, predictions2)
}


val evaluationMetrics2 = new MulticlassMetrics(labeledPredictions2.map(x => (x._1, x._2)))



println("Confusion matrix:")
println(evaluationMetrics2.confusionMatrix)

// Overall Statistics
val accuracy = evaluationMetrics2.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels2 = evaluationMetrics2.labels
labels2.foreach { l =>
  println(s"Precision($l) = " + evaluationMetrics2.precision(l))
}

// Recall by label
labels2.foreach { l =>
  println(s"Recall($l) = " + evaluationMetrics2.recall(l))
}

// False positive rate by label
labels2.foreach { l =>
  println(s"FPR($l) = " + evaluationMetrics2.falsePositiveRate(l))
}

// F-measure by label
labels2.foreach { l =>
  println(s"F1-Score($l) = " + evaluationMetrics2.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${evaluationMetrics2.weightedPrecision}")
println(s"Weighted recall: ${evaluationMetrics2.weightedRecall}")
println(s"Weighted F1 score: ${evaluationMetrics2.weightedFMeasure}")
println(s"Weighted false positive rate: ${evaluationMetrics2.weightedFalsePositiveRate}")






