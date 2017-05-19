package BigData.KafkaTwitterProducer

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD

object Classification {
  class decisionTree {
    def train(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) = {
      val numClasses = 7
      val categoricalFeaturesInfo = Map[Int, Int]() // feature has continuous values
      val impurity = "entropy"
      val maxDepth = 5
      val maxBins = 100
      val model = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo,
        impurity, maxDepth, maxBins)
        
      val Prediction = testData.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }
      val accuracy = Prediction.filter(r => r._1 == r._2).count().toDouble / testData.count()
      println("Accuracy of Decision tree classifier = " + accuracy)
    }
}
  
  class naiveBayes {
    def train(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): Unit = {
      val model = NaiveBayes.train(trainData, lambda = 0.1, modelType = "multinomial")
      val predictionLabel = testData.map(p => (model.predict(p.features), p.label))

      val accuracy: Double = predictionLabel.filter(x => x._1 == x._2).count().toDouble / testData.count()
      println("Accuracy of Naive Bayes = " + accuracy)
    }
  }

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val sc = new SparkContext("local[*]", "classifier")
    val data = sc.textFile("C:/Users/VIGNESH/Desktop/BigDataHW3/hw3datasetnew/glass.data")
    val parserData = data.map { line =>
      val x = line.split(",").map(_.toDouble)
      LabeledPoint(x(10) - 1, Vectors.dense(x.tail))
    }
    
    val NewData = parserData.randomSplit(Array(0.6, 0.4))
    val (trainingData, testData) = (NewData(0), NewData(1))
    val decisionTree = new decisionTree
    decisionTree.train(trainingData, testData)
    val naiveBayes = new naiveBayes
    naiveBayes.train(trainingData, testData)

  }

}