package BigData.KafkaTwitterProducer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

object Recommendation {
  


  def main(args: Array[String]): Unit = {
    
    Logger.getLogger("org").setLevel(Level.ERROR)

    val sc = new SparkContext("local[*]", "classifier")

    val data = sc.textFile("C:/Users/VIGNESH/Desktop/BigDataHW3/hw3datasetnew/ratings.dat")
    val stars = data.map { line => val parts = line.split("::")
      Rating(parts(0).toInt, parts(1).toInt, parts(2).toInt)
    }
    val NewData = stars.randomSplit(Array(0.6, 0.4))
    val (trainingData, testData) = (NewData(0), NewData(1))

    val rank = 3
    val Iterations = 20
    val TrainingModel = ALS.train(trainingData, rank, Iterations, 0.01)

    val movie = testData.map { case Rating(userid, movieid, rating) => (userid, movieid) }
    val prediction =
      TrainingModel.predict(movie).map { case Rating(userid, movieid, rating) =>
        ((userid, movieid), rating)
      }

    val PredRating = testData.map { case Rating(userid, movieid, rating) =>
      ((userid, movieid), rating)
    }.join(prediction)

    val error = PredRating.map { case ((userid, movieid), (r1, r2)) =>
      val err = (r1 - r2)
      err
    }.mean()
    println("Accuracy = " + (1 - error))
  }

}