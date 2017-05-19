package BigData.KafkaTwitterProducer
import org.apache.log4j._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext


object Kmeans {
  
  def main(args: Array[String]) {
  
  val sparkConf = new SparkConf().setAppName("SparkSessionZipsExample").setMaster("local")
  val sc = new SparkContext(sparkConf)//.set("spark.some.config.option", "some-value")
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  
  val data = sqlContext.sparkContext.textFile("C:/Users/VIGNESH/Desktop/BigDataHW3/hw3datasetnew/itemusermat")
    val parsedData = data.map(v => (Vectors.dense(v.trim.split(' ').drop(1).map(_.toDouble)))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 10
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    val prediction = data.map(v => (v.split(" ")(0).toInt, clusters.predict(Vectors.dense(v.trim.split(' ').drop(1).map(_.toDouble)))))

    val movieData = sqlContext.sparkContext.textFile("C:/Users/VIGNESH/Desktop/BigDataHW3/hw3datasetnew/movies.dat").map { line =>
      val parts = line.split("::")
      (parts(0).toInt, (parts(1) + "," + parts(2)))
    }

    val NewData = prediction.join(movieData)

    val result = NewData.map {
      line => (line._2._1, (line._1, line._2._2))
    }.groupByKey()

    result.foreach(p => {
      println("\n\n\n")
      var size = p._2.size
      println("Cluster " + p._1 + " with size : " + size)
      val movie = p._2.toList
      if (size < 5) {
        for (i <- 0 to size - 1)
          println(movie(i)._1 + " ," + movie(i)._2)
      }
      else {
        for (i <- 0 to 4)
          println(movie(i)._1 + " ," + movie(i)._2)
      }
    }
    )
  
}
}