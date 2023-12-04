import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class AwsSparkWinePredictionTest {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("awsSparkWinePredictionTest").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        String path = "s3:/cs643/ValidationDataset.csv";
        
        // Loading the data and casting relevant columns
        Dataset<Row> data = spark.read().format("csv").option("header", "true").option("sep", ";").load(path);

        // Renaming the 'quality' column to 'label'
        data = data.withColumnRenamed("quality", "label");

        // StringIndexer to convert string labels to float labels
        for (String colName : data.columns()) {
            if (!colName.equals("quality")) {
                data = data.withColumn(colName, data.col(colName).cast("float"));
            }
        }
        data = data.withColumnRenamed("quality", "label");

        // VectorAssembler to assemble feature vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(data.columns()).setOutputCol("features");
        Dataset<Row> df = assembler.transform(data).select("features", "indexedLabel");

        // Splitting the data into training and validation sets
        Dataset<Row>[] splits = df.randomSplit(new double[]{0.85, 0.15}, 1234);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> validationData = splits[1];

        // Random Forest Classifier
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("features")
                .setNumTrees(10);

        // Pipeline with stages: StringIndexer, VectorAssembler, and RandomForestClassifier
        Pipeline pipeline = new Pipeline().setStages(new StringIndexer[]{labelIndexer}).setStages(new VectorAssembler[]{assembler}).setStages(new RandomForestClassifier[]{rf});

        // Training the model
        PipelineModel model = pipeline.fit(trainingData);

        // Making predictions on the validation set
        Dataset<Row> predictions = model.transform(validationData);

        // Evaluating the model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);
        double accuracy = evaluator.evaluate(accuracy);
        
        System.out.println("F1-score for test:  " + f1);
        System.out.println("Accuracy for test: " + accuracy);

        sc.close();
    }
}
