from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import mlflow
import logging
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/logs/lakehouse.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_spark_session():
    logger.info("Creating Spark session for Gold layer...")
    spark = SparkSession.builder \
        .appName("Lakehouse Gold Layer") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    return spark

def create_aggregations(df):
    logger.info("Creating aggregations...")
    

    product_aggs = df.groupBy("product_category") \
        .agg(
            count("*").alias("total_purchases"),
            avg("purchase_amount").alias("avg_purchase_amount"),
            sum("purchase_amount").alias("total_revenue"),
            approx_count_distinct("age").alias("unique_customers")
        )
    
    customer_aggs = df.groupBy("age_group", "income_category") \
        .agg(
            count("*").alias("total_purchases"),
            avg("purchase_amount").alias("avg_purchase_amount"),
            sum("purchase_amount").alias("total_revenue")
        )
    
    time_aggs = df.groupBy("purchase_year", "purchase_month") \
        .agg(
            count("*").alias("total_purchases"),
            sum("purchase_amount").alias("monthly_revenue"),
            avg("purchase_amount").alias("avg_purchase_amount")
        )
    
    return product_aggs, customer_aggs, time_aggs

def prepare_ml_data(df):
    logger.info("Preparing data for ML model...")
    
    categorical_cols = ["product_category", "education", "income_category", "age_group"]
    
    stages = []
    
    for col in categorical_cols:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
        encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
        stages += [indexer, encoder]
    
    numeric_cols = ["age", "income"]
    
    assembler_inputs = [f"{col}_vec" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    stages += [assembler]
    
    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    prepared_data = pipeline_model.transform(df)
    
    return prepared_data, pipeline_model

def train_model(df):
    logger.info("Training model...")
    
    prepared_data, pipeline_model = prepare_ml_data(df)
    
    train_data, test_data = prepared_data.randomSplit([0.8, 0.2], seed=42)
    
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="purchase_amount",
        numTrees=100,
        maxDepth=10
    )
    
    model = rf.fit(train_data)
    
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(
        labelCol="purchase_amount",
        predictionCol="prediction",
        metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)
    
    return model, pipeline_model, rmse

def save_results(product_aggs, customer_aggs, time_aggs, model, pipeline_model, rmse):
    logger.info("Saving results...")
    
    # Сохранение агрегаций
    product_aggs.write.format("delta") \
        .mode("overwrite") \
        .save("/app/data/gold/product_analytics")
        
    customer_aggs.write.format("delta") \
        .mode("overwrite") \
        .save("/app/data/gold/customer_analytics")
        
    time_aggs.write.format("delta") \
        .mode("overwrite") \
        .save("/app/data/gold/time_analytics")
    
    # Логирование метрик и параметров в MLflow
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("rmse", rmse)
    
    # Сохранение модели
    mlflow.spark.log_model(pipeline_model, "feature_pipeline")
    mlflow.spark.log_model(model, "random_forest_model")

def main():
    logger.info("Starting Gold layer processing...")
    try:
        # Создание директорий
        os.makedirs("/app/data/gold", exist_ok=True)
        
        # Инициализация Spark и MLflow
        spark = create_spark_session()
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("customer_analysis")
        
        # Загрузка данных из silver слоя
        logger.info("Loading data from Silver layer...")
        silver_df = spark.read.format("delta").load("/app/data/silver/customer_purchases")
        
        # Создание агрегаций
        product_aggs, customer_aggs, time_aggs = create_aggregations(silver_df)
        
        # Обучение модели
        with mlflow.start_run():
            model, pipeline_model, rmse = train_model(silver_df)
            save_results(product_aggs, customer_aggs, time_aggs, model, pipeline_model, rmse)
        
        logger.info("Gold layer processing completed successfully!")
    except Exception as e:
        logger.error(f"Error in Gold layer: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 