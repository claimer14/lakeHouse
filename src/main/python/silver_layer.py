from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
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
    logger.info("Creating Spark session for Silver layer...")
    spark = SparkSession.builder \
        .appName("Lakehouse Silver Layer") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .config("spark.delta.optimizeWrite.enabled", "true") \
        .config("spark.delta.autoCompact.enabled", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .master("local[2]") \
        .getOrCreate()
    return spark

def clean_data(df):
    logger.info("Cleaning and processing data...")
    
    df = df.dropDuplicates()
    
    df = df.filter(
        (col("age").between(18, 100)) &  
        (col("income") > 0) &           
        (col("purchase_amount") >= 0)    
    )
    
    df = df.withColumn("purchase_year", year(col("purchase_date"))) \
          .withColumn("purchase_month", month(col("purchase_date"))) \
          .withColumn("income_category", when(col("income") < 30000, "low")
                                      .when(col("income") < 70000, "medium")
                                      .when(col("income") < 150000, "high")
                                      .otherwise("luxury")) \
          .withColumn("age_group", when(col("age") < 25, "18-24")
                                 .when(col("age") < 35, "25-34")
                                 .when(col("age") < 50, "35-49")
                                 .when(col("age") < 65, "50-64")
                                 .otherwise("65+"))
    
    df = df.repartition(4, "income_category", "product_category")
    
    return df

def save_optimized(df, path):
    logger.info(f"Saving processed data to {path}...")
    
    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .partitionBy("income_category", "product_category") \
        .save(path)
    
    spark = df.sparkSession
    spark.sql(f"OPTIMIZE delta.`{path}`")
    spark.sql(f"VACUUM delta.`{path}`")

def main():
    logger.info("Starting Silver layer processing...")
    try:

        os.makedirs("/app/data/silver", exist_ok=True)

        spark = create_spark_session()
        
        logger.info("Loading data from Bronze layer...")
        bronze_df = spark.read.format("delta").load("/app/data/bronze/customer_purchases")
        
        silver_df = clean_data(bronze_df)
        
        # Сохранение результатов
        save_optimized(silver_df, "/app/data/silver/customer_purchases")
        
        logger.info("Silver layer processing completed successfully!")
    except Exception as e:
        logger.error(f"Error in Silver layer: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 