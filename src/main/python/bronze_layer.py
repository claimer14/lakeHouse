from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime, timedelta
import os
import random
import logging
from delta.tables import DeltaTable


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
    logger.info("Creating Spark session...")
    spark = SparkSession.builder \
        .appName("Lakehouse Bronze Layer") \
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
        .config("spark.sql.catalogImplementation", "hive") \
        .master("local[2]") \
        .getOrCreate()
    return spark

def generate_sample_data(spark):
    logger.info("Generating sample data...")
    # Определяем схему данных
    schema = StructType([
        StructField("age", IntegerType(), False),
        StructField("income", DoubleType(), False),
        StructField("education", StringType(), False),
        StructField("purchase_amount", DoubleType(), False),
        StructField("purchase_date", DateType(), False),
        StructField("product_category", StringType(), False)
    ])
    
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2020, 1, 1)
    data = []
    for i in range(100000):
        days_range = (end_date - start_date).days
        purchase_date = (start_date + timedelta(days=random.randint(0, days_range))).date()
        income = float(format(random.uniform(20000, 200000), '.2f'))
        purchase_amount = float(format(random.uniform(0, 1000), '.2f'))
        data.append((
            random.randint(18, 80),  # age
            income,  # income
            random.choice(["high_school", "bachelor", "master", "phd"]),  # education
            purchase_amount,  # purchase_amount
            purchase_date,  # purchase_date
            random.choice(["electronics", "clothing", "books", "food", "other"])  # product_category
        ))
    
    df = spark.createDataFrame(data, schema=schema)

    df = df.repartition(4)
    
    return df

def save_optimized(spark, df, path):
    logger.info(f"Saving data to {path}...")

    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .partitionBy("education") \
        .save(path)

def main():
    logger.info("Starting Bronze layer processing...")
    try:

        os.makedirs("/app/data/bronze", exist_ok=True)
        os.makedirs("/app/logs", exist_ok=True)
        
        spark = create_spark_session()
        df = generate_sample_data(spark)
        save_optimized(spark, df, "/app/data/bronze/customer_purchases")
        logger.info("Bronze layer processing completed successfully!")
    except Exception as e:
        logger.error(f"Error in Bronze layer: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 