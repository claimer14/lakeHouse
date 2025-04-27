from pyspark.sql import SparkSession
from delta.tables import DeltaTable

def create_spark_session():
    spark = SparkSession.builder \
        .appName("Lakehouse Optimizer") \
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

def optimize_delta_table(spark, path):
    try:
        delta_table = DeltaTable.forPath(spark, path)
        delta_table.optimize().executeCompaction()
        print(f"Successfully optimized Delta table at {path}")
    except Exception as e:
        print(f"Error optimizing Delta table at {path}: {str(e)}")

def main():
    spark = create_spark_session()
    
    paths = [
        "data/bronze/sample_data",
        "data/silver/processed_data",
        "data/gold/age_statistics",
        "data/gold/education_statistics"
    ]
    
    for path in paths:
        optimize_delta_table(spark, path)
    
    spark.stop()

if __name__ == "__main__":
    main() 