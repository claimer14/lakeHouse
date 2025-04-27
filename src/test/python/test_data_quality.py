import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
import os

@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder \
        .appName("Data Quality Tests") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .master("local[1]") \
        .getOrCreate()
    yield spark
    spark.stop()

def test_bronze_layer(spark):
    """Проверка качества данных в bronze слое"""
    df = spark.read.format("delta").load("data/bronze/sample_data")
    
    assert df.count() >= 100000, "Количество строк должно быть >= 100000"
    
    required_columns = {"age", "income", "education", "purchase_date", "purchase_amount", "product_category"}
    assert set(df.columns) == required_columns, "Отсутствуют необходимые колонки"
    
    assert df.schema["age"].dataType.typeName() == "integer", "age должен быть integer"
    assert df.schema["income"].dataType.typeName() == "double", "income должен быть double"
    assert df.schema["education"].dataType.typeName() == "string", "education должен быть string"
    assert df.schema["purchase_date"].dataType.typeName() == "date", "purchase_date должен быть date"
    assert df.schema["purchase_amount"].dataType.typeName() == "double", "purchase_amount должен быть double"
    assert df.schema["product_category"].dataType.typeName() == "string", "product_category должен быть string"
    
    for column in required_columns:
        null_count = df.filter(col(column).isNull()).count()
        assert null_count == 0, f"В колонке {column} есть null значения"

def test_silver_layer(spark):
    """Проверка качества данных в silver слое"""
    df = spark.read.format("delta").load("data/silver/processed_data")
    
    assert df.count() >= 100000, "Количество строк должно быть >= 100000"
    
    assert df.schema["age"].dataType.typeName() == "integer", "age должен быть integer"
    assert df.schema["income"].dataType.typeName() == "double", "income должен быть double"
    assert df.schema["education"].dataType.typeName() == "string", "education должен быть string"
    assert df.schema["purchase_date"].dataType.typeName() == "date", "purchase_date должен быть date"
    assert df.schema["purchase_amount"].dataType.typeName() == "double", "purchase_amount должен быть double"
    assert df.schema["product_category"].dataType.typeName() == "string", "product_category должен быть string"
    
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        assert null_count == 0, f"В колонке {column} есть null значения"
    
    min_date = df.select("purchase_date").agg({"purchase_date": "min"}).collect()[0][0]
    max_date = df.select("purchase_date").agg({"purchase_date": "max"}).collect()[0][0]
    assert min_date.year >= 2020, "Даты должны быть не раньше 2020 года"
    assert max_date.year <= 2024, "Даты должны быть не позже 2024 года"

def test_gold_layer(spark):
    """Проверка качества данных в gold слое"""
    age_stats = spark.read.format("delta").load("data/gold/age_statistics")
    assert age_stats.count() > 0, "Должны быть агрегации по возрасту"
    assert set(age_stats.columns) == {"age_group", "avg_income", "avg_purchase", "customer_count"}, \
        "Неверные колонки в age_statistics"
    
    education_stats = spark.read.format("delta").load("data/gold/education_statistics")
    assert education_stats.count() > 0, "Должны быть агрегации по образованию"
    assert set(education_stats.columns) == {"education", "avg_income", "avg_purchase", "customer_count"}, \
        "Неверные колонки в education_statistics"

def main():
    spark = create_spark_session()
    
    print("Проверка bronze слоя...")
    test_bronze_layer(spark)
    print("✅ Bronze слой прошел проверку")
    
    print("\nПроверка silver слоя...")
    test_silver_layer(spark)
    print("✅ Silver слой прошел проверку")
    
    print("\nПроверка gold слоя...")
    test_gold_layer(spark)
    print("✅ Gold слой прошел проверку")
    
    spark.stop()
    print("\nВсе проверки пройдены успешно!")

if __name__ == "__main__":
    main() 