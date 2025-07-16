# Advanced Databricks Data Architecture Pipeline

This project demonstrates a robust, production-ready data pipeline implementing Bronze, Silver, and Gold layers with data quality checks, logging, retry logic, and Delta Lake optimizations.

---

## Pipeline Code

```python
import re
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    current_timestamp, lit, when, avg, count, round, col
)
import logging

# --- Configurable Parameters ---
RAW_TABLE = 'ai_job_trends_dataset'
BRONZE_TABLE = 'ai_jobs_data_bronze'
SILVER_TABLE = 'ai_jobs_data_silver'
GOLD_TABLE = 'ai_jobs_data_gold'
PARTITION_COL = 'industry'
MAX_RETRIES = 3

# Set up logging
logger = logging.getLogger("DataPipelineLogger")
logger.setLevel(logging.INFO)

# Simple in-memory data quality rules
DATA_QUALITY_RULES = [
    {
        "column": "median_salary_usd",
        "rule": lambda df: df.filter(col("median_salary_usd") >= 0),
        "description": "Median salary must be non-negative"
    },
    {
        "column": "ssn",
        "rule": lambda df: df.filter(col("ssn").isNotNull()),
        "description": "SSN cannot be null"
    }
]

def retry_operation(func, max_retries=3):
    """Decorator to retry functions on failure."""
    def wrapper(*args, **kwargs):
        attempts = 0
        while attempts < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempts += 1
                logger.error(f"Attempt {attempts} failed with error: {e}")
                if attempts == max_retries:
                    raise
                logger.info(f"Retrying {func.__name__} (attempt {attempts + 1})...")
    return wrapper

def clean_column_names(df: DataFrame) -> DataFrame:
    """Clean column names to snake_case and remove special chars."""
    def clean(name):
        name = name.lower()
        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")
        return name
    new_cols = [clean(c) for c in df.columns]
    return df.toDF(*new_cols)

@retry_operation
def write_delta_table(df: DataFrame, table_name: str, partition_col: str = None):
    """Write a Delta table with optional partitioning and optimize."""
    writer = df.write.format("delta").mode("overwrite")
    if partition_col:
        writer = writer.partitionBy(partition_col)
    writer.saveAsTable(table_name)
    logger.info(f"Written table {table_name} successfully.")
    # Optimize and ZORDER for query speed if partitioned
    if partition_col:
        spark.sql(f"OPTIMIZE {table_name} ZORDER BY ({partition_col})")
        logger.info(f"Optimized table {table_name} with ZORDER on {partition_col}.")

def validate_data(df: DataFrame, rules: list):
    """Apply data quality rules and return if data passes all checks."""
    for rule in rules:
        count_invalid = df.exceptAll(rule["rule"](df)).count()
        if count_invalid > 0:
            logger.error(f"Data quality check failed: {rule['description']} ({count_invalid} invalid rows)")
            raise ValueError(f"Data quality violation: {rule['description']}")
    logger.info("All data quality checks passed.")
    return True

def bronze_layer(raw_table: str, bronze_table: str) -> DataFrame:
    logger.info("Starting Bronze layer ingestion...")
    df_raw = spark.table(raw_table)

    df_bronze = (df_raw
        .withColumn("ingestion_timestamp", current_timestamp())
        .withColumn("source_table", lit(raw_table))
    )
    df_bronze_cleaned = clean_column_names(df_bronze)
    write_delta_table(df_bronze_cleaned, bronze_table)
    logger.info("Bronze layer ingestion complete.")
    return df_bronze_cleaned

def silver_layer(bronze_table: str, silver_table: str) -> DataFrame:
    logger.info("Starting Silver layer processing...")
    df_bronze = spark.table(bronze_table)
    df_silver = df_bronze.dropDuplicates()
    df_silver = df_silver.na.fill({'median_salary_usd': 0})

    # Flag validity
    df_silver = df_silver.withColumn(
        "is_valid",
        when(col("median_salary_usd") > 0, True).otherwise(False)
    )

    validate_data(df_silver, DATA_QUALITY_RULES)
    write_delta_table(df_silver, silver_table)
    logger.info("Silver layer processing complete.")
    return df_silver

def gold_layer(silver_table: str, gold_table: str, partition_col: str) -> DataFrame:
    logger.info("Starting Gold layer aggregation...")
    df_silver = spark.table(silver_table)

    df_gold = (
        df_silver.groupBy("industry", "job_status")
        .agg(
            round(avg("median_salary_usd"), 2).alias("avg_median_salary"),
            count("*").alias("job_count")
        )
    )

    df_gold = df_gold.withColumn("created_at", current_timestamp()) \
        .withColumn("source_table", lit(silver_table))

    write_delta_table(df_gold, gold_table, partition_col=partition_col)
    logger.info("Gold layer aggregation complete.")
    return df_gold

if __name__ == "__main__":
    try:
        bronze_df = bronze_layer(RAW_TABLE, BRONZE_TABLE)
        silver_df = silver_layer(BRONZE_TABLE, SILVER_TABLE)
        gold_df = gold_layer(SILVER_TABLE, GOLD_TABLE, PARTITION_COL)
        display(gold_df)
        logger.info("Data pipeline executed successfully!")
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        # Integrate alerting here if needed (email, Slack, etc.)
        raise
