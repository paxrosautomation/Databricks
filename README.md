import re
from pyspark.sql.functions import current_timestamp, lit, when, avg, count, round, current_timestamp as ct

# ---------------- Bronze Layer ----------------

# Step 1: Load raw table
df_raw = spark.table('ai_job_trends_dataset')

# Step 2: Add ingestion metadata
df_bronze = (df_raw
    .withColumn("ingestion_timestamp", current_timestamp())
    .withColumn("source_table", lit("ai_job_trends_dataset"))
)

# Step 3: Define a function to clean column names
def clean_column_names(df):
    def clean(name):
        name = name.lower()
        # Replace any character not a-z, 0-9, underscore with _
        name = re.sub(r"[^a-z0-9_]", "_", name)
        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name)
        # Remove leading/trailing underscores
        name = name.strip("_")
        return name
    new_cols = [clean(c) for c in df.columns]
    return df.toDF(*new_cols)

# Step 4: Clean columns
df_bronze_cleaned = clean_column_names(df_bronze)

# Step 5: Write the cleaned DataFrame as a Delta table
df_bronze_cleaned.write.format("delta").mode("overwrite").saveAsTable("ai_jobs_data_bronze")

# ---------------- Silver Layer ----------------

# Load Bronze table
df_bronze = spark.table("ai_jobs_data_bronze")

# Remove duplicates
df_silver = df_bronze.dropDuplicates()

# Fill missing values or filter
df_silver = df_silver.na.fill({'median_salary_usd': 0})

# Flag rows with errors
df_silver = df_silver.withColumn(
    "is_valid",
    when(df_silver["median_salary_usd"] > 0, True).otherwise(False)
)

# Write Silver layer
df_silver.write.format("delta").mode("overwrite").saveAsTable("ai_jobs_data_silver")

# ---------------- Gold Layer ----------------

# Load Silver table
df_silver = spark.table("ai_jobs_data_silver")

# Aggregate average median salary and job count by industry and job status
df_gold = (
    df_silver.groupBy("industry", "job_status")
    .agg(
        round(avg("median_salary_usd"), 2).alias("avg_median_salary"),
        count("*").alias("job_count")
    )
)

# Add metadata columns
df_gold = df_gold.withColumn("created_at", current_timestamp()) \
    .withColumn("source_table", lit("ai_jobs_data_silver"))

# Write Gold layer Delta table
df_gold.write.format("delta").mode("overwrite").saveAsTable("ai_jobs_data_gold")

# Display Gold layer data
display(df_gold)
