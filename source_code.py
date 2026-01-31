from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
import os, sys

os.environ["JAVA_HOME"] = "C:\\Users\\PC\\.jdks\\jdk-17.0.2"

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("billupsTest") \
    .config("spark.driver.memory", "4g")    \
    .config("spark.executor.memory", "4g")  \
    .config("spark.sql.shuffle.partitions", "16")       \
    .config("spark.sql.warehouse.dir", "file:///C:/tmp/spark-warehouse") \
    .config("spark.hadoop.validateOutputSpecs", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df_facts = (spark.read
                .parquet("C:\\Users\\PC\\Desktop\\JOB\\billups_test_project\\data\\part-00000-tid-860771939793626614-979f966a-6d53-4896-9692-f81194d27b99-109986-1-c000.snappy.parquet")
                .withColumn("year_month", F.date_trunc("month", F.to_timestamp("purchase_date")))
                .withColumn("hour", F.date_format(F.col("purchase_date"), "HH00"))
                .withColumn("category", F.coalesce(F.col("category"), F.lit("Unknown category")))
                )

df_merc = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv("C:\\Users\\PC\\Desktop\\JOB\\billups_test_project\\data\\merchants-subset.csv")
            )

df_facts.cache()






#=====================================================================================================================================
# Question 1: Generate the top 5 merchants by purchase_amount for each month in the dataset,
# for each city in the dataset.

# aggregate data first to reduce amount of rows. also spark version >= 4 allows to use aliases in groupBy and change column names
df_q1 = (
    df_facts.groupBy(
            F.col("year_month").alias("Month"),
            F.col("city_id").alias("City"),
            F.col("merchant_id").alias("merchant_id")
            )
        .agg(
                F.round(F.sum("purchase_amount"), 3).cast("decimal(15,3)").alias("Purchase_total"),
                F.count("*").alias("No_of_sales")
            )
)


window_spec = Window.partitionBy("Month", "City").orderBy(F.desc("Purchase_total"))

df_q1 = (
    df_q1.join(df_merc, on="merchant_id", how="left")
    .withColumn("rnk", F.row_number().over(window_spec))
    .filter(F.col("rnk") <= 5)
    .orderBy(F.col("Month"), F.col("City"), F.col("rnk"))
    .drop(F.col("rnk"))
    .select(
        F.date_format(F.col("Month"), "yyyy MMM").alias("Month"),
        F.col("City"),
        F.coalesce(F.col("merchant_name"), F.col("merchant_id")).alias("Merchant"),
        F.col("Purchase_total"),
        F.col("No_of_sales"),
    )
    
)

print("=" * 80)
print("Question 1: Generate the top 5 merchants by purchase_amount for each month in the dataset")
print("for each city in the dataset.")

df_q1.show(n=100, truncate=False)

print("=" * 80)

#=====================================================================================================================================
# Question 2: What is the average sale amount (purchase_amount) of each merchant in each
# state. Consider returning the merchants with the largest sales first:

df_q2 = (
    df_facts.groupBy(
        F.col("merchant_id"),
        F.col("state_id").alias("State_id_facts")
    )
    .agg(
        F.round(F.avg("purchase_amount"), 3).cast("decimal(15,3)").alias("Average_amount"),
    )
)

df_q2 = (
    df_q2.join(df_merc, on="merchant_id", how="left")
    .select(
        F.coalesce(F.col("merchant_name"), F.col("merchant_id")).alias("Merchant"),
        F.col("State_id_facts").alias("State_id"),
        F.col("Average_amount")
    )
    .orderBy(F.desc(F.col("Average_amount")))
)

print("=" * 80)
print("Question 2: What is the average sale amount (purchase_amount) of each merchant in each")
print("state. Consider returning the merchants with the largest sales first")

df_q2.show(n=100, truncate=False)

print("=" * 80)

#=====================================================================================================================================
# Question 3: Identify the top 3 hours where the largest amount of sales (purchase_amount) are
# recorded for each product category (category).

df_q3 = (
    df_facts.groupBy(
        F.col("category").alias("Category"),
        F.col("hour").alias("Hour")
    )
    .agg(
        F.round(F.sum("purchase_amount"), 3).cast("decimal(15,3)").alias("purchase_amount")
    )
)

window_cpec = Window.partitionBy("Category").orderBy(F.desc("purchase_amount"))

df_q3 = (
    df_q3.withColumn("rnk", F.row_number().over(window_cpec))
    .filter(F.col("rnk") <= 3)
    .drop(F.col("rnk"), F.col("purchase_amount"))
)

print("=" * 80)
print("Question 3: Identify the top 3 hours where the largest amount of sales (purchase_amount) are")
print("recorded for each product category (category)")

df_q3.show(n=100, truncate=False)

print("=" * 80)

#=====================================================================================================================================
# Question 4: In which cities are the most popular merchants located. Is there a correlation
# between the location (city_id) and the categories (category) the merchant sells.
# Note: Consider popularity in terms of the number of sales transactions of each merchant

# 4.1. In which cities are the most popular merchants located
df_q4_1 = (
   df_facts.groupBy(
       F.col("city_id")
   )
   .agg(
       F.count("*").alias("nun_of_sales")
   )
   .orderBy(F.desc(F.col("nun_of_sales")))
   .drop(F.col("nun_of_sales"))
   .limit(10) #I take top 10 popular cities since it's not defined in the task
)
print("=" * 80)
print("Question 4: In which cities are the most popular merchants located. Is there a correlation")
print("between the location (city_id) and the categories (category) the merchant sells.")
print("Note: Consider popularity in terms of the number of sales transactions of each merchant")

print("=" * 80)
print("Question 4.1: In which cities are the most popular merchants located?")
df_q4_1.show(n=100, truncate=False)

# 4.2 Is there a correlation between the location (city_id) and the categories (category) the merchant sells
df_q4_2 = (
   df_facts.groupBy(
       F.col("city_id"),
       F.col("category")
   )
   .agg(
       F.count("*").alias("nun_of_sales"),
       F.round(F.sum("purchase_amount"), 3).cast("decimal(15,3)").alias("purchase")
   )
   .orderBy(F.desc(F.col("purchase")))
)
pivot_df = df_q4_2.groupBy("city_id").pivot("category").sum("purchase").fillna(0)

print("=" * 80)
print("Question 4.2 Is there a correlation between the location (city_id) and the categories (category) the merchant sells?")
pivot_df.orderBy(F.desc("A")).show(n=10000, truncate=False)
print("=" * 80)

#=====================================================================================================================================
# Question 5: A new merchant is coming in to do businessand you have been assigned to give
# advice based strictly on the historical transactions. You are expected to provide a response to
# the following questions.
# Note: Remember to state your assumptions if any.
# a. Which cities would you advise them to focus on and why?
# b. Which categories would you recommend they sell
# c. Are there particular periods (months) that have interesting sales behaviors?
# d. What hours would you recommend they open and close for the day?
# e. Would you recommend accepting payments in installments? Assume a credit default
#     rate of 22.9% per month.
#     For this question, consider the “installments” header in the historical transactions and the
#     impact it may have, if any, on merchant sales (merchant sales in terms of
#     purchase_amounts). We are making a simplistic assumption that 25% of sales is gross
#     profit to merchants, there are equal installments and everyone who defaulted did so after
#     making half payment

print("=" * 80)
print("""
    Question 5: A new merchant is coming in to do businessand you have been assigned to give
    advice based strictly on the historical transactions. You are expected to provide a response to
    the following questions.
    Note: Remember to state your assumptions if any.
    a. Which cities would you advise them to focus on and why?
    b. Which categories would you recommend they sell
    c. Are there particular periods (months) that have interesting sales behaviors?
    d. What hours would you recommend they open and close for the day?
    e. Would you recommend accepting payments in installments? Assume a credit default
        rate of 22.9% per month.
        For this question, consider the “installments” header in the historical transactions and the
        impact it may have, if any, on merchant sales (merchant sales in terms of
        purchase_amounts). We are making a simplistic assumption that 25% of sales is gross
        profit to merchants, there are equal installments and everyone who defaulted did so after
        making half payment
""")
print("=" * 80)
print("question 5a. Which cities would you advise them to focus on and why?")
print("""
    Based on the pivot table of sales amounts and transaction volumes from point 4.2, merchant should prioritize the following cities due to their high historical market capacity:
    -   City 69 (Primary Focus): This is the largest market by a significant margin. It has generated over 13 billion in Category A sales and nearly 10 billion in Category B sales.
    -   City 19 & City 158 (Secondary Focus): These cities represent the next tier of high-volume locations, with combined sales across major categories exceeding 3 billion and 2 billion respectively.
    -   City 1 (Strategic Opportunity): While City 69 leads in Category A, City 1 is a massive hub for Category B, with historical sales exceeding 9 billion. It also shows the highest historical volume for Category C (over 2.3 billion).
      """)

print("=" * 80)
print("question 5b. Which categories would you recommend they sell?")
print("""
    Category A (Highest Recommendation): This is the most dominant category across the entire dataset.
        -   Popularity: The top 6 highest transaction records in the dataset all belong to Category A (ranging from 4,901 to 5,078 transactions per merchant).
        -   Profitability: It consistently generates the highest sales amounts in almost every major city.
    Category B (Strong Alternative): This category serves as a robust second choice, particularly in City 1 where it actually outperforms Category A. High-performing merchants in this category still achieve nearly 4,800 transactions.
    Category C (Niche/Specific Locations): Only recommend this if focusing on specific hubs like City 1 or City 160, as its overall transaction volume and sales are generally lower than A and B.
    Recommendation: To maximize potential, the merchant should enter City 69 selling Category A products. If they prefer a market where Category B is dominant, they should focus exclusively on City 1.
      """)

print("=" * 80)

print("Question 5c  Are there particular periods (months) that have interesting sales behaviors?")
df_q5c = (
    df_facts.groupBy(
            F.col("year_month").alias("Month"),
            )
        .agg(
                F.round(F.sum("purchase_amount"), 3).cast("decimal(15,3)").alias("Purchase_total"),
                F.count("*").alias("No_of_sales")
            )
        .orderBy(F.col("Month"))
        .select(
            F.col("Month").alias("Month"),
            F.col("Purchase_total"),
            F.col("No_of_sales")
        )
)

window_spec = Window.orderBy("Month")

analysis_df = (
    df_q5c.withColumn("Avg_Transaction_Value", F.col("Purchase_total") / F.col("No_of_sales"))
    .withColumn("Previous_Month_Sales", F.lag("Purchase_total", 1).over(window_spec))
    .withColumn("MoM_Growth_Pct", 
                ((F.col("Purchase_total") - F.col("Previous_Month_Sales")) / F.col("Previous_Month_Sales")) * 100)
    .select(
        F.date_format(F.col("Month"), "yyyy MMM").alias("Month"),
        F.col("Purchase_total"),
        F.col("No_of_sales"),
        F.col("Avg_Transaction_Value"),
        F.col("Previous_Month_Sales"),
        F.col("MoM_Growth_Pct")
    )
)

analysis_df.show(n=10000, truncate=False)


print("""
    1. The "Holiday Peak" (December 2017)
        December 2017 is the most significant outlier in the dataset.
        Sales Volume: It reached a peak of $17.13 Billion, which is a 20.7% increase over November 2017.
        Transaction Volume: The number of sales jumped to 852,200, the highest in the entire period.
        Behavior: This suggests a massive end-of-year or holiday shopping surge.
    2. Massive Year-over-Year Growth
        Comparing the start of 2017 to the start of 2018 shows that the business expanded exponentially:
        January Comparison: Sales in January 2018 ($14.34B) were 149% higher than in January 2017 ($5.75B).
        February Comparison: Sales in February 2018 ($12.19B) were more than double (101% increase) the sales of February 2017 ($6.05B).
        Conclusion: While there is a seasonal dip after December, the "floor" for sales in 2018 is significantly higher than it was in 2017, indicating strong business scaling.
    3. Consistency in Average Transaction Value
        Despite the massive fluctuations in total sales and transaction counts, the Average Sale Value remains remarkably stable:
        Jan 2017: ~$20,089 per sale.
        Dec 2017: ~$20,101 per sale.
      
        Jan 2018: ~$20,093 per sale.
        That means that the growth is being driven entirely by an increase in the number of customers/transactions, not by selling more expensive items or increasing prices.
    4. The Post-Holiday "Correction" (Jan - Feb 2018)
        After the December peak, there is a clear downward trend entering the new year.
        Sales dropped from $17.13B (Dec) to $14.34B (Jan) to $12.19B (Feb).
        However, because February 2018 is still higher than any month in 2017 except for the holiday quarter (Nov/Dec), the business remains in a very healthy position despite the month-over-month decline.
      """)
print("=" * 80)
print("Question 5d What hours would you recommend they open and close for the day?")

df_q5d = (
    df_facts.groupBy(
        F.col("hour").alias("Hour")
    )
    .agg(
        F.round(F.sum("purchase_amount"), 3).cast("decimal(15,3)").alias("purchase_amount")
    )
    .orderBy(
        F.desc(F.col("purchase_amount"))
    )
)

df_q5d.show(n=25, truncate=False)

print("I would recommend to work form 11AM till 20PM if it's 9 hours working day")
print("=" * 80)
print("Question 5e Would you recommend accepting payments in installments?")

installment_behavior = df_facts.groupBy("installments") \
    .agg(
        F.avg("purchase_amount").alias("avg_spend"),
        F.sum("purchase_amount").alias("total_volume")
    ).orderBy("installments")

installment_behavior.show()

default_rate = 0.229
profit_margin = 0.25
recovery_on_default = 0.50 # Paid half, lost half

financial_analysis = installment_behavior.withColumn(
    "Gross_Profit", F.col("total_volume") * profit_margin
).withColumn(
    "Expected_Loss", 
    F.when(F.col("installments") > 1, (F.col("total_volume") * default_rate * recovery_on_default))
    .otherwise(0)
).withColumn(
    "Net_Profit", F.col("Gross_Profit") - F.col("Expected_Loss")
)

financial_analysis.select(
    "installments", 
    "total_volume", 
    "Gross_Profit", 
    "Expected_Loss", 
    "Net_Profit"
).show()

print("""
        Accept payments in installments, with limits.
        A controlled installment offering (especially 2–6 payments) is economically justified and advisable.
      """)
print("=" * 80)

df_facts.unpersist()
spark.stop()