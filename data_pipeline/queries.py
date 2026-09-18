import sqlite3
import pandas as pd

DATABASE_FILE = "data_pipeline/books.db"
OUTPUT_FILE = "data_pipeline/query_results.txt"

connection = sqlite3.connect(DATABASE_FILE)
output = open(OUTPUT_FILE, "w", encoding="utf-8")

query1 = """
SELECT title, rating, price_gbp
FROM books
WHERE rating = 5;
"""

result1 = pd.read_sql(query1, connection)

print("\n========== QUERY 1 ==========")
print(query1)
print(result1)

output.write("\n========== QUERY 1 ==========\n")
output.write(query1)
output.write("\n")
output.write(result1.to_string(index=False))
output.write("\n")

query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
"""

result2 = pd.read_sql(query2, connection)

print("\n========== QUERY 2 ==========")
print(query2)
print(result2)

output.write("\n========== QUERY 2 ==========\n")
output.write(query2)
output.write("\n")
output.write(result2.to_string(index=False))
output.write("\n")

query3 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
"""

result3 = pd.read_sql(query3, connection)

print("\n========== QUERY 3 ==========")
print(query3)
print(result3)

output.write("\n========== QUERY 3 ==========\n")
output.write(query3)
output.write("\n")
output.write(result3.to_string(index=False))
output.write("\n")

query4 = """
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 40 AND 50
ORDER BY price_gbp;
"""

result4 = pd.read_sql(query4, connection)

print("\n========== QUERY 4 ==========")
print(query4)
print(result4)

output.write("\n========== QUERY 4 ==========\n")
output.write(query4)
output.write("\n")
output.write(result4.to_string(index=False))
output.write("\n")

query5 = """
SELECT
    b.title,
    b.price_gbp,
    b.rating,
    c.category_name
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id;
"""

result5 = pd.read_sql(query5, connection)

print("\n========== QUERY 5 ==========")
print(query5)
print(result5)

output.write("\n========== QUERY 5 ==========\n")
output.write(query5)
output.write("\n")
output.write(result5.to_string(index=False))
output.write("\n")

books_df = pd.read_sql(
    "SELECT * FROM books",
    connection
)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    connection
)

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

merged_df = merged_df[
    ["title", "price_gbp", "rating", "category_name"]
]

sql_join_sorted = result5.sort_values(
    by=["title", "price_gbp"]
).reset_index(drop=True)

pandas_join_sorted = merged_df.sort_values(
    by=["title", "price_gbp"]
).reset_index(drop=True)

equivalent = sql_join_sorted.equals(
    pandas_join_sorted
)

print("\n========== PANDAS MERGE ==========")
print(merged_df)

print("\nSQL JOIN and Pandas merge equivalent:")
print(equivalent)

output.write("\n========== PANDAS MERGE ==========\n")
output.write(merged_df.to_string(index=False))
output.write("\n")

output.write(
    "\nSQL JOIN and Pandas merge equivalent: "
    + str(equivalent)
    + "\n"
)

output.close()
connection.close()

print("\nQuery results saved to:")
print(OUTPUT_FILE)

print("\nDatabase connection closed.")