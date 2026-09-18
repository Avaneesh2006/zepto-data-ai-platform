import sqlite3
import pandas as pd

INPUT_FILE = "data_pipeline/clean_books.csv"
DATABASE_FILE = "data_pipeline/books.db"

df = pd.read_csv(INPUT_FILE)

print("Clean data loaded:", df.shape)

connection = sqlite3.connect(DATABASE_FILE)
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    category_id INTEGER,
    book_url TEXT,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")

categories = df["category"].drop_duplicates()

for category in categories:
    cursor.execute(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )

for _, row in df.iterrows():
    cursor.execute(
        """
        SELECT category_id
        FROM categories
        WHERE category_name = ?
        """,
        (row["category"],)
    )

    category_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO books
        (title, price_gbp, price_inr, rating,
         in_stock, category_id, book_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            row["in_stock"],
            category_id,
            row["book_url"]
        )
    )

connection.commit()

print("\nDatabase created successfully!")

print("\nTables:")
tables = cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type = 'table'
""").fetchall()

for table in tables:
    print("-", table[0])

print("\nNumber of categories:")
print(
    cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]
)

print("\nNumber of books:")
print(
    cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]
)

connection.close()

print("\nDatabase connection closed.")