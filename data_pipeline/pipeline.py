import pandas as pd


# --------------------------------------------------
# 1. LOAD RAW DATA
# --------------------------------------------------

INPUT_FILE = "data_pipeline/raw_books.csv"
OUTPUT_FILE = "data_pipeline/clean_books.csv"

# Required fixed conversion rate
GBP_TO_INR = 105.50

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 3. CLEAN PRICE
# --------------------------------------------------

# Remove the £ symbol and convert to float
df["price_gbp"] = (
    df["price"]
    .str.replace("£", "", regex=False)
    .astype(float)
)


# --------------------------------------------------
# 4. CLEAN RATING
# --------------------------------------------------

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["rating"].map(rating_map)


# --------------------------------------------------
# 5. CLEAN AVAILABILITY
# --------------------------------------------------

df["in_stock"] = (
    df["availability"]
    .str.contains("In stock", case=False, na=False)
)


# --------------------------------------------------
# 6. CONVERT GBP TO INR
# --------------------------------------------------

df["price_inr"] = df["price_gbp"] * GBP_TO_INR

# The source website uses "Add a comment" as a category
# for five books. It is not a meaningful book category,
# so we mark it as Unknown in the cleaned dataset.
df["category"] = df["category"].replace(
    "Add a comment",
    "Unknown"
)

df = df.drop(
    columns=["price", "availability"]
)


# --------------------------------------------------
# 8. CHECK INVALID VALUES
# --------------------------------------------------

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nRating values:")
print(df["rating"].unique())

print("\nIn-stock values:")
print(df["in_stock"].unique())


# --------------------------------------------------
# 9. CHECK CATEGORIES
# --------------------------------------------------

print("\nCategories:")
print(sorted(df["category"].unique()))


# --------------------------------------------------
# 10. DATA TYPES
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 11. SAVE CLEAN DATA
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nClean data saved to:")
print(OUTPUT_FILE)

print("\nFinal shape:", df.shape)

print("\nFirst 5 cleaned rows:")
print(df.head())