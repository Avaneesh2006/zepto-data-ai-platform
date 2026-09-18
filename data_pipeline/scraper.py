import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd

BASE_URL = "https://books.toscrape.com/"


def scrape_book(book, page_url):
    """
    Extract book details from a book listing.
    """

    title = book.h3.a["title"]

    price = book.select_one(".price_color").get_text(strip=True)

    rating = book.select_one(".star-rating")["class"][1]

    availability = book.select_one(".availability").get_text(
        " ", strip=True
    )

    book_url = urljoin(page_url, book.h3.a["href"])

    # Open the individual book page to get its category
    book_response = requests.get(book_url, timeout=10)
    book_response.raise_for_status()
    book_response.encoding = book_response.apparent_encoding

    book_soup = BeautifulSoup(
        book_response.text,
        "html.parser"
    )

    # Extract category from the product page
    category_element = book_soup.select_one(
        "ul.breadcrumb li:nth-of-type(3)"
    )

    if category_element:
        category = category_element.get_text(strip=True)
    else:
        category = "Unknown"

    return {
        "title": title,
        "price": price,
        "rating": rating,
        "availability": availability,
        "category": category,
        "book_url": book_url
    }


def scrape_page(page_number):
    """
    Scrape all books from one catalogue page.
    """

    if page_number == 1:
        page_url = BASE_URL
    else:
        page_url = urljoin(
            BASE_URL,
            f"catalogue/page-{page_number}.html"
        )

    response = requests.get(page_url, timeout=10)
    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    books = soup.select("article.product_pod")

    page_books = []

    for book in books:
        book_data = scrape_book(book, page_url)
        page_books.append(book_data)

    return page_books


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

all_books = []

for page_number in range(1, 6):

    print(f"Scraping page {page_number}...")

    page_books = scrape_page(page_number)

    all_books.extend(page_books)

    print(
        f"Books collected so far: {len(all_books)}"
    )


# --------------------------------------------------
# SAVE RAW DATA
# --------------------------------------------------

df = pd.DataFrame(all_books)

df.to_csv(
    "data_pipeline/raw_books.csv",
    index=False,
    encoding="utf-8"
)

print("\nScraping completed!")
print("Total books:", len(df))

print("\nRaw data saved to:")
print("data_pipeline/raw_books.csv")

print("\nFirst book:")
print(df.iloc[0])

print("\nCategories found:")
print(df["category"].unique())