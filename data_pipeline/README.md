# Module 1 — Data Pipeline

## Overview

This module implements a complete data pipeline:

Scrape → Clean → Convert → Store → Query

The data source is Books to Scrape:

https://books.toscrape.com/

The first 5 catalogue pages were scraped, producing 100 book records.

## Files

- `scraper.py` — scrapes book data using Requests and BeautifulSoup
- `raw_books.csv` — raw scraped data
- `pipeline.py` — cleans and converts the scraped data
- `clean_books.csv` — cleaned dataset
- `database.py` — creates and populates the SQLite database
- `books.db` — SQLite database
- `queries.py` — executes SQL queries and Pandas JOIN verification
- `query_results.txt` — saved SQL queries and outputs

## Installation

Activate the virtual environment and install:

```bash
python -m pip install requests beautifulsoup4 pandas


## Validation

The completed pipeline was validated with:

- 100 books stored in the SQLite database
- 29 unique categories
- Five SQL queries executed successfully
- SQL JOIN and Pandas `merge()` produced equivalent results
- SQL JOIN and Pandas merge equivalence check returned `True`