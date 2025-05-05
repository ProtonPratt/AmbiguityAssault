import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
import random # For slightly randomized delays

# List of stock tickers
tickers = ['AAPL', 'TSLA', 'AMZN', 'NKE', 'NVDA']

# --- Selectors (THESE MIGHT NEED UPDATING IF YAHOO CHANGES ITS HTML) ---
# These are based on inspection at the time of writing. Use browser dev tools (F12)
# on the Yahoo Finance ticker page (e.g., https://finance.yahoo.com/quote/AAPL)
# to verify or find new selectors if this breaks.

# Common container for news items (might vary)
# Look for <ul> or <div> elements containing the list
NEWS_CONTAINER_SELECTOR = 'ul.stream' # Often a UL with class 'stream' or similar
# Individual news item selector (within the container)
NEWS_ITEM_SELECTOR = 'li.js-stream-contentPos' # Often list items with this class
# Title/Link selector within a news item
TITLE_LINK_SELECTOR = 'h3 a' # Usually an <a> tag inside an <h3>
# Source/Publisher selector within a news item
SOURCE_SELECTOR = 'div > span' # Often a <span> inside a <div> sibling to the title area
# Time selector within a news item
TIME_SELECTOR = 'div > span' # Can be the same generic selector as source, check position/content

# --- End Selectors ---


# List to store all formatted news articles
scraped_news_list = []

# Use a session object for potential cookie handling and connection pooling
session = requests.Session()

# Set a realistic User-Agent header
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
}
session.headers.update(headers)


# Loop through each ticker
for ticker_symbol in tickers:
    print(f"Scraping news for {ticker_symbol}...")
    url = f"https://finance.yahoo.com/quote/{ticker_symbol}?p={ticker_symbol}" # Standard ticker page URL

    try:
        response = session.get(url, timeout=15) # Timeout in seconds
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser') # Or use 'lxml' if installed

        # Find the main news stream container
        news_container = soup.select_one(NEWS_CONTAINER_SELECTOR)

        if not news_container:
            print(f" -> Could not find the main news container ({NEWS_CONTAINER_SELECTOR}) for {ticker_symbol}. Skipping.")
            time.sleep(random.uniform(1.5, 3.5)) # Wait even if failed
            continue

        # Find all individual news items within the container
        news_items = news_container.select(NEWS_ITEM_SELECTOR)
        print(f" -> Found {len(news_items)} potential news items for {ticker_symbol}.")

        if not news_items:
             print(f" -> No news items found using selector '{NEWS_ITEM_SELECTOR}'. Check selectors or page structure.")

        # Process each news item
        items_processed = 0
        for item in news_items:
            title = "N/A"
            summary = "N/A" # Often same as title from list view
            published_info = "N/A" # Will store the time string found
            link = "N/A"
            source = "N/A"

            try:
                # Find Title and Link
                title_link_tag = item.select_one(TITLE_LINK_SELECTOR)
                if title_link_tag:
                    title = title_link_tag.get_text(strip=True)
                    summary = title # Use title as summary
                    link = title_link_tag.get('href', 'N/A')
                    # Often links are relative, prepend base URL if needed
                    if link.startswith('/'):
                       link = f"https://finance.yahoo.com{link}"

                # Find Source and Time (These can be tricky)
                # Often source and time are in spans within a div near the title
                meta_div = item.find('div', class_='stream-item-footer') # Common footer div
                if meta_div:
                    spans = meta_div.find_all('span', recursive=False) # Get direct children spans
                    if len(spans) >= 2:
                        source = spans[0].get_text(strip=True)
                        published_info = spans[1].get_text(strip=True) # This is likely relative ("2 hours ago")
                    elif len(spans) == 1:
                         # Sometimes only one span is present, might be source or time
                         possible_info = spans[0].get_text(strip=True)
                         # Basic check if it looks like a source vs time (very heuristic)
                         if any(kw in possible_info.lower() for kw in ['ago', 'yesterday', 'min', 'hour', ':', 'am', 'pm']) or possible_info.isdigit():
                              published_info = possible_info
                         else:
                              source = possible_info

                # --- Basic Validation: Only add if a title was found ---
                if title != "N/A":
                    formatted_article = {
                        "published_date": published_info, # NOTE: This is often relative time!
                        "title": title,
                        "summary": summary,
                        "ticker": ticker_symbol,
                        "source": source,
                        "link": link
                    }
                    scraped_news_list.append(formatted_article)
                    items_processed += 1

            except Exception as item_err:
                print(f"   - Error processing one item for {ticker_symbol}: {item_err}")
                # Continue to the next item

        print(f" -> Successfully processed {items_processed} items for {ticker_symbol}.")

    except requests.exceptions.RequestException as e:
        print(f"!! Could not fetch page for {ticker_symbol}: {e}")
    except Exception as e:
        print(f"!! An unexpected error occurred for {ticker_symbol}: {e}")

    # --- Polite Delay ---
    wait_time = random.uniform(2.0, 5.0) # Wait between 2 and 5 seconds
    print(f"   Waiting {wait_time:.1f} seconds...")
    time.sleep(wait_time)


# Define the output JSON filename
output_filename = 'stock_news_scraped.json'

# Save the collected & formatted news data to a JSON file
print(f"\nSaving {len(scraped_news_list)} scraped news articles to {output_filename}...")
try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(scraped_news_list, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved news data to {output_filename}")
except Exception as e:
    print(f"!! Error saving data to JSON file: {e}")

print("\nScraping process finished.")