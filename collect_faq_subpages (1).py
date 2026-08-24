"""
Fix for the fees_funding_faqs.html gap discovered during evaluation.

Root cause (corrected from the original "JavaScript accordion" hypothesis):
each FAQ question on hull.ac.uk actually has its OWN dedicated URL with real,
statically-scrapable answer content (e.g.
/help-centre/fees-and-funding/how-do-i-apply-for-student-finance). The
original collect_corpus.py only scraped the FAQ INDEX page, which lists
question titles as links but not their answers - it never followed those
links to the individual answer pages. This is a straightforward scraper bug,
not a fundamental JS-rendering limitation.

This script:
  1. Fetches the FAQ index page(s)
  2. Extracts every <a href> pointing to an individual /help-centre/... page
  3. Fetches each of those pages and extracts its real answer text
  4. Saves them as additional corpus documents

Run this BEFORE chunk_corpus.py so the new pages get chunked and indexed
alongside the rest of the corpus.
"""

import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "raw")

# Index pages known to list individual FAQ sub-pages as links
FAQ_INDEX_PAGES = [
    "https://www.hull.ac.uk/help-centre/fees-and-funding",
    "https://www.hull.ac.uk/help-centre/applying-to-hull",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (dissertation research crawler)"}


def discover_faq_links(index_url: str) -> list:
    """Find every link on the index page that points to an individual
    /help-centre/... answer page (not external links, not the index itself)."""
    resp = requests.get(index_url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(index_url, href)
        parsed = urlparse(full_url)
        # Keep only same-domain help-centre sub-pages, not the index page itself
        if (parsed.netloc == "www.hull.ac.uk"
                and "/help-centre/" in parsed.path
                and full_url.rstrip("/") != index_url.rstrip("/")):
            links.add(full_url)
    return sorted(links)


def slug_to_filename(url: str) -> str:
    slug = urlparse(url).path.strip("/").split("/")[-1]
    return f"faq_{slug}.html"


def fetch_and_save(url: str) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"FAIL {url}: {e}")
        return False

    filename = slug_to_filename(url)
    dest = os.path.join(OUT_DIR, filename)
    with open(dest, "wb") as f:
        f.write(resp.content)
    print(f"OK   {filename}  ({len(resp.content)} bytes)  <- {url}")
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_links = set()

    for index_url in FAQ_INDEX_PAGES:
        print(f"Discovering links from {index_url} ...")
        try:
            links = discover_faq_links(index_url)
            print(f"  found {len(links)} candidate sub-pages")
            all_links.update(links)
        except requests.RequestException as e:
            print(f"  FAILED to fetch index: {e}")
        time.sleep(1)

    print(f"\nTotal unique FAQ sub-pages discovered: {len(all_links)}")
    print("Fetching each one...\n")

    success_count = 0
    for url in sorted(all_links):
        if fetch_and_save(url):
            success_count += 1
        time.sleep(0.5)  # be polite to the server

    print(f"\nDone: {success_count}/{len(all_links)} FAQ pages saved successfully.")
    print("Next steps:")
    print("  1. Re-run chunk_corpus.py to include these new pages")
    print("  2. Re-run build_index.py to re-embed the expanded corpus")
    print("  3. Update qa_dataset_template.csv gold_answers for questions")
    print("     that are now genuinely answerable")


if __name__ == "__main__":
    main()
