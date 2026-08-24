"""
Component A, step 1: build your raw corpus.

Downloads the public University of Hull scholarship/finance pages and PDFs
identified during exploration, saving them to ./corpus/raw/ ready for
ingestion into Dify (or a custom chunker if you extend beyond Dify's default
chunking for your dissertation's methodology section).

Extend SOURCES as you crawl https://www.hull.ac.uk/policies-and-information
for the additional scholarship policy PDFs linked from that index page.
"""

import os
import time
import requests

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "raw")

SOURCES = {
    "ug_scholarships_awards.html":
        "https://www.hull.ac.uk/study/undergraduate/fees-and-funding/scholarships-and-awards",
    "pgt_scholarships_awards.html":
        "https://www.hull.ac.uk/study/postgraduate-taught/fees-and-funding/scholarships-and-awards",
    "pgr_fees_funding.html":
        "https://www.hull.ac.uk/study/postgraduate-research/fees-and-funding",
    "chancellors_scholarship_policy.pdf":
        "https://www.hull.ac.uk/asset-library/docs/chancellors-scholarship-policy.pdf",
    "fees_and_finance_hub.html":
        "https://www.hull.ac.uk/choose-hull/study-at-hull/money/index.aspx",
    "finance_information_and_policies.html":
        "https://www.hull.ac.uk/choose-hull/study-at-hull/money/information-and-policies",
    "financial_advice_and_support.html":
        "https://www.hull.ac.uk/life-at-hull/student-wellbeing-and-inclusion/financial-support",
    "fees_funding_faqs.html":
        "https://www.hull.ac.uk/help-centre/fees-and-funding",
    "policies_and_information_index.html":
        "https://www.hull.ac.uk/policies-and-information",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (dissertation research crawler)"}

    for filename, url in SOURCES.items():
        dest = os.path.join(OUT_DIR, filename)
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            print(f"OK  {filename}  ({len(resp.content)} bytes)")
        except requests.RequestException as e:
            print(f"FAIL {filename}: {e}")
        time.sleep(1)  # be polite to the university's server


if __name__ == "__main__":
    main()
