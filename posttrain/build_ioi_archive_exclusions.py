"""Hash public historical IOI statements without retaining statement text."""

import concurrent.futures
import hashlib
import json
import re
import subprocess
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

from build_pool import ROOT, shingles, title_key, write_json

BASE = "https://ioinformatics.org"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.label = ""

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = ""

    def handle_data(self, data):
        if self.href:
            self.label += data

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, self.label.strip()))
            self.href = None


def get_links(url):
    response = httpx.get(url, follow_redirects=True, timeout=40)
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    return parser.links


def year_links(item):
    year, url = item
    try:
        links = get_links(url)
        tasks = [
            (year, urljoin(BASE, href), label)
            for href, label in links
            if href.lower().endswith(".pdf")
            and re.search(r"problem|task|statement|round", href, re.I)
            and not re.search(
                r"solution|analysis|result|statistic", href + " " + label, re.I
            )
        ]
        return tasks, None
    except Exception as error:
        return [], {"year": year, "error": type(error).__name__}


def fingerprint(item):
    year, url, title = item
    entry = {"year": year, "url": url, "title": title_key(title), "shingles": []}
    try:
        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        result = subprocess.run(
            ["pdftotext", "-layout", "-", "-"],
            input=response.content,
            capture_output=True,
            timeout=30,
            check=True,
        )
        text = result.stdout.decode(errors="replace")
        entry.update(
            sha256=hashlib.sha256(response.content).hexdigest(),
            shingles=sorted(shingles(text)),
            extracted_words=len(text.split()),
        )
    except Exception as error:
        entry["error"] = type(error).__name__
    return entry


def main():
    years = [
        (int(match.group(1)), urljoin(BASE, href))
        for href, label in get_links(BASE + "/page/contests/10")
        if (match := re.search(r"/page/ioi-((?:19|20)\d{2})/", href))
        and int(match.group(1)) <= 2023
    ]
    years = sorted(set(years))
    if len(years) != 35:
        raise RuntimeError(f"Expected 35 archive years, found {len(years)}")
    tasks, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for found, error in executor.map(year_links, years):
            tasks.extend(found)
            if error:
                errors.append(error)
    tasks = list(dict.fromkeys(tasks))
    print(
        json.dumps(
            {
                "archive_years": len(years),
                "task_pdf_links": len(tasks),
                "year_fetch_errors": len(errors),
            }
        ),
        flush=True,
    )
    cache = ROOT / "ioi_archive_fingerprints.json"
    existing = (
        {entry["url"]: entry for entry in json.loads(cache.read_text())["entries"]}
        if cache.exists()
        else {}
    )
    entries = [existing[url] for year, url, title in tasks if url in existing]
    tasks = [item for item in tasks if item[1] not in existing]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for entry in executor.map(fingerprint, tasks):
            entries.append(entry)
    report = {
        "years_requested": [year for year, _ in years],
        "year_fetch_errors": errors,
        "years_without_pdf_links": sorted(
            {year for year, _ in years} - {entry["year"] for entry in entries}
        ),
        "entries": entries,
        "method": "Titles and hashed 8-word shingles only; historical statement text and PDFs are never persisted.",
    }
    write_json(ROOT / "ioi_archive_fingerprints.json", report)
    print(
        json.dumps(
            {
                "fingerprinted_statements": sum(bool(e["shingles"]) for e in entries),
                "pdf_errors": sum("error" in e for e in entries),
                "years_without_pdf_links": report["years_without_pdf_links"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
