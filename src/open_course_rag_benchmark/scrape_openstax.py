from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://openstax.org"
USER_AGENT = "open-course-rag-benchmark/0.1 (+https://github.com/rajantripathi/open-course-rag-benchmark)"


def fetch(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    time.sleep(2)
    return response.text


def parse_toc_links(html: str, slug: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[dict] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if f"/books/{slug}/pages/" not in href:
            continue
        section_url = urljoin(BASE_URL, href)
        if section_url in seen:
            continue
        seen.add(section_url)
        section_slug = section_url.rstrip("/").split("/pages/")[-1]
        title = " ".join(anchor.stripped_strings) or section_slug
        links.append({"section_slug": section_slug, "title": title, "url": section_url})
    return links


def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one(".os-text") or soup.select_one("main") or soup.select_one("article")
    if main is None:
        raise ValueError("could not find main content block")
    for tag in main.select("script,style,nav,footer,img,figure,aside"):
        tag.decompose()
    lines: list[str] = []
    for node in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        text = " ".join(node.stripped_strings)
        if not text:
            continue
        prefix = ""
        if node.name in {"h1", "h2", "h3", "h4"}:
            prefix = "#" * int(node.name[1]) + " "
        elif node.name == "li":
            prefix = "- "
        lines.append(prefix + text)
    content = "\n\n".join(lines).strip()
    if not content:
        raise ValueError("empty extracted content")
    return content


def scrape_book(book_slug: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    toc_url = f"{BASE_URL}/books/{book_slug}/pages/preface"
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    toc_html = fetch(session, toc_url)
    toc = parse_toc_links(toc_html, book_slug)
    if not toc:
        raise RuntimeError(f"no TOC links found for {book_slug}")
    for entry in toc:
        html = fetch(session, entry["url"])
        content = extract_main_content(html)
        (output_dir / f"{entry['section_slug']}.md").write_text(content, encoding="utf-8")
    (output_dir / "toc.json").write_text(json.dumps(toc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Scraped {len(toc)} sections into {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape structured OpenStax HTML pages into section markdown files.")
    parser.add_argument("--book", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    scrape_book(args.book, args.output)


if __name__ == "__main__":
    main()

