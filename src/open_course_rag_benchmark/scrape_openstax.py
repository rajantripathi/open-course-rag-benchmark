from __future__ import annotations

import argparse
import json
import re
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


def strip_html(text: str) -> str:
    return " ".join(BeautifulSoup(text, "html.parser").stripped_strings)


def extract_preloaded_state(html: str) -> dict:
    marker = "__PRELOADED_STATE__ = "
    start = html.find(marker)
    if start == -1:
        raise ValueError("could not find __PRELOADED_STATE__ in OpenStax page")
    start += len(marker)
    depth = 0
    end = None
    for index, char in enumerate(html[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError("could not parse __PRELOADED_STATE__ JSON block")
    return json.loads(html[start:end])


def infer_numbering(section_slug: str) -> tuple[str | None, str | None]:
    match = re.match(r"(?P<chapter>\d+)(?:-(?P<section>\d+))?", section_slug)
    if not match:
        return None, None
    return match.group("chapter"), match.group("section")


def flatten_toc_node(book_slug: str, node: dict, entries: list[dict]) -> None:
    slug = node.get("slug")
    contents = node.get("contents") or []
    if slug and (node.get("toc_target_type") or not contents):
        chapter_number, section_number = infer_numbering(slug)
        entries.append(
            {
                "section_slug": slug,
                "title": strip_html(node.get("title", slug)) or slug,
                "chapter_number": chapter_number,
                "section_number": section_number,
                "url": f"{BASE_URL}/books/{book_slug}/pages/{slug}",
            }
        )
    for child in contents:
        flatten_toc_node(book_slug, child, entries)


def parse_toc_links(html: str, slug: str) -> list[dict]:
    state = extract_preloaded_state(html)
    tree = state["content"]["book"]["tree"]
    entries: list[dict] = []
    flatten_toc_node(slug, tree, entries)
    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in entries:
        section_slug = entry["section_slug"]
        if section_slug in seen:
            continue
        seen.add(section_slug)
        deduped.append(entry)
    return deduped


def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = (
        soup.select_one('div[data-book-content="true"]')
        or soup.select_one('div[data-type="page"]')
        or soup.select_one("main")
        or soup.select_one("article")
    )
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
