#!/usr/bin/env python3
"""
Scrape H. P. Lovecraft fiction tales from hplovecraft.com into Markdown files.

Source: https://www.hplovecraft.com/writings/sources/cfhplcb.aspx
Outputs all scraped tales as .md files into a 'tales/' directory.
"""

import argparse
import html
import os
import re
import sys
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import unicodedata

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def sanitize_filename(title: str, separator: str = "_") -> str:
    """
    Convert a tale title into a safe, clean filename.
    Example: 'Dagon' -> 'dagon.md'
             'Herbert West—Reanimator' -> 'herbert_west_reanimator.md'
             'Celephaïs' -> 'celephais.md'
    """
    # Normalize unicode (e.g. ï -> i)
    normalized = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('utf-8')
    # Remove apostrophes and quotes
    cleaned = re.sub(r"['’\"]", "", normalized.lower())
    # Replace non-alphanumeric characters with separator
    cleaned = re.sub(r"[^a-z0-9]+", separator, cleaned).strip(separator)
    return f"{cleaned}.md"


def html_to_markdown(container: BeautifulSoup, tale_title: str) -> str:
    """
    Convert the story HTML container into clean, well-formatted Markdown.
    """
    # Work on a copy of the container
    soup = BeautifulSoup(str(container), 'html.parser').find()

    # 1. Remove non-content tags: images, scripts, styles
    for tag in soup.find_all(['img', 'script', 'style']):
        tag.decompose()

    # 2. Handle links inside the text (unwrap or remove navigation)
    for a in soup.find_all('a'):
        if 'Return to' in a.get_text():
            a.decompose()
        else:
            a.unwrap()

    # 3. Handle section dividers / Roman numerals (<center> tags)
    for center in soup.find_all('center'):
        c_text = center.get_text().strip()
        if c_text:
            # Skip if it's just the title or author repetition
            if c_text.lower() == tale_title.lower() or 'by h. p. lovecraft' in c_text.lower():
                center.decompose()
            else:
                center.replace_with(f"\n\n### {c_text}\n\n")
        else:
            center.decompose()

    # 4. Handle blockquotes (epigraphs, letters, incantations)
    for bq in soup.find_all('blockquote'):
        bq_text = bq.get_text().strip()
        if bq_text:
            lines = [f"> {line}" if line else ">" for line in bq_text.splitlines()]
            bq.replace_with("\n\n" + "\n".join(lines) + "\n\n")
        else:
            bq.decompose()

    # 5. Handle headers
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(h.name[1])
        h_text = h.get_text().strip()
        if h_text:
            h.replace_with(f"\n\n{'#' * level} {h_text}\n\n")
        else:
            h.decompose()

    # 6. Handle bold text
    for b in soup.find_all(['b', 'strong']):
        b_text = b.get_text().strip()
        if b_text:
            b.replace_with(f"**{b_text}**")
        else:
            b.decompose()

    # 7. Handle italic text
    for i in soup.find_all(['i', 'em']):
        i_text = i.get_text().strip()
        if i_text:
            i.replace_with(f"*{i_text}*")
        else:
            i.decompose()

    # 8. Convert line breaks
    for br in soup.find_all('br'):
        br.replace_with("\n\n")

    # 9. Handle paragraphs
    for p in soup.find_all('p'):
        p_text = p.get_text().strip()
        if p_text:
            p.replace_with(f"\n\n{p_text}\n\n")

    # Extract text and unescape HTML entities
    raw_text = html.unescape(soup.get_text())

    # Reconstruct paragraphs and normalize newlines
    raw_lines = raw_text.split('\n')
    paragraphs = []
    current_p = []

    for line in raw_lines:
        line_s = line.strip()
        if not line_s:
            if current_p:
                paragraphs.append(" ".join(current_p))
                current_p = []
        elif line_s.startswith('#') or line_s.startswith('>'):
            if current_p:
                paragraphs.append(" ".join(current_p))
                current_p = []
            paragraphs.append(line_s)
        else:
            current_p.append(line_s)

    if current_p:
        paragraphs.append(" ".join(current_p))

    return "\n\n".join(paragraphs).strip()


def scrape_tale(session: requests.Session, title: str, info_url: str) -> tuple[str, str]:
    """
    Given a tale's info page URL, navigate to the tale text and extract its content.
    Returns (tale_title, markdown_content).
    """
    # 1. Fetch info page
    resp_info = session.get(info_url, timeout=15)
    resp_info.raise_for_status()
    soup_info = BeautifulSoup(resp_info.content, 'html.parser')

    # 2. Find the Electronic Text link
    elec_header = None
    for h in soup_info.find_all(['h3', 'h4', 'b']):
        if 'Electronic Text' in h.get_text():
            elec_header = h
            break

    if not elec_header:
        # Fallback search for a link to texts/
        cand = soup_info.find('a', href=lambda h: h and 'texts/' in h)
        if not cand:
            raise ValueError(f"Could not find Electronic Text link on {info_url}")
        tale_url = urljoin(info_url, cand['href'])
    else:
        elec_container = elec_header.find_next(['ul', 'p', 'ol'])
        first_a = elec_container.find('a') if elec_container else None
        if not first_a:
            raise ValueError(f"No link found in Electronic Text section on {info_url}")
        tale_url = urljoin(info_url, first_a['href'])

    # 3. Fetch tale text page
    resp_tale = session.get(tale_url, timeout=20)
    resp_tale.raise_for_status()
    soup_tale = BeautifulSoup(resp_tale.content, 'html.parser')

    # 4. Locate the story container
    justify_divs = soup_tale.find_all('div', align='justify')
    if justify_divs:
        story_container = justify_divs[-1]
    else:
        tds = soup_tale.find_all('td', valign='top')
        if tds:
            story_container = tds[0]
        else:
            raise ValueError(f"Could not locate story content container on {tale_url}")

    # 5. Extract Markdown body
    story_body = html_to_markdown(story_container, title)

    # 6. Compose full Markdown with title header
    full_markdown = f"# {title}\n\n**By H. P. Lovecraft**\n\n---\n\n{story_body}\n"
    return title, full_markdown


def get_contents_tales(session: requests.Session, contents_url: str) -> list[tuple[str, str]]:
    """
    Extract all tale links under 'Contents' on the main page.
    """
    resp = session.get(contents_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')

    contents_header = None
    for h in soup.find_all(['h3', 'h4', 'b']):
        if 'Contents' in h.get_text():
            contents_header = h
            break

    if not contents_header:
        raise ValueError(f"Could not find 'Contents' header on {contents_url}")

    contents_ul = contents_header.find_next(['ul', 'ol'])
    if not contents_ul:
        raise ValueError("Could not find list after 'Contents' header")

    tales = []
    for li in contents_ul.find_all('li'):
        a = li.find('a')
        if a and a.get('href'):
            tale_title = a.get_text().strip()
            tale_info_url = urljoin(contents_url, a['href'])
            tales.append((tale_title, tale_info_url))

    return tales


def main():
    parser = argparse.ArgumentParser(
        description="Scrape H. P. Lovecraft fiction tales into Markdown files."
    )
    parser.add_argument(
        "--url",
        default="https://www.hplovecraft.com/writings/sources/cfhplcb.aspx",
        help="Main contents URL (default: https://www.hplovecraft.com/writings/sources/cfhplcb.aspx)",
    )
    parser.add_argument(
        "--output-dir",
        default="tales",
        help="Directory to save the markdown files (default: tales)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay in seconds between requests (default: 0.3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tales to scrape (useful for testing)",
    )
    parser.add_argument(
        "--separator",
        default="_",
        help="Separator for filename words: '_' (default) or '-'",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = os.path.abspath(args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    print(f"Target output directory: {output_path}")

    # Set up session with User-Agent
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    print(f"Fetching tale list from: {args.url}")
    tales = get_contents_tales(session, args.url)
    print(f"Found {len(tales)} tales in 'Contents'.")

    if args.limit:
        tales = tales[:args.limit]
        print(f"Limiting to first {len(tales)} tales.")

    success_count = 0
    error_count = 0

    for title, info_url in tqdm(tales, desc="Scraping tales", unit="tale"):
        filename = sanitize_filename(title, separator=args.separator)
        file_path = os.path.join(output_path, filename)

        try:
            _, md_content = scrape_tale(session, title, info_url)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            success_count += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to scrape '{title}' ({info_url}): {e}", file=sys.stderr)
            error_count += 1

        if args.delay > 0:
            time.sleep(args.delay)

    print("\n" + "=" * 50)
    print(f"Scraping complete!")
    print(f"Successfully saved: {success_count} tales to {output_path}")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
