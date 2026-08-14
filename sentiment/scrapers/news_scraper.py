"""
news_scraper.py
================
Scraper for Malaysian political news across the political spectrum.
Tags every article with a verified source_lean so sentiment scores
can be reported both "independent-only" and "all-sources".

Source lean verified from:
  - Media Bias/Fact Check (Malaysiakini: Left-Center, -3.8)
  - RSF Malaysia country report (Bernama: state-controlled)
  - General consensus (Quora/journalism community): FMT/Malaysiakini
    lean left-reformist, Malay Mail most centrist, Utusan/Star lean
    right/establishment.

This is NOT a claim of scientific precision — it is a transparent,
documented best-effort categorization so bias can be measured and
disclosed rather than hidden.
"""

import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

DATA_DIR = Path("data/raw/news")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── RSS Feeds across the political spectrum ────────────────────────

RSS_FEEDS = {
    # Independent, left/reformist-leaning
    "fmt": {
        "name": "Free Malaysia Today",
        "url":  "https://www.freemalaysiatoday.com/feed/",
        "lean": "independent_left",
    },
    "malaysiakini": {
        "name": "Malaysiakini",
        "url":  "https://www.malaysiakini.com/rss/en/news.rss",
        "lean": "independent_left",
    },

    # Most centrist available Malaysian outlet
    "malaymail": {
        "name": "Malay Mail",
        "url":  "https://www.malaymail.com/feed/rss/malaysia",
        "lean": "centrist",
    },

    # Establishment / right-leaning
    "utusan": {
        "name": "Utusan Malaysia",
        "url":  "https://www.utusan.com.my/feed/",
        "lean": "establishment_right",
    },

    # Government-controlled (RSF: "toe the line of whatever
    # government is in power" — disclosed, not treated as neutral)
    "bernama": {
        "name": "Bernama",
        "url":  "https://www.bernama.com/en/rssfeed.php",
        "lean": "government_official",
    },
}

# Lookup table used when tagging articles that came from
# Google News historical search (source field varies by outlet name)
SOURCE_LEAN_LOOKUP = {
    "Free Malaysia Today":              "independent_left",
    "GoogleNews-Free Malaysia Today":   "independent_left",
    "Malaysiakini":                     "independent_left",
    "GoogleNews-Malaysiakini":          "independent_left",
    "Malay Mail":                       "centrist",
    "GoogleNews-Malay Mail":            "centrist",
    "GoogleNews-The Star":              "establishment_right",
    "The Star":                         "establishment_right",
    "GoogleNews-NST Online":            "establishment_right",
    "New Straits Times":                "establishment_right",
    "Utusan Malaysia":                  "establishment_right",
    "GoogleNews-Utusan Malaysia":       "establishment_right",
    "Bernama":                          "government_official",
    "GoogleNews-Bernama":               "government_official",
    "GoogleNews-Sinar Daily":           "mainstream_malay",
    "GoogleNews-The Edge Malaysia":     "business_neutral",
}


def get_source_lean(source_name: str) -> str:
    """Look up political lean for a source name. Unknown -> 'unknown'."""
    return SOURCE_LEAN_LOOKUP.get(source_name, "unknown")


# ── Political relevance filter ─────────────────────────────────────

POLITICAL_KEYWORDS = [
    "election", "BN", "Harapan", "Pakatan", "Barisan",
    "UMNO", "DAP", "PKR", "Amanah", "Bersatu", "PN",
    "Johor", "Negeri Sembilan", "Melaka",
    "parliament", "seat", "constituency", "vote",
    "minister", "opposition", "government",
    "pilihan raya", "parlimen", "kawasan", "undi",
    "kerajaan", "pembangkang", "calon", "PRN", "PRU",
]


def is_political(text: str) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in POLITICAL_KEYWORDS)


# ── State tagging ────────────────────────────────────────────────

STATE_KEYWORDS = {
    'johor':        ['johor', 'jb', 'johor bahru', 'iskandar'],
    'neg_sembilan': ['negeri sembilan', 'seremban', 'n9', 'n. sembilan',
                      'nsembilan'],
    'melaka':       ['melaka', 'malacca'],
}


def scrape_and_tag_states(df: pd.DataFrame) -> pd.DataFrame:
    """Tag articles with relevant Malaysian states"""
    for state, keywords in STATE_KEYWORDS.items():
        df[f'mentions_{state}'] = df['full_text'].str.lower().apply(
            lambda text: any(kw.lower() in text for kw in keywords)
        )
    return df


# ── RSS parsing ─────────────────────────────────────────────────────

def parse_rss(feed_url: str, source_name: str, source_lean: str) -> list:
    """Parse RSS feed and return political articles with lean tagged."""
    articles = []

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (research bot)'}
        response = requests.get(feed_url, headers=headers, timeout=10)
        response.raise_for_status()

        content = response.content.decode('utf-8', errors='ignore')
        content = content.replace('&', '&amp;')  # fix unescaped ampersands
        content = content.replace('&amp;amp;', '&amp;')  # avoid double-escape
        content = content.replace('&amp;lt;', '&lt;').replace('&amp;gt;', '&gt;')
        content = content.replace('&amp;quot;', '&quot;').replace("&amp;#39;", "&#39;")

        root = ET.fromstring(response.content)
        items = (
            root.findall('.//item')
            or root.findall('.//{http://www.w3.org/2005/Atom}entry')
        )

        for item in items:
            title = (
                item.findtext('title')
                or item.findtext('{http://www.w3.org/2005/Atom}title')
                or ''
            ).strip()

            description = (
                item.findtext('description')
                or item.findtext('{http://www.w3.org/2005/Atom}summary')
                or ''
            ).strip()

            link = (
                item.findtext('link')
                or item.findtext('{http://www.w3.org/2005/Atom}link')
                or ''
            ).strip()

            pub_date = (
                item.findtext('pubDate')
                or item.findtext('{http://www.w3.org/2005/Atom}published')
                or datetime.now().isoformat()
            ).strip()

            full_text = f"{title} {description}"

            if is_political(full_text):
                articles.append({
                    'source':      source_name,
                    'source_lean': source_lean,
                    'title':       title,
                    'description': description[:500],
                    'url':         link,
                    'published':   pub_date,
                    'scraped_at':  datetime.now().isoformat(),
                    'language':    'en',
                    'full_text':   full_text[:1000],
                })

        print(f"  {source_name} ({source_lean}): {len(articles)} political articles")

    except Exception as e:
        print(f"  {source_name} failed: {e}")

    return articles


def scrape_all_news() -> pd.DataFrame:
    """Scrape all RSS feeds across the political spectrum"""
    all_articles = []

    for source_id, config in RSS_FEEDS.items():
        print(f"Scraping {config['name']}...")
        articles = parse_rss(config['url'], config['name'], config['lean'])
        all_articles.extend(articles)
        time.sleep(1)

    if not all_articles:
        print("No articles scraped")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df = df.drop_duplicates(subset=['title'])
    return df


# ── Google News historical search (for older articles) ─────────────

def scrape_google_news_historical(query: str, days_back: int = 90) -> list:
    """
    Google News RSS supports date-ranged search.
    Free, no API key needed. Source lean is looked up from outlet name.
    """
    encoded_query = quote(query)
    url = (
        f"https://news.google.com/rss/search?q={encoded_query}"
        f"+when:{days_back}d&hl=en-MY&gl=MY&ceid=MY:en"
    )

    articles = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        root = ET.fromstring(response.content)

        for item in root.findall('.//item'):
            title    = item.findtext('title', '').strip()
            link     = item.findtext('link', '').strip()
            pub_date = item.findtext('pubDate', '').strip()
            source   = item.findtext('source', 'Google News')

            if is_political(title):
                source_name = f'GoogleNews-{source}'
                articles.append({
                    'source':      source_name,
                    'source_lean': get_source_lean(source_name),
                    'title':       title,
                    'description': '',
                    'url':         link,
                    'published':   pub_date,
                    'scraped_at':  datetime.now().isoformat(),
                    'language':    'en',
                    'full_text':   title,
                })

        print(f"  Google News '{query}': {len(articles)} articles")

    except Exception as e:
        print(f"  Google News search failed: {e}")

    return articles


def scrape_state_historical(state: str, extra_queries: list = None,
                             days_back: int = 90) -> pd.DataFrame:
    """
    Generic historical scraper for any state.
    Usage:
        scrape_state_historical('melaka', days_back=90)
        scrape_state_historical('neg_sembilan', days_back=45)
    """
    state_display = {
        'johor': 'Johor', 'neg_sembilan': 'Negeri Sembilan', 'melaka': 'Melaka'
    }.get(state, state)

    default_queries = [
        f"{state_display} election BN PN 2026",
        f"PRN {state_display} 2026",
        f"{state_display} seat negotiations",
        f"{state_display} Umno PAS Bersatu",
    ]
    queries = default_queries + (extra_queries or [])

    all_articles = []
    for query in queries:
        articles = scrape_google_news_historical(query, days_back)
        all_articles.extend(articles)
        time.sleep(1)

    df = pd.DataFrame(all_articles)
    if df.empty:
        print(f"No historical articles found for {state}")
        return df

    df = df.drop_duplicates(subset=['title'])
    df = scrape_and_tag_states(df)

    save_path = DATA_DIR / f"{state}_historical_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(save_path, index=False)
    print(f"\nSaved {len(df)} historical {state_display} articles to {save_path}")

    return df


# Convenience wrappers (kept for backward compatibility with earlier calls)
def scrape_melaka_historical(days_back: int = 90) -> pd.DataFrame:
    return scrape_state_historical('melaka', days_back=days_back)


def scrape_ns_historical(days_back: int = 45) -> pd.DataFrame:
    return scrape_state_historical('neg_sembilan', days_back=days_back)


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Scraping Malaysian political news (balanced sources)...\n")

    df = scrape_all_news()

    if not df.empty:
        df = scrape_and_tag_states(df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = DATA_DIR / f"news_{timestamp}.csv"
        df.to_csv(save_path, index=False)
        print(f"\nSaved {len(df)} tagged articles to {save_path}")

        print(f"\nSummary:")
        print(f"  Total articles: {len(df)}")
        print(f"\n  By source:")
        print(df['source'].value_counts().to_string())
        print(f"\n  By political lean:")
        print(df['source_lean'].value_counts().to_string())

        print(f"\n  State mentions:")
        for state in ['johor', 'neg_sembilan', 'melaka']:
            col = f'mentions_{state}'
            if col in df.columns:
                count = df[col].sum()
                print(f"    {state}: {count} articles")

        print(f"\n  Sample headlines:")
        for title in df['title'].head(5):
            print(f"    -> {title}")
    else:
        print("No articles scraped. Check RSS feed URLs.")