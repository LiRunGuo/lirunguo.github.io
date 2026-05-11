"""
Build one day's AI digest for runguoli.com/daily/.

Flow:
  1. Load sources.yml and state.json (seen-guid set + per-source metadata).
  2. For each source:
     - kind=rss:    parse feed, filter to "new since last run",
                    optionally fetch full HTML text with trafilatura.
     - kind=arxiv:  query the arXiv Atom API for the last N hours of
                    submissions in one category (default cs.AI / cs.LG).
                    Abstract is taken straight from the feed.
  3. For every item, call OpenRouter once to produce:
       - a short academic English summary (~80-120 words)
       - 5 keywords
     Falls back between models if the primary returns 404/model-unavailable.
  4. Render _daily/YYYY-MM-DD.md with YAML front matter containing the
     sections / items structure expected by _layouts/daily.html.
  5. Update state.json (merge new guids, truncate to last ~30 days).

Environment variables:
  OPENROUTER_API_KEY   — required
  OPENROUTER_MODEL     — optional; default deepseek/deepseek-chat
  OPENROUTER_FALLBACK  — optional; default deepseek/deepseek-v4-flash
  DIGEST_DATE          — optional YYYY-MM-DD override (default: today UTC+8)
  DIGEST_DRY_RUN       — if set, skip LLM calls and stub out summaries
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import feedparser
import requests
import trafilatura
import yaml
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# -----------------------------------------------------------------------------
# Constants / paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCES_PATH = SCRIPT_DIR / "sources.yml"
STATE_PATH = SCRIPT_DIR / "state.json"
DAILY_DIR = REPO_ROOT / "_daily"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "runguoli-daily-digest/1.0 (+https://runguoli.com/daily/)"
)
HTTP_TIMEOUT = 20  # seconds
LLM_TIMEOUT = 90
MAX_FULL_TEXT_CHARS = 12_000   # truncate before sending to LLM
STATE_RETAIN_DAYS = 45          # keep ~45d of seen-guids

SHANGHAI = dt.timezone(dt.timedelta(hours=8))

SYSTEM_PROMPT = (
    "You are an academic research assistant preparing a daily AI briefing "
    "for a computer-science graduate student. Your tone is precise, "
    "scholarly, and concise — similar to an ACL/NeurIPS abstract. Prefer "
    "technical vocabulary (e.g., 'in-context learning', 'diffusion prior', "
    "'KV-cache'), use the full proper names of models and benchmarks, and "
    "quantify claims whenever the source does so (accuracy, parameter "
    "counts, context length, etc.). Write in English regardless of the "
    "source language. Do not speculate beyond what the source states; if "
    "the provided text is too short or unclear, say so plainly in the "
    "summary instead of inventing details."
)

USER_TEMPLATE = """Source: {source}
Title: {title}
URL: {url}

Article text (may be truncated):
\"\"\"
{body}
\"\"\"

Respond with strict JSON matching this schema:
{{
  "summary": "80-120 words, academic English, self-contained. Begin with the contribution, then the method, then the results. No filler like 'This article discusses'.",
  "keywords": ["5", "distinct", "technical", "terms", "lowercase"]
}}

Return ONLY the JSON object, no prose, no code fences."""


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily")


# -----------------------------------------------------------------------------
# State (seen-guids across runs)
# -----------------------------------------------------------------------------


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"seen": {}}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("state.json is corrupt — starting fresh")
        return {"seen": {}}


def save_state(state: dict) -> None:
    # Truncate seen-guids older than STATE_RETAIN_DAYS
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=STATE_RETAIN_DAYS)).isoformat()
    state["seen"] = {guid: ts for guid, ts in state.get("seen", {}).items() if ts >= cutoff}
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Item:
    source: str
    section: str
    title: str
    url: str
    guid: str
    published: str | None
    authors: str | None
    body: str           # text to feed to the LLM
    # Filled in by summarise():
    summary: str | None = None
    keywords: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": clean_text(self.title),
            "url": self.url,
            "published": self.published,
            "authors": clean_text(self.authors) if self.authors else None,
            "summary": clean_text(self.summary) if self.summary else None,
            "keywords": self.keywords or [],
        }


@dataclasses.dataclass
class SourceError:
    source: str
    message: str

    def to_dict(self) -> dict:
        return {"source": self.source, "message": clean_text(self.message)}


# -----------------------------------------------------------------------------
# Fetching helpers
# -----------------------------------------------------------------------------


def http_get(url: str) -> requests.Response | None:
    try:
        r = requests.get(
            url,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            allow_redirects=True,
        )
        if r.status_code >= 400:
            log.warning("HTTP %s for %s", r.status_code, url)
            return None
        return r
    except requests.RequestException as e:
        log.warning("HTTP error for %s: %s", url, e)
        return None


def parse_feed(url: str) -> feedparser.FeedParserDict | None:
    r = http_get(url)
    if r is None:
        return None
    parsed = feedparser.parse(r.content)
    if parsed.bozo and not parsed.entries:
        log.warning("Feed %s was unparseable: %s", url, parsed.bozo_exception)
        return None
    return parsed


def extract_article_text(url: str) -> str:
    """Try to pull the article body from an HTML page."""
    r = http_get(url)
    if r is None:
        return ""
    try:
        text = trafilatura.extract(
            r.text,
            include_comments=False,
            include_tables=False,
            favor_recall=False,
        )
    except Exception as e:
        log.warning("trafilatura failed on %s: %s", url, e)
        return ""
    return text or ""


# -----------------------------------------------------------------------------
# RSS → Items
# -----------------------------------------------------------------------------


def entry_guid(entry) -> str:
    for key in ("id", "guid", "link"):
        if entry.get(key):
            return str(entry[key])
    # Last resort: hash title
    return entry.get("title", "untitled")


def entry_text(entry) -> str:
    # Try richest content first
    for key in ("content", "summary_detail"):
        blocks = entry.get(key)
        if isinstance(blocks, list) and blocks:
            raw = blocks[0].get("value", "")
            if raw:
                return strip_html(raw)
        if isinstance(blocks, dict) and blocks.get("value"):
            return strip_html(blocks["value"])
    if entry.get("summary"):
        return strip_html(entry["summary"])
    return ""


def entry_published_iso(entry) -> str | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            try:
                return dt.datetime(*t[:6], tzinfo=dt.timezone.utc).isoformat()
            except Exception:
                continue
    return None


def entry_authors(entry) -> str | None:
    if entry.get("authors"):
        names = [a.get("name") for a in entry["authors"] if a.get("name")]
        if names:
            return ", ".join(names[:4])
    if entry.get("author"):
        return entry["author"]
    return None


def collect_rss_source(src: dict, seen: dict) -> tuple[list[Item], SourceError | None]:
    """Fetch one RSS feed and convert unseen entries to Items."""
    candidates: list[str] = [src["url"]]
    fb = src.get("fallback")
    if isinstance(fb, str):
        candidates.append(fb)
    elif isinstance(fb, list):
        candidates.extend(fb)

    parsed = None
    for url in candidates:
        parsed = parse_feed(url)
        if parsed is not None and parsed.entries:
            if url != candidates[0]:
                log.info("Using fallback %s for %s", url, src["name"])
            break
        if parsed is None:
            log.info("  fetch failed: %s", url)
        elif not parsed.entries:
            log.info("  feed parsed but no entries: %s", url)
            parsed = None  # try next fallback

    if parsed is None:
        return [], SourceError(src["name"], f"all feed URLs failed (last tried: {candidates[-1]})")

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    items: list[Item] = []
    # Only consider the first ~30 entries to bound work
    for entry in parsed.entries[:30]:
        guid = entry_guid(entry)
        if guid in seen:
            continue
        title = html.unescape(str(entry.get("title", "untitled")).strip())
        url = str(entry.get("link", "")).strip()
        if not url:
            continue

        body = entry_text(entry)
        if src.get("fetch_full_text") and len(body) < 800:
            extracted = extract_article_text(url)
            if len(extracted) > len(body):
                body = extracted

        items.append(
            Item(
                source=src["name"],
                section=src["section"],
                title=title,
                url=url,
                guid=guid,
                published=entry_published_iso(entry),
                authors=entry_authors(entry),
                body=body[:MAX_FULL_TEXT_CHARS],
            )
        )
        # Mark seen immediately (so a retry within the run doesn't double-process)
        seen[guid] = now_iso

    log.info("  %s: %d new items", src["name"], len(items))
    return items, None


# -----------------------------------------------------------------------------
# arXiv
# -----------------------------------------------------------------------------


def arxiv_query(category: str, lookback_hours: int, max_items: int) -> list[dict]:
    """Call arXiv's Atom API for submissions within the lookback window.

    We sort by submittedDate desc and then client-side filter to the
    lookback window, because arXiv's date-range queries can miss very
    recent entries with timezone edge cases.
    """
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=cat:{category}"
        f"&start=0&max_results={max_items}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    r = http_get(url)
    if r is None:
        return []
    parsed = feedparser.parse(r.content)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lookback_hours)
    out: list[dict] = []
    for entry in parsed.entries:
        pub = entry.get("published_parsed")
        if not pub:
            continue
        pub_dt = dt.datetime(*pub[:6], tzinfo=dt.timezone.utc)
        if pub_dt < cutoff:
            # sorted desc — we can stop once we cross the cutoff
            break
        out.append(entry)
    return out


def collect_arxiv_source(src: dict, seen: dict) -> tuple[list[Item], SourceError | None]:
    entries = arxiv_query(src["category"], src["lookback_hours"], src["max_items"])
    if not entries:
        return [], SourceError(src["name"], f"arXiv returned no entries for {src['category']}")

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    items: list[Item] = []
    for entry in entries:
        guid = entry.get("id", "").strip()
        if not guid or guid in seen:
            continue
        title = re.sub(r"\s+", " ", html.unescape(str(entry.get("title", "")))).strip()
        url = guid  # arXiv IDs are canonical URLs
        abstract = strip_html(entry.get("summary", "")).strip()
        if not abstract:
            continue
        authors = entry_authors(entry)
        items.append(
            Item(
                source=src["name"],
                section=src["section"],
                title=title,
                url=url,
                guid=guid,
                published=entry_published_iso(entry),
                authors=authors,
                body=abstract[:MAX_FULL_TEXT_CHARS],
            )
        )
        seen[guid] = now_iso

    log.info("  %s: %d new items", src["name"], len(items))
    return items, None


# -----------------------------------------------------------------------------
# LLM summarisation
# -----------------------------------------------------------------------------


class LLMClient:
    def __init__(self, api_key: str, primary: str, fallback: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=LLM_TIMEOUT,
        )
        self.primary = primary
        self.fallback = fallback
        self.active = primary
        # Remember if primary already failed with "model not found" so we
        # don't keep hammering it during one run.
        self._primary_dead = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, model: str, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
            extra_headers={
                "HTTP-Referer": "https://runguoli.com/daily/",
                "X-Title": "runguoli.com Daily AI Digest",
            },
        )
        return resp.choices[0].message.content or ""

    def summarise(self, item: Item) -> None:
        user_msg = USER_TEMPLATE.format(
            source=item.source,
            title=item.title,
            url=item.url,
            body=item.body[:MAX_FULL_TEXT_CHARS] or "(no body text available)",
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            if not self._primary_dead:
                content = self._call(self.primary, messages)
                self.active = self.primary
            else:
                raise RuntimeError("primary previously disabled")
        except Exception as e:
            msg = str(e)
            if any(t in msg.lower() for t in ("not found", "404", "unknown model", "invalid model")):
                log.warning("Primary model %s unavailable (%s); switching to %s", self.primary, e, self.fallback)
                self._primary_dead = True
            else:
                log.warning("Primary model call failed (%s); trying fallback", e)
            try:
                content = self._call(self.fallback, messages)
                self.active = self.fallback
            except Exception as e2:
                log.error("Both models failed for %r: %s / %s", item.title, e, e2)
                item.summary = "(Summary unavailable — model call failed. Please refer to the original article.)"
                item.keywords = []
                return

        parsed = parse_llm_json(content)
        item.summary = parsed.get("summary") or "(No summary returned.)"
        keywords = parsed.get("keywords") or []
        if isinstance(keywords, list):
            item.keywords = [str(k).strip().lower() for k in keywords if str(k).strip()][:6]
        else:
            item.keywords = []


def parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    # strip ```json fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # Isolate the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # Try a lenient fix: single-quotes → double-quotes
        try:
            return json.loads(match.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            log.warning("Could not parse LLM JSON: %r", raw[:200])
            return {}


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------


TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(s: str) -> str:
    if not s:
        return ""
    return WS_RE.sub(" ", TAG_RE.sub(" ", html.unescape(s))).strip()


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    # Protect YAML: collapse whitespace, escape no special chars (we'll quote later)
    return WS_RE.sub(" ", s).strip()


# -----------------------------------------------------------------------------
# Markdown rendering
# -----------------------------------------------------------------------------


def render_markdown(date: dt.date, sections: list[dict], errors: list[SourceError], llm_model: str) -> str:
    total = sum(len(s["items"]) for s in sections)
    tagline = build_tagline(sections, total)
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fm = {
        "title": f"Daily AI Digest — {date.isoformat()}",
        "date": f"{date.isoformat()} 03:00:00 +0800",
        "item_count": total,
        "llm_model": llm_model,
        "generated_at": generated_at,
        "tagline": tagline,
        "sections": sections,
        "source_errors": [e.to_dict() for e in errors] if errors else [],
    }
    yaml_block = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False, width=1_000_000, default_flow_style=False)
    return f"---\n{yaml_block}---\n"


def build_tagline(sections: list[dict], total: int) -> str:
    if total == 0:
        return "No new items captured today."
    per_section = ", ".join(
        f"{len(s['items'])} {s['title'].split(' ', 1)[-1].lower()}"
        for s in sections if s["items"]
    )
    return f"{total} items · {per_section}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    dry_run = bool(os.environ.get("DIGEST_DRY_RUN"))
    if not api_key and not dry_run:
        log.error("OPENROUTER_API_KEY is not set. Set DIGEST_DRY_RUN=1 for a dry run.")
        return 2

    primary = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")
    fallback = os.environ.get("OPENROUTER_FALLBACK", "deepseek/deepseek-v4-flash")

    # Date on the site is UTC+8 "today", since that's what the user asked for.
    date_str = os.environ.get("DIGEST_DATE")
    if date_str:
        today = dt.date.fromisoformat(date_str)
    else:
        today = dt.datetime.now(SHANGHAI).date()

    log.info("Building digest for %s (primary=%s, fallback=%s, dry_run=%s)", today, primary, fallback, dry_run)

    sources_config = yaml.safe_load(SOURCES_PATH.read_text())
    sections_config = sources_config["sections"]
    sources = sources_config["sources"]

    state = load_state()
    seen = state.setdefault("seen", {})
    prior_count = len(seen)

    all_items: list[Item] = []
    errors: list[SourceError] = []

    for src in sources:
        log.info("Source: %s (%s)", src["name"], src["kind"])
        try:
            if src["kind"] == "rss":
                items, err = collect_rss_source(src, seen)
            elif src["kind"] == "arxiv":
                items, err = collect_arxiv_source(src, seen)
            else:
                err = SourceError(src["name"], f"unknown kind {src['kind']}")
                items = []
        except Exception as e:
            log.exception("Unhandled error in source %s", src["name"])
            items, err = [], SourceError(src["name"], f"exception: {e}")
        if err:
            errors.append(err)
        all_items.extend(items)
        time.sleep(0.5)

    log.info("Collected %d items across %d sources (%d new guids in this run)",
             len(all_items), len(sources), len(seen) - prior_count)

    # Summarise ------------------------------------------------------------
    if all_items:
        if dry_run:
            log.info("Dry run — stubbing summaries")
            for item in all_items:
                item.summary = item.body[:400] + ("…" if len(item.body) > 400 else "")
                item.keywords = ["dry-run"]
        else:
            llm = LLMClient(api_key=api_key, primary=primary, fallback=fallback)
            for i, item in enumerate(all_items, 1):
                log.info("  LLM [%d/%d] %s — %s", i, len(all_items), item.source, item.title[:60])
                llm.summarise(item)
    # ---------------------------------------------------------------------

    # Build sections in the order declared in sources.yml
    sections_out: list[dict] = []
    for sec in sections_config:
        sec_items = [item.to_dict() for item in all_items if item.section == sec["id"]]
        sections_out.append({
            "id": sec["id"],
            "title": sec["title"],
            "items": sec_items,
        })

    effective_model = (
        primary if not dry_run else "dry-run"
    )

    md = render_markdown(today, sections_out, errors, effective_model)

    DAILY_DIR.mkdir(exist_ok=True)
    if dry_run:
        # Never overwrite the canonical YYYY-MM-DD.md or the seen-guid state
        # during a dry run — we don't want a verification run to (a) replace
        # a real digest with a stub or (b) eat the guids of items that the
        # next real run is supposed to process.
        target = DAILY_DIR / f"dryrun-{today.isoformat()}.md"
        target.write_text(md)
        log.info("Dry-run output: %s (%d bytes). state.json was NOT updated.", target, len(md))
    else:
        target = DAILY_DIR / f"{today.isoformat()}.md"
        # Refuse to overwrite an existing real digest with an empty one — if
        # every source failed today, leave yesterday's content where it is and
        # surface the failure via Actions logs / commit diff.
        total_items = sum(len(s["items"]) for s in sections_out)
        if total_items == 0 and target.exists():
            log.warning(
                "Refusing to overwrite existing %s with a 0-item digest. "
                "Inspect the source-error list above and rerun once the feeds recover.",
                target,
            )
            return 1
        target.write_text(md)
        log.info("Wrote %s (%d bytes)", target, len(md))
        save_state(state)
        log.info("State saved (%d guids retained).", len(seen))

    return 0


if __name__ == "__main__":
    sys.exit(main())
