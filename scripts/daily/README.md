# Daily AI Digest

A GitHub Actions workflow that builds one Markdown file per day under `_daily/`
from a curated list of AI RSS / arXiv sources, with each item summarised by an
LLM (OpenRouter, DeepSeek primary / fallback).

Published on the site at **<https://runguoli.com/daily/>**, with an RSS feed at
**<https://runguoli.com/daily.xml>**.

## Files

| Path | Purpose |
|------|---------|
| `scripts/daily/build_digest.py` | Fetches sources, calls the LLM, writes the daily markdown. |
| `scripts/daily/sources.yml` | Sources and their sections. Edit to add / remove feeds. |
| `scripts/daily/requirements.txt` | Python dependencies pinned for CI. |
| `scripts/daily/state.json` | Maintained automatically — seen article GUIDs to prevent duplicates. Do **not** edit by hand. |
| `.github/workflows/daily-digest.yml` | Cron workflow (03:00 UTC+8) and manual trigger. |
| `_daily/YYYY-MM-DD.md` | Generated digests; committed by the bot. |
| `_layouts/daily.html` | Jekyll layout for a single day. |
| `_pages/daily.html` | Index page listing the most recent digests. |
| `daily.xml` | RSS feed of the last 30 digests. |

## First-time setup (once)

1. Go to **repo → Settings → Secrets and variables → Actions → New repository secret**.
2. Name: `OPENROUTER_API_KEY`  
   Value: your full OpenRouter key (starts with `sk-or-v1-…`).
3. Save.

That's it — the nightly schedule will take over.

## Trigger a run manually

1. GitHub → **Actions** tab → *Daily AI Digest*.
2. **Run workflow**. Optional inputs:
   - `dry_run` = true: skip all LLM calls, emit stub summaries. Useful to verify the plumbing without spending tokens.
   - `digest_date`: override the date (e.g. `2026-05-11`).
3. The workflow commits `_daily/<date>.md` and an updated `scripts/daily/state.json`. GitHub Pages redeploys automatically.

## Editing the sources

Open `scripts/daily/sources.yml`:

- `sections:` controls the grouping headers and order on the rendered page.
- `sources:` is a list; each entry has a `kind` (`rss` or `arxiv`) and a `section`.
- For RSS sources, set `fetch_full_text: true` to scrape the article body with trafilatura before summarising. Skip it for feeds whose `summary` already has enough text (e.g. short announcements).
- For arXiv sources, set the `category` (`cs.AI`, `cs.LG`, `cs.CL`, …) and a `lookback_hours` window.

Changes take effect at the next scheduled run.

## Cost and rate limits

With ~100-150 items/day × ~500 tokens avg, the daily OpenRouter bill on
`deepseek/deepseek-chat` stays under **$0.10/day** (roughly $3/month). Set a
monthly spend limit on OpenRouter to be safe.

## Troubleshooting

- **A source is marked unavailable on the digest page.** The warning box at the
  top of the day's page names the source and the error. Most commonly this is
  an RSS URL that changed or a flaky mirror — add a `fallback:` URL in `sources.yml`.
- **LLM calls fail with "model not found".** The script already falls back from
  `deepseek/deepseek-chat` to `deepseek/deepseek-v4-flash`. If both fail, check
  OpenRouter's model catalogue and update `OPENROUTER_MODEL` / `OPENROUTER_FALLBACK`
  in the workflow.
- **State got corrupted / want to replay a day.** Delete the offending guid
  entries from `scripts/daily/state.json`, or pass `digest_date` via the manual
  trigger.

## Locally running a dry run

```bash
pip install -r scripts/daily/requirements.txt
DIGEST_DRY_RUN=1 DIGEST_DATE=2026-05-12 python scripts/daily/build_digest.py
```

No OpenRouter key needed in dry-run; summaries will be placeholders.
