# Project Showcase Page Guide

This guide explains how to create a showcase page under **Projects** for each paper. A project page is an academic showcase: it summarizes the work with key figures and tables, an interactive demo when one exists, and links. It does not reproduce the full paper.

> This file lives in `_notes/`. Jekyll ignores directories that begin with an underscore by default, so it is not published.

## 1. Principles

- **One page per paper.** Each project page corresponds to an entry in `_publications/` and uses the same slug.
- **Showcase, not full text.** The full content stays in the PDF. A project page keeps a short summary, 3–6 key figures or tables, demos, and links.
- **Whole page in English.** Titles, summaries, figure captions, and notes are all in English.
- **Projects is separate from Portfolio.** Projects covers research papers only. Open-source contributions stay in Portfolio.

## 2. Naming and files

| Item | Location | Notes |
|---|---|---|
| Page source | `_projects/<slug>.md` | collection `projects`; URL `/projects/<slug>/` |
| Figures | `images/projects/<slug>/` | e.g. `fig1-main-results.png` |
| Demos and other static files | `files/projects/<slug>/` | e.g. `demo/index.html` |
| List teaser | `images/projects/<slug>/teaser.png` | optional |

- **Slug format:** `YYYY-MM-<short-name>`, e.g. `2026-10-breadthkv`. The year and month come from the `date` in `_publications/`. The short name matches the existing PDF or BibTeX file name.
- **File names:** lowercase English, words separated by hyphens, no spaces.

## 3. Front matter template

```yaml
---
title: "Spend Bytes on Breadth: ..."                    # full English title, as in the paper
collection: projects
permalink: /projects/2026-10-breadthkv/
date: 2026-10-05
status: "arXiv preprint"                                # or "Under review", "Accepted at XXX"
venue: "arXiv"
excerpt: "One-sentence summary shown on the Projects list."
teaser: "/images/projects/2026-10-breadthkv/teaser.png" # optional
tags: ["KV cache", "long reasoning"]                    # optional
paperurl: "https://arxiv.org/abs/2610.05685"
pdfurl: "/files/breadthkv-kv-compression.pdf"           # optional
codeurl: "https://github.com/..."                       # optional
bibtexurl: "/files/bibtex/li2026breadthkv.bib"          # optional
publication: "/publication/2026-10-breadthkv"           # permalink of the _publications entry
---
```

## 4. Page structure

Write each page in this order:

1. **Summary (2–4 sentences).** State the problem, the method, and the main result.
2. **Key results (3–6 figures or tables).** Each one has the original title and a short note stating the takeaway.
3. **Demo (if any).** Place it after the key results.
4. **Links and citation.** arXiv, PDF, code, BibTeX, and the `citation` text.

### Figures

The example below is a template. Replace the figure number and content with the actual figure from the paper.

```markdown
![Figure 2: Main results across three reasoning models.](/images/projects/2026-10-breadthkv/fig2-main-results.png){: .align-center}

**Figure 2.** Across three reasoning models, BreadthKV beats the eviction-only baseline in 17 of 18 settings.
```

- The note states the takeaway rather than restating the figure. Keep it to three sentences at most.
- Figure numbers match the paper so readers can cross-reference the PDF.

### Tables

- Include only the 3–5 most important rows, as a Markdown table so the text is searchable.
- The full table stays in the PDF. Add a one-sentence note below the table.

### Demos

If the demo is static HTML, embed it with an iframe:

```html
<iframe src="/files/projects/2026-10-breadthkv/demo/index.html" width="100%" height="520" loading="lazy" title="Interactive demo"></iframe>
```

- Demo files go in `files/projects/<slug>/demo/`, with `index.html` as the entry point.
- Each video or GIF should stay under 10 MB. For larger files, link to YouTube or another video host.
- A demo must run standalone, without external API keys or local paths.

## 5. Asset rules

- **Image format:** PNG or WebP, no wider than 1600 px, no larger than 500 KB per file.
- **Export:** Prefer exporting the figure from the paper source as vector PDF, then convert to PNG at 300 dpi or higher.
- **Copyright:** Figures from your own papers can be reused. For figures from other papers, cite the source and confirm that reuse is permitted.
- **Third-party assets in demos:** confirm the license before including them.

## 6. Publishing checklist

1. Confirm that `_publications/<slug>` exists and that `paperurl` and `bibtexurl` are correct.
2. Put the images in `images/projects/<slug>/`.
3. Create `_projects/<slug>.md`, following sections 3 and 4.
4. Preview locally:

   ```bash
   bundle exec jekyll serve -l -H localhost
   ```

   Check that the image paths resolve, the iframe demo loads, and the layout works at mobile width.
5. Review the changes before committing.

## 7. One-time setup (applied)

These pieces were added once. New projects only need a new file in `_projects/`.

- **`_config.yml`:** the `projects` collection in `collections:` and a `_projects` entry in `defaults:`.
- **`_pages/projects.html`:** the Projects list page, sorted by date with the newest first. It shows a placeholder message while no projects exist.
- **`_data/navigation.yml`:** a Projects link placed after Publications.

## 8. Example: BreadthKV

- Publication entry: `_publications/2026-10-breadthkv.md`, `permalink: /publication/2026-10-breadthkv`
- Project page: `_projects/2026-10-breadthkv.md`, `permalink: /projects/2026-10-breadthkv/`
- Figures: `images/projects/2026-10-breadthkv/`
- BibTeX: `files/bibtex/li2026breadthkv.bib` (already exists)
- PDF: `files/breadthkv-kv-compression.pdf` (already exists)
