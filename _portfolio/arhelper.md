---
title: "ARH — AI Research Helper"
excerpt: "Open-source CLI-first research assistant agent for AI researchers. Tool use, three-layer memory, plan–execute safety, skill self-learning, multi-LLM fallback.<br/>"
collection: portfolio
link: https://github.com/LiRunGuo/Arhelper
---

**ARH (AI Research Helper)** is a CLI-first research assistant agent for AI researchers. It turns everyday research actions — reading papers, searching code, running experiments, writing summaries — into a runtime with tool calling, context management, approval-gated writes, and persistent sessions.

All state lives locally under `~/.arh/`; your conversations, memories, and API keys never leave your machine.

- **Repository:** [github.com/LiRunGuo/Arhelper](https://github.com/LiRunGuo/Arhelper)
- **Language / Stack:** Python 3.9+, Pydantic, FastAPI, SQLAlchemy, OpenAI / Anthropic / Ollama SDKs
- **License:** MIT
- **Status:** 0.2.0-alpha (actively developed)

### Highlights

- **Plan / Execute safety model** — by default only read-only tools are exposed; write-class actions must first produce a plan and pass a user approval gate, so the agent never silently modifies your files.
- **Three-layer memory** — regex + LLM automatic extraction of user preferences and research directions; BM25 cross-session recall; end-of-session LLM reflection summary, auto-injected next time.
- **Skill self-learning** — successful complex tasks can be saved (after approval) as reusable Skills, callable via `/skill` or by the LLM.
- **Multi-LLM + fallback** — switch between OpenAI, Anthropic, and Ollama with one command; automatic cooldown-based failover to a backup model.
- **Automatic context compression** — when tokens exceed budget, an LLM summarises older turns while preserving the system prompt, recent rounds, and tool-call pairs.
- **Checkpoint & resume** — snapshot and `--resume` any session; no lost context on interruption.
- **Multi-channel gateway** — Hub-Spoke architecture; CLI and Telegram shipped, Feishu and WeChat in progress; HTTP endpoints exposed via FastAPI.
- **Research-native tools** — arXiv / HuggingFace / GitHub / OpenReview crawlers, deep paper interpretation, topic surveys, trend analysis, daily digest — all built in.

### Quick install

```bash
git clone https://github.com/LiRunGuo/Arhelper.git
cd Arhelper
pip install -e .
arh doctor
```

Or run via Docker Compose (`docker compose up`).
