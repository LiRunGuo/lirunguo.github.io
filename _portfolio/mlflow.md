---
title: "MLflow — SQLAlchemy Stores & Search"
excerpt: "Contributor to MLflow: filed and root-caused the numeric-attribute search failures under PostgreSQL with psycopg v3, tracing them to a string return in SearchUtils that psycopg2 tolerated but psycopg v3 does not."
collection: portfolio
category: contribution
order: 4
permalink: /portfolio/mlflow/
---

I report bugs to [MLflow](https://github.com/mlflow/mlflow) the way I would want to receive them — with a reproducer, the mechanism traced to a specific line, and an honest statement of where I stop short of knowing. The bug below was then fixed upstream in [#26180](https://github.com/mlflow/mlflow/pull/26180) by the maintainers.

### Bug reports

**[Issue #26142 — numeric attribute filters fail on PostgreSQL with psycopg v3](https://github.com/mlflow/mlflow/issues/26142)** · Filed September 24, 2026; closed as completed September 25, 2026.

The documented run-time filters fail against a psycopg v3 backend store:

```text
attributes.start_time >= 1664067852747
psycopg.errors.UndefinedFunction: operator does not exist: bigint > character varying
... AND runs.start_time > $4::VARCHAR ...
```

Three separate search surfaces are affected: `search_runs` and `search_datasets` (`created_time` / `last_update_time`), plus all three MCP registry searches — `search_mcp_servers`, `search_mcp_server_versions`, and `search_mcp_access_endpoints` (`created_at` / `last_updated_at`).

**Root cause.** For attributes in `NUMERIC_ATTRIBUTES`, `SearchUtils._get_value` (`mlflow/utils/search_utils.py:455-462`) checks that the token is numeric but returns `token.value`, which is a `str`. Each SQLAlchemy filter builder then binds that string against a `BigInteger` column — runs and evaluation datasets in `sqlalchemy_store.py`, and the three MCP searches in `mcp_server_registry/sqlalchemy_mixin.py`. psycopg v3 sends bind parameters with an explicit type (`$n::VARCHAR`), so PostgreSQL rejects the comparison; psycopg2 sends them untyped and PostgreSQL coerces, which is why the default `postgresql://` URI and the CI job both miss it.

Two details made the report more useful than the symptom. First, I noted that `SearchTraceUtils`, `SearchModelVersionUtils` and `SearchLoggedModelsUtils` *override* `_get_value` and return `int`/`float`, so their numeric filters work — which localises the defect to the base class rather than to SQLAlchemy or the database driver. Second, I ran every affected filter through the SQLAlchemy stores against the same PostgreSQL 16 database under both drivers, so the write-up carries a driver-by-driver comparison rather than a single failure trace.

I filed the closely related [`search_experiments` breakage](https://github.com/mlflow/mlflow/issues/26143) separately, since it touches a different set of store functions, and both were resolved upstream the next day.

[View my MLflow issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+author%3ALiRunGuo)
