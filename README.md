# Chat With Any Website

Enter a URL, then ask questions about the page's content. Answers are grounded in the page and cite the excerpts they come from.

## How it works

```
Website → Content extraction → Cleaning → Chunking → Embeddings → Vector database → Retrieval → LLM response
```

| Step | Module | Implementation |
|---|---|---|
| Website | `app/extraction.py` | `httpx` download with size limit, HTML-only, and blocking of private/local addresses |
| Content extraction | `app/extraction.py` | `trafilatura` keeps the main content as Markdown and drops navigation, ads and footers |
| Cleaning | `app/cleaning.py` | Unicode and whitespace normalization, footnote markup removal, duplicate paragraph removal |
| Chunking | `app/chunking.py` | Recursive paragraph → sentence → word splitting (about 2000 characters, 300 overlap); headings stay with their section |
| Embeddings | `app/embeddings.py` | Cohere `embed-v5.0-pro`; each chunk is embedded with its page title and section heading as a prefix (`search_document` for chunks, `search_query` for questions) |
| Vector database | `app/vector_store.py` | Weaviate Cloud collection with self-provided vectors, one set of chunks per URL |
| Retrieval | `app/vector_store.py` | Hybrid search (keyword + vector, relative score fusion) filtered by URL, 25 candidates |
| Reranking | `app/reranking.py` | Cohere `rerank-v4.0-fast` keeps the best 5 candidates |
| LLM response | `app/llm.py` | `openai/gpt-oss-120b` on Groq, instructed to answer only from the retrieved excerpts |

`app/pipeline.py` ties the steps together, and `app/main.py` exposes them through a FastAPI app, with security headers (including a strict content security policy) and a web UI made of `app/static/index.html`, `styles.css` and `app.js`.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- [Node.js](https://nodejs.org/) 24 or newer, only to run the tests of the page script (`.nvmrc` has the version)
- API keys for [Cohere](https://dashboard.cohere.com/api-keys), [Weaviate Cloud](https://console.weaviate.cloud) and [Groq](https://console.groq.com/keys)

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create a Weaviate Cloud cluster (the free sandbox works) and copy its REST endpoint and an API key.

3. Create your environment file and fill in the keys:

   ```bash
   cp .env.example .env
   ```

## Run

```bash
uv run uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, load a page, and start asking questions.

## Docker

The `Dockerfile` builds a small image with only the runtime dependencies, runs as an unprivileged user, and contains no settings: they come from environment variables (see [`.env.example`](.env.example)), and the `.env` file is kept out of the build by `.dockerignore`.

```bash
docker build -t chat-with-any-website .
docker run --rm -p 8000:8000 --env-file .env -v chat-data:/app/data chat-with-any-website
```

- The volume keeps the usage records of stored pages (`data/state.db`) when the container is replaced.
- Behind a proxy, add `-e FORWARDED_ALLOW_IPS=<the proxy's address>` so that each visitor's own address is used for the [limits per visitor](#limits-per-visitor). Without it, every visitor looks like the proxy.
- The image does not include the optional browser fallback for JavaScript-rendered pages.
- The base images are pinned to exact digests, and Dependabot proposes updates. CI builds the image and checks it on every pull request.

## Evaluate

`eval/run_eval.py` loads the pages in a question set, asks every question, and reports retrieval hit rate, faithfulness and correctness. The last two are scored by a judge model. See `eval/questions.example.json` for the format, and add your own pages and questions.

```bash
uv run python eval/run_eval.py eval/questions.example.json --out eval/results.json
```

Run it before and after changing chunk size, `HYBRID_ALPHA`, `TOP_K` or the reranker to see whether the change helped.

## JavaScript-rendered pages

Pages that build their content in the browser return little text to a plain download. Set `BROWSER_FALLBACK=true` to retry those pages in a headless browser (when the plain HTML has fewer than `MIN_TEXT_CHARS` characters of text):

```bash
uv sync --extra browser
uv run crawl4ai-setup
```

The browser loads the page and the resources it requests, so only enable this when you trust the sites users will enter.

## API

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/api/ingest` | `{"url": "...", "refresh": false}` | Makes a page ready: reuses a stored copy if there is one, otherwise fetches, cleans, chunks, embeds and stores it |
| `DELETE` | `/api/page?url=...` | none | Deletes the stored copy of a page |
| `POST` | `/api/ask` | `{"url": "...", "question": "..."}` | Answers a question about a previously loaded page |

Interactive API docs are available at http://127.0.0.1:8000/docs.

## Configuration

All settings are read from environment variables or `.env`. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `COHERE_EMBED_MODEL` | `embed-v5.0-pro` | Cohere embedding model |
| `COHERE_EMBED_DIMENSION` | `1024` | Embedding size |
| `RERANK_ENABLED` | `true` | Rerank retrieved chunks with Cohere |
| `COHERE_RERANK_MODEL` | `rerank-v4.0-fast` | Cohere rerank model |
| `WEAVIATE_COLLECTION` | `WebsiteChunk` | Weaviate collection name |
| `WEAVIATE_INIT_TIMEOUT_SECONDS` | `30` | Seconds to wait for Weaviate at startup (the client's own default of 2 can fail on a slow connection) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Chat model on Groq |
| `GROQ_BACKUP_API_KEYS` | empty | More Groq keys, separated by commas. `GROQ_API_KEY` is used until its limit is reached, then each backup key in turn, and after the last one the first again; a key whose limit was reached is skipped until it has recovered |
| `CHUNK_SIZE` | `2000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `300` | Characters shared between consecutive chunks |
| `RETRIEVE_K` | `25` | Candidates fetched from Weaviate before reranking |
| `TOP_K` | `3` | Chunks passed to the LLM (3 and 5 scored the same on the evaluation set; 3 sends fewer tokens) |
| `HYBRID_ALPHA` | `0.65` | Blend of vector and keyword search (1 = vector only, 0 = keyword only) |
| `BROWSER_FALLBACK` | `false` | Retry thin pages in a headless browser |
| `MIN_TEXT_CHARS` | `500` | Text length below which the browser fallback is tried |
| `RETRY_ATTEMPTS` | `4` | Most tries per call to Cohere or Groq, including the first |
| `RETRY_MAX_WAIT_SECONDS` | `20` | Longest single wait; a service that asks for more is not waited for |
| `RETRY_BUDGET_SECONDS` | `45` | Longest total waiting per call |
| `RATE_LIMIT_ENABLED` | `true` | Limit how often one visitor may ask questions and load pages |
| `RATE_LIMIT_ASKS_PER_MINUTE` | `6` | Questions one visitor may ask in a minute |
| `RATE_LIMIT_ASKS_PER_DAY` | `60` | Questions one visitor may ask in a day (a sliding 24 hours) |
| `RATE_LIMIT_LOADS_PER_HOUR` | `20` | Page loads and deletes one visitor may make in an hour |
| `PAGE_IDLE_MINUTES` | `15` | Minutes without a question after which a stored page is deleted |
| `PAGE_MAX_AGE_HOURS` | `12` | Age after which a stored page is deleted even if it is in use |
| `MAX_STORED_CHUNKS` | `60000` | Size limit; least recently used pages are removed first |
| `CLEANUP_INTERVAL_SECONDS` | `60` | How often the background cleanup runs |
| `STATE_DB_PATH` | `data/state.db` | SQLite file with the usage times of stored pages |

If you change the embedding model, the dimension, the chunk settings or the code that builds chunks, use a new `WEAVIATE_COLLECTION` name, because vectors and chunks stored in an existing collection won't match the new settings (or re-load each page).

## Development

One command runs everything that must pass before a commit: type-check (mypy, strict), lint (Ruff), format check (Ruff) and tests (pytest with coverage):

```bash
uv run poe check
```

| Command | What it does |
|---|---|
| `uv run poe check` | Types, lint, format check, Python tests and page script tests |
| `uv run poe fix` | Apply Ruff's automatic lint fixes and formatting |
| `uv run poe types` / `lint` / `format` / `test` / `test-js` | Run a single step |

### Git hooks

Install the hooks once after cloning:

```bash
uv run pre-commit install
```

Before every commit the hooks then run:

- file hygiene checks (merge conflict markers, large files, private keys, valid YAML/TOML/JSON, line endings)
- a secret scan of the staged changes (gitleaks), and a block on committing `.env` files
- `uv run poe check`

The commit message must follow [Conventional Commits](https://www.conventionalcommits.org/), for example `feat: add reranking` or `fix(extraction): reject invalid ports`. Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

### Conventions

- Public modules, classes and functions have docstrings (Google style), enforced by Ruff.
- Logic that does not need the network lives in plain functions that are unit-tested in `tests/`; the external services (Cohere, Weaviate, Groq) are replaced by fakes in tests.
- The interface colours live in `app/static/styles.css` and follow the system light or dark setting. `tests/js/contrast.test.js` reads that file and checks the contrast of every text and background pair in both themes, so a colour change that hurts readability fails the build.
- Secrets live only in `.env`, which is git-ignored.

## Contributing

Work happens on branches created from `develop` and is merged into `develop` through pull requests; releases go from `develop` to `main`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full flow and [SECURITY.md](SECURITY.md) to report a vulnerability.

## License

Released under the [MIT License](LICENSE).

## How sources are shown

A source is the whole section of the page that was used, but a question is usually about one fact in it. So each source first shows the line or sentences that best match the question and the answer, with the matching words highlighted, and the rest of the section is one click away (**Show full section**). The match is calculated in the browser from the question, the answer and the section text, so it costs no extra calls to any service. It compares three-letter groups instead of whole words, so it works with word forms, with Arabic, and with languages written without spaces. When the answer names code in backticks, the line that defines that name is chosen. If nothing matches well, the beginning of the section is shown instead.

## Rate limits

The free plans of Cohere and Groq limit how fast requests may come (Groq's free plan allows about 8,000 tokens a minute, which is a few questions). When a service answers "too many requests" or is briefly unavailable, the app waits for the time the service names, or a growing pause when it names none, and tries again, up to `RETRY_ATTEMPTS` tries and `RETRY_BUDGET_SECONDS` of waiting. While it waits, the page says it is still working. If a service asks for a longer wait than `RETRY_MAX_WAIT_SECONDS`, the app does not hold the visitor up: it answers with a message that says how long to wait ("We're busy right now. Please try again in about 23 seconds.") and sends a `Retry-After` header. Errors that trying again cannot fix, such as a wrong key, are shown at once.

### Limits per visitor

To keep one visitor from using up the free plans, the API limits each visitor (told apart by network address) to `RATE_LIMIT_ASKS_PER_MINUTE` questions a minute, `RATE_LIMIT_ASKS_PER_DAY` questions a day, and `RATE_LIMIT_LOADS_PER_HOUR` page loads and deletes an hour. A visitor over a limit gets a message that says how long to wait ("You're asking questions too quickly. Please try again in about 20 seconds.") and a `Retry-After` header, and the request is not passed on to any service. The counts are kept in memory, so they restart with the server and are not shared between several server processes. Behind a proxy, start uvicorn with `--proxy-headers` and `--forwarded-allow-ips` set to the proxy's address, so that each visitor's own address is used; otherwise every visitor looks like the proxy. The `X-Forwarded-For` header itself is never read by the app, because anyone can send one.

## How long pages are kept

Each address is stored once, and everyone who asks about the same address shares that stored copy, so a second visitor does not wait or use more API calls. Different addresses are kept apart, and every question only searches the page it is about.

A stored page is deleted when:

- nobody has asked about it for `PAGE_IDLE_MINUTES` (15 by default), because the last visitor has left or stopped asking;
- it is older than `PAGE_MAX_AGE_HOURS` (12 by default), even if it is still in use, so a changed website is picked up; or
- the store holds more than `MAX_STORED_CHUNKS`, in which case the least recently used pages go first.

A visitor can also press **Delete data** on the page bar to remove the copy at once, and **Refresh** to fetch the latest version. If a copy has expired while someone is still asking, the page loads again automatically on their next question.

The times of use are kept in a small SQLite file (`STATE_DB_PATH`, git-ignored) and a background task checks them every `CLEANUP_INTERVAL_SECONDS`. This design is for a single server process; running several would need a shared store such as Redis.

## Privacy

The text of every page you load is sent to Cohere (embeddings and reranking) and stored in your Weaviate database, and the excerpts used to answer a question are sent to Groq. Nothing is sent anywhere else. Answers can be wrong, so check the cited sources. Stored pages are deleted automatically as described above, or at once with **Delete data**.

## Limitations

- Only the single page at the given URL is indexed; links are not crawled.
- Pages that render their content with JavaScript may return little or no text unless the browser fallback is enabled.
- Each question is answered independently; earlier messages are not used as context.
