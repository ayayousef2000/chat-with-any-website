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
| `POST` | `/api/ingest` | `{"url": "..."}` | Fetches, cleans, chunks, embeds and stores a page |
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
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Chat model on Groq |
| `CHUNK_SIZE` | `2000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `300` | Characters shared between consecutive chunks |
| `RETRIEVE_K` | `25` | Candidates fetched from Weaviate before reranking |
| `TOP_K` | `5` | Chunks passed to the LLM |
| `HYBRID_ALPHA` | `0.65` | Blend of vector and keyword search (1 = vector only, 0 = keyword only) |
| `BROWSER_FALLBACK` | `false` | Retry thin pages in a headless browser |
| `MIN_TEXT_CHARS` | `500` | Text length below which the browser fallback is tried |

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

## Privacy

The text of every page you load is sent to Cohere (embeddings and reranking) and stored in your Weaviate database, and the excerpts used to answer a question are sent to Groq. Nothing is sent anywhere else. Answers can be wrong, so check the cited sources. Stored pages stay in Weaviate until you delete them.

## Limitations

- Only the single page at the given URL is indexed; links are not crawled.
- Pages that render their content with JavaScript may return little or no text unless the browser fallback is enabled.
- Each question is answered independently; earlier messages are not used as context.
