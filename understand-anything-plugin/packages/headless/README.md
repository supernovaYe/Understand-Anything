# Understand Anything Headless

Non-interactive code knowledge generation for trusted repository snapshots.
The command reuses Understand Anything's deterministic scanner, import
resolver, Tree-sitter extractors, and graph types, then calls an
OpenAI-compatible model for bounded source-backed knowledge documents.

## Build And Test

```bash
pnpm --filter @understand-anything/core build
pnpm --filter @understand-anything/headless test
pnpm --filter @understand-anything/headless typecheck
pnpm --filter @understand-anything/headless build
```

## Deployment Bundle

Headless depends on unpublished workspace packages, including the core analysis
engine and Dart WASM parser. A plain `pnpm pack` tarball is therefore not a
standalone deployment artifact. Build the production directory with pnpm
deploy, verify its executable, and archive the complete directory:

```bash
pnpm --filter @understand-anything/headless --prod deploy --legacy /tmp/understand-headless
node /tmp/understand-headless/dist/cli.js --version
tar -C /tmp -czf /tmp/understand-headless.tar.gz understand-headless
shasum -a 256 /tmp/understand-headless.tar.gz
/tmp/understand-headless/bin/understand-headless --version
```

GitHub Actions creates a deterministic Linux archive and uploads it together
with its SHA-256 file as the `understand-headless-deployment` artifact. The
Agent Runtime image must unpack that archive unchanged, run the version smoke
check, and set `UNDERSTAND_HEADLESS_PACKAGE_SHA256` to the archived bundle's
checksum.

## Generate

The source directory must be a clean Git checkout at the exact commit passed to
the command. Tracked or untracked worktree changes are rejected. Branch
selection is intentionally unsupported; the caller prepares an authoritative
`master` snapshot before invocation.

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='http://algoagent.zhuaninc.com/algo'
export OPENAI_MODEL='qwen3.7-max'

node dist/cli.js generate \
  --source /absolute/read-only/source \
  --output /absolute/empty/output \
  --project-key self-analysis-platform \
  --source-id 1001 \
  --repo-type BACKEND \
  --commit <full-master-sha>
```

Optional environment variables:

- `OPENAI_TIMEOUT_MS`, default `300000`
- `OPENAI_MAX_RETRIES`, default `1`, transport retries only
- `OPENAI_COMPLETIONS_PATH`, default `/chat/completions`
- `OPENAI_MAX_RESPONSE_BYTES`, default `4194304`
- `OPENAI_MAX_COMPLETION_TOKENS`, default `8192`
- `UNDERSTAND_HEADLESS_FORK_COMMIT`, immutable full GitHub Fork commit embedded
  by the image build; local development defaults to `development`
- `UNDERSTAND_HEADLESS_PACKAGE_SHA256`, checksum of the complete Headless
  deployment bundle embedded by the image build; local development defaults to
  `development`
- `OPENAI_JSON_RESPONSE_FORMAT=true` for endpoints that support the OpenAI
  `response_format` request field; it is disabled by default for the internal
  algoagent-compatible endpoint

The API key is accepted only through the environment. It is never accepted as a
CLI argument, added to prompts, or written to artifacts.

## Configuration

Pass an absolute JSON file with `--config` to override these defaults:

```json
{
  "maxSourceFiles": 20000,
  "maxSourceBytes": 536870912,
  "maxSingleFileBytes": 1048576,
  "maxBatchInputCharacters": 105000,
  "maxLlmPromptCharacters": 140000,
  "maxLlmConcurrency": 2,
  "maxDocuments": 100,
  "maxDocumentCharacters": 4000,
  "maxDocumentBytes": 16384,
  "language": "zh-CN",
  "includePaths": [],
  "excludePaths": []
}
```

`includePaths` and `excludePaths` accept project-relative prefixes or `*`/`**`
patterns. The generator never silently truncates an oversized repository,
source line, model batch, model response, or document set. Each accepted
document is bounded by both semantic body characters and serialized UTF-8
bytes.

## Output

The empty output directory receives three atomic files:

- `knowledge-documents.json`: canonical structured facts, at most 99 functional
  documents plus exactly one `REPOSITORY_OVERVIEW`
- `generation-report.json`: counts, controlled exclusions, warnings, timings,
  generator/model/prompt versions, source commit, applied limits, LLM usage,
  coverage counts, and SHA-256 checksums for the knowledge and graph artifacts
- `knowledge-graph.json`: validated structural and semantic diagnostic graph

Generation first enriches the structural graph with source-bounded semantic
file summaries and relationships. It then deterministically builds functional
candidate neighborhoods from that graph before asking the model for final
knowledge documents. The report records the official base commit, Fork commit,
Node version, and package checksum alongside the model and prompt versions.

The model does not author `stableKey`, the repository overview, readiness
marker, or final RAG metadata. Every model source reference is checked against
the scanned source set, extracted symbols, original line bounds, and exact
master snapshot before output is written. Every analyzed file must be cited by
at least one accepted document or carry one of the seven explicit exclusion
reasons; incomplete coverage fails generation.

Tracked source paths are resolved before analysis. A symlink or symlinked parent
that escapes the source root fails generation before source content reaches the
model.

Progress is emitted as NDJSON on stdout. Controlled failures are emitted as one
JSON object on stderr. Exit codes are `10` invalid configuration, `20` invalid
source, `30` deterministic analysis failure, `40` LLM failure, `50` generated
artifact failure, `60` output failure, and `70` cancellation. SIGTERM/SIGINT
stop active work and do not promote partial knowledge artifacts.
