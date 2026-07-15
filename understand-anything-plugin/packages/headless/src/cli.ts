#!/usr/bin/env node

import { execFile } from "node:child_process";
import { readdir } from "node:fs/promises";
import { promisify } from "node:util";

import { parseCliArguments } from "./cli-options.js";
import { loadHeadlessConfig } from "./config.js";
import { classifyHeadlessFailure } from "./errors.js";
import { GENERATOR_VERSION, generateKnowledge } from "./generator.js";
import { OpenAiCompatibleLlmClient } from "./llm-client.js";

const execFileAsync = promisify(execFile);

async function main(signal: AbortSignal): Promise<void> {
  if (process.argv.slice(2).length === 1 && process.argv[2] === "--version") {
    process.stdout.write(`${GENERATOR_VERSION}\n`);
    return;
  }

  const cli = parseCliArguments(process.argv.slice(2));
  const actualCommit = (await execFileAsync(
    "git", ["-C", cli.sourceRoot, "rev-parse", "HEAD"])).stdout.trim().toLowerCase();
  if (actualCommit !== cli.commitSha) {
    throw new Error(
      `SOURCE_COMMIT_MISMATCH: expected ${cli.commitSha}, actual ${actualCommit}`);
  }
  const worktreeStatus = (await execFileAsync(
    "git",
    ["-C", cli.sourceRoot, "status", "--porcelain=v1", "--untracked-files=all"],
  )).stdout;
  if (worktreeStatus.length > 0) {
    throw new Error("SOURCE_WORKTREE_DIRTY: source must exactly match the requested commit");
  }
  await assertOutputIsEmpty(cli.outputRoot);

  const apiKey = process.env.OPENAI_API_KEY;
  const baseUrl = process.env.OPENAI_BASE_URL;
  const model = process.env.OPENAI_MODEL;
  if (!apiKey || !baseUrl || !model) {
    throw new Error("OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL are required");
  }

  const config = await loadHeadlessConfig(cli.configPath);
  const llm = new OpenAiCompatibleLlmClient({
    apiKey,
    baseUrl,
    model,
    timeoutMs: Number(process.env.OPENAI_TIMEOUT_MS ?? "300000"),
    maxRetries: Number(process.env.OPENAI_MAX_RETRIES ?? "1"),
    jsonResponseFormat: process.env.OPENAI_JSON_RESPONSE_FORMAT === "true",
    completionsPath: process.env.OPENAI_COMPLETIONS_PATH ?? "/chat/completions",
    maxResponseBytes: Number(process.env.OPENAI_MAX_RESPONSE_BYTES ?? "4194304"),
    maxCompletionTokens: Number(process.env.OPENAI_MAX_COMPLETION_TOKENS ?? "8192"),
  });
  const result = await generateKnowledge({
    sourceRoot: cli.sourceRoot,
    outputRoot: cli.outputRoot,
    projectKey: cli.projectKey,
    sourceId: cli.sourceId,
    repoType: cli.repoType,
    commitSha: cli.commitSha,
    config,
    signal,
    onProgress: (event) => process.stdout.write(`${JSON.stringify({
      type: "progress",
      ...event,
    })}\n`),
  }, { llm, semanticLlm: llm });
  process.stdout.write(`${JSON.stringify({
    type: "completed",
    status: "COMPLETED",
    documents: result.envelope.documents.length,
    readinessMarker: result.envelope.readinessMarker,
  })}\n`);
}

async function assertOutputIsEmpty(outputRoot: string): Promise<void> {
  try {
    const entries = await readdir(outputRoot);
    if (entries.length > 0) throw new Error("output directory must be empty");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
}

const cancellation = new AbortController();
const cancel = (): void => cancellation.abort("process signal received");
process.once("SIGTERM", cancel);
process.once("SIGINT", cancel);

try {
  await main(cancellation.signal);
} catch (error) {
  const failure = classifyHeadlessFailure(error);
  process.stderr.write(`${JSON.stringify({
    status: "FAILED",
    code: failure.code,
    message: failure.message,
  })}\n`);
  process.exitCode = failure.exitCode;
} finally {
  process.removeListener("SIGTERM", cancel);
  process.removeListener("SIGINT", cancel);
}
