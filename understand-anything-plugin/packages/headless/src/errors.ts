import { ZodError } from "zod";

export interface HeadlessFailure {
  exitCode: 10 | 20 | 30 | 40 | 50 | 60 | 70;
  code: string;
  message: string;
}

export class InvalidConfigurationError extends Error {
  override readonly name = "InvalidConfigurationError";
}

export function classifyHeadlessFailure(error: unknown): HeadlessFailure {
  const message = error instanceof Error ? error.message : String(error);
  if (isAbortError(error)) {
    return { exitCode: 70, code: "HEADLESS_CANCELLED", message };
  }
  if (error instanceof InvalidConfigurationError || error instanceof ZodError || matches(message, [
    "usage:", "unknown option", "missing value", "duplicate option",
    "missing required option", "must be absolute", "repo-type must",
    "commit must", "OPENAI_API_KEY", "timeoutMs", "maxRetries",
    "maxResponseBytes", "maxCompletionTokens",
  ])) {
    return { exitCode: 10, code: "INVALID_CONFIGURATION", message };
  }
  if (matches(message, [
    "SOURCE_COMMIT_MISMATCH", "SOURCE_WORKTREE_DIRTY", "not a git repository",
    "source and output directories must be disjoint", "SOURCE_PATH_ESCAPE",
    "SOURCE_SYMLINK_ESCAPE", "ENOENT",
  ])) {
    return { exitCode: 20, code: "SOURCE_INVALID", message };
  }
  if (matches(message, [
    "LLM completion failed", "LLM_SCHEMA_REPAIR_EXHAUSTED",
    "SEMANTIC_GRAPH_SCHEMA_REPAIR_EXHAUSTED",
  ])) {
    return { exitCode: 40, code: "LLM_FAILED", message };
  }
  if (matches(message, [
    "source reference", "SOURCE_COVERAGE_INCOMPLETE", "STRUCTURE_COVERAGE_INCOMPLETE",
    "DOCUMENT_LIMIT_EXCEEDED",
    "BATCH_DOCUMENT_LIMIT_EXCEEDED", "NO_ANALYZABLE_SOURCE_FILES",
    "DOCUMENT_CONTENT_TOO_LARGE", "DOCUMENT_BYTES_TOO_LARGE",
    "DUPLICATE_DOCUMENT_CONFLICT",
    "SEMANTIC_GRAPH_INVALID", "SEMANTIC_GRAPH_UNKNOWN_",
  ])) {
    return { exitCode: 50, code: "GENERATED_ARTIFACT_INVALID", message };
  }
  if (matches(message, [
    "output directory must be empty", "EACCES", "EPERM", "ENOSPC", "EISDIR",
  ])) {
    return { exitCode: 60, code: "OUTPUT_FAILED", message };
  }
  return { exitCode: 30, code: "ANALYSIS_FAILED", message };
}

function matches(message: string, fragments: string[]): boolean {
  return fragments.some((fragment) => message.includes(fragment));
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError" ||
    error instanceof Error && error.name === "AbortError";
}
