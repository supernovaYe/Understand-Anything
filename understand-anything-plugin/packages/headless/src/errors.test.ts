import { describe, expect, it } from "vitest";
import { z } from "zod";

import { classifyHeadlessFailure, InvalidConfigurationError } from "./errors.js";

describe("classifyHeadlessFailure", () => {
  it("maps public failure classes to stable exit codes", () => {
    expect(classifyHeadlessFailure(new Error("SOURCE_WORKTREE_DIRTY")))
      .toMatchObject({ exitCode: 20, code: "SOURCE_INVALID" });
    expect(classifyHeadlessFailure(new Error("LLM completion failed: HTTP 503")))
      .toMatchObject({ exitCode: 40, code: "LLM_FAILED" });
    expect(classifyHeadlessFailure(new Error("SOURCE_COVERAGE_INCOMPLETE")))
      .toMatchObject({ exitCode: 50, code: "GENERATED_ARTIFACT_INVALID" });
    expect(classifyHeadlessFailure(new Error("output directory must be empty")))
      .toMatchObject({ exitCode: 60, code: "OUTPUT_FAILED" });
    expect(classifyHeadlessFailure(new DOMException("cancelled", "AbortError")))
      .toMatchObject({ exitCode: 70, code: "HEADLESS_CANCELLED" });
  });

  it("classifies schema validation as invalid configuration", () => {
    const result = z.object({ count: z.number() }).safeParse({ count: "bad" });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(classifyHeadlessFailure(result.error))
        .toMatchObject({ exitCode: 10, code: "INVALID_CONFIGURATION" });
    }
  });

  it("classifies typed config I/O and parsing failures as invalid configuration", () => {
    expect(classifyHeadlessFailure(new InvalidConfigurationError("invalid JSON")))
      .toMatchObject({ exitCode: 10, code: "INVALID_CONFIGURATION" });
    expect(classifyHeadlessFailure(new InvalidConfigurationError("ENOENT")))
      .toMatchObject({ exitCode: 10, code: "INVALID_CONFIGURATION" });
  });

  it("classifies all bounded LLM client limits as invalid configuration", () => {
    expect(classifyHeadlessFailure(new Error(
      "LLM maxResponseBytes must be a positive integer")))
      .toMatchObject({ exitCode: 10, code: "INVALID_CONFIGURATION" });
    expect(classifyHeadlessFailure(new Error(
      "LLM maxCompletionTokens must be a positive integer")))
      .toMatchObject({ exitCode: 10, code: "INVALID_CONFIGURATION" });
  });
});
