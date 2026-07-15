import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { HeadlessConfigSchema, loadHeadlessConfig } from "./config.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) =>
    rm(path, { recursive: true, force: true })));
});

describe("HeadlessConfigSchema", () => {
  it("uses the accepted 100-document MVP defaults", () => {
    expect(HeadlessConfigSchema.parse({})).toMatchObject({
      maxDocuments: 100,
      maxLlmConcurrency: 2,
      maxSourceFiles: 20_000,
      language: "zh-CN",
      includePaths: [],
      excludePaths: [],
    });
  });

  it("rejects a document limit above the RAG batch contract", () => {
    expect(() => HeadlessConfigSchema.parse({ maxDocuments: 101 }))
      .toThrow();
  });

  it("rejects unknown settings instead of silently applying defaults", () => {
    expect(() => HeadlessConfigSchema.parse({ maxSourceByte: 1 }))
      .toThrow(/Unrecognized key/);
  });

  it("classifies malformed and missing config files as invalid configuration", async () => {
    const root = await mkdtemp(resolve(tmpdir(), "understand-headless-config-invalid-"));
    temporaryDirectories.push(root);
    const malformed = resolve(root, "malformed.json");
    await writeFile(malformed, '{"maxDocuments":', "utf8");

    await expect(loadHeadlessConfig(malformed)).rejects.toMatchObject({
      name: "InvalidConfigurationError",
    });
    await expect(loadHeadlessConfig(resolve(root, "missing.json"))).rejects.toMatchObject({
      name: "InvalidConfigurationError",
    });
  });
});
