import { readFile } from "node:fs/promises";

import { z } from "zod";

import { InvalidConfigurationError } from "./errors.js";

export const HeadlessConfigSchema = z.object({
  maxSourceFiles: z.number().int().positive().default(20_000),
  maxSourceBytes: z.number().int().positive().default(536_870_912),
  maxSingleFileBytes: z.number().int().positive().default(1_048_576),
  maxBatchInputCharacters: z.number().int().positive().default(105_000),
  maxLlmPromptCharacters: z.number().int().positive().default(140_000),
  maxLlmConcurrency: z.number().int().positive().max(16).default(2),
  maxDocuments: z.number().int().positive().max(100).default(100),
  maxDocumentCharacters: z.number().int().positive().default(4_000),
  maxDocumentBytes: z.number().int().positive().default(16_384),
  language: z.string().min(1).max(32).default("zh-CN"),
  includePaths: z.array(z.string().min(1)).default([]),
  excludePaths: z.array(z.string().min(1)).default([]),
}).strict();

export type HeadlessConfig = z.output<typeof HeadlessConfigSchema>;
export type HeadlessConfigInput = z.input<typeof HeadlessConfigSchema>;

export async function loadHeadlessConfig(configPath?: string): Promise<HeadlessConfig> {
  if (!configPath) return HeadlessConfigSchema.parse({});
  let raw: string;
  try {
    raw = await readFile(configPath, "utf8");
  } catch (error) {
    throw new InvalidConfigurationError(
      `cannot read config file: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch (error) {
    throw new InvalidConfigurationError(
      `config file is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  return HeadlessConfigSchema.parse(parsed);
}
