import { isAbsolute } from "node:path";

export interface GenerateCliArguments {
  command: "generate";
  sourceRoot: string;
  outputRoot: string;
  projectKey: string;
  sourceId: string;
  repoType: "FRONTEND" | "BACKEND" | "COMMON" | "DOCS";
  commitSha: string;
  configPath?: string;
}

const OPTION_NAMES = new Set([
  "--source",
  "--output",
  "--project-key",
  "--source-id",
  "--repo-type",
  "--commit",
  "--config",
]);

export function parseCliArguments(arguments_: string[]): GenerateCliArguments {
  const [command, ...tokens] = arguments_;
  if (command !== "generate") {
    throw new Error("usage: understand-headless generate [options]");
  }
  const values = new Map<string, string>();
  for (let index = 0; index < tokens.length; index += 2) {
    const option = tokens[index];
    const value = tokens[index + 1];
    if (!option?.startsWith("--") || !OPTION_NAMES.has(option)) {
      throw new Error(`unknown option ${option ?? "<missing>"}`);
    }
    if (!value || value.startsWith("--")) {
      throw new Error(`missing value for ${option}`);
    }
    if (values.has(option)) throw new Error(`duplicate option ${option}`);
    values.set(option, value);
  }

  const sourceRoot = required(values, "--source");
  const outputRoot = required(values, "--output");
  if (!isAbsolute(sourceRoot)) throw new Error("source must be absolute");
  if (!isAbsolute(outputRoot)) throw new Error("output must be absolute");
  const configPath = values.get("--config");
  if (configPath && !isAbsolute(configPath)) throw new Error("config must be absolute");

  const repoType = required(values, "--repo-type");
  if (!isRepoType(repoType)) {
    throw new Error("repo-type must be FRONTEND, BACKEND, COMMON, or DOCS");
  }
  const commitSha = required(values, "--commit");
  if (!/^[a-f0-9]{40,64}$/.test(commitSha)) {
    throw new Error("commit must be a full lowercase hexadecimal Git SHA");
  }

  return {
    command: "generate",
    sourceRoot,
    outputRoot,
    projectKey: required(values, "--project-key"),
    sourceId: required(values, "--source-id"),
    repoType,
    commitSha,
    configPath,
  };
}

function required(values: Map<string, string>, option: string): string {
  const value = values.get(option);
  if (!value) throw new Error(`missing required option ${option}`);
  return value;
}

function isRepoType(value: string): value is GenerateCliArguments["repoType"] {
  return value === "FRONTEND" || value === "BACKEND" ||
    value === "COMMON" || value === "DOCS";
}
