import { describe, expect, it } from "vitest";

import { parseCliArguments } from "./cli-options.js";

const required = [
  "generate",
  "--source", "/workspace/source",
  "--output", "/workspace/output",
  "--project-key", "order-service",
  "--source-id", "1001",
  "--repo-type", "BACKEND",
  "--commit", "a".repeat(40),
];

describe("parseCliArguments", () => {
  it("parses the fixed master generation contract", () => {
    expect(parseCliArguments(required)).toEqual({
      command: "generate",
      sourceRoot: "/workspace/source",
      outputRoot: "/workspace/output",
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha: "a".repeat(40),
      configPath: undefined,
    });
  });

  it("rejects a caller-selected branch", () => {
    expect(() => parseCliArguments([...required, "--branch", "feature/test"]))
      .toThrow(/unknown option --branch/);
  });

  it("rejects relative source and output paths", () => {
    const relative = required.map((value) =>
      value === "/workspace/source" ? "./source" : value);
    expect(() => parseCliArguments(relative)).toThrow(/source must be absolute/);
  });
});
