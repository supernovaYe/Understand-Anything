import { execFile, spawn } from "node:child_process";
import { appendFile, cp, mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { afterEach, describe, expect, it } from "vitest";

const execFileAsync = promisify(execFile);
const here = dirname(fileURLToPath(import.meta.url));
const fixtureRoot = resolve(here, "fixtures/order-service");
const cliPath = resolve(here, "../dist/cli.js");
const temporaryDirectories: string[] = [];
const servers: ReturnType<typeof createServer>[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) =>
    new Promise<void>((resolvePromise) => server.close(() => resolvePromise()))));
  await Promise.all(temporaryDirectories.splice(0).map((path) =>
    rm(path, { recursive: true, force: true })));
});

describe("understand-headless CLI", () => {
  it("reports the package version from the built executable", async () => {
    const packageJson = JSON.parse(await readFile(
      resolve(here, "../package.json"),
      "utf8",
    )) as { version: string };
    const result = await runProcess(process.execPath, [cliPath, "--version"], process.env);

    expect(result).toMatchObject({ code: 0, stderr: "" });
    expect(result.stdout.trim()).toBe(packageJson.version);
  });

  it("rejects a dirty source checkout even when HEAD matches the requested commit", async () => {
    const root = await mkdtemp(resolve(tmpdir(), "understand-headless-cli-dirty-"));
    temporaryDirectories.push(root);
    const source = resolve(root, "source");
    const output = resolve(root, "output");
    await cp(fixtureRoot, source, { recursive: true });
    await execFileAsync("git", ["init", source]);
    await execFileAsync("git", ["-C", source, "config", "user.email", "fixture@example.com"]);
    await execFileAsync("git", ["-C", source, "config", "user.name", "Fixture"]);
    await execFileAsync("git", ["-C", source, "add", "."]);
    await execFileAsync("git", ["-C", source, "commit", "-m", "fixture"]);
    const commit = (await execFileAsync(
      "git", ["-C", source, "rev-parse", "HEAD"])).stdout.trim();
    await appendFile(
      resolve(source, "src/main/java/example/OrderService.java"),
      "\n// uncommitted change\n",
    );

    const result = await runProcess(process.execPath, [
      cliPath,
      "generate",
      "--source", source,
      "--output", output,
      "--project-key", "order-service",
      "--source-id", "1001",
      "--repo-type", "BACKEND",
      "--commit", commit,
    ], {
      ...process.env,
      OPENAI_API_KEY: "fixture-key",
      OPENAI_BASE_URL: "http://127.0.0.1:1/v1",
      OPENAI_MODEL: "fixture-model",
    });

    expect(result.code).toBe(20);
    expect(result.stdout).toBe("");
    expect(JSON.parse(result.stderr)).toMatchObject({
      status: "FAILED",
      code: "SOURCE_INVALID",
      message: expect.stringContaining("SOURCE_WORKTREE_DIRTY"),
    });
  });

  it("generates artifacts from an exact Git commit and emits NDJSON progress", async () => {
    const root = await mkdtemp(resolve(tmpdir(), "understand-headless-cli-"));
    temporaryDirectories.push(root);
    const source = resolve(root, "source");
    const output = resolve(root, "output");
    await cp(fixtureRoot, source, { recursive: true });
    await execFileAsync("git", ["init", source]);
    await execFileAsync("git", ["-C", source, "config", "user.email", "fixture@example.com"]);
    await execFileAsync("git", ["-C", source, "config", "user.name", "Fixture"]);
    await execFileAsync("git", ["-C", source, "add", "."]);
    await execFileAsync("git", ["-C", source, "commit", "-m", "fixture"]);
    const commit = (await execFileAsync(
      "git", ["-C", source, "rev-parse", "HEAD"])).stdout.trim();

    const server = createServer((request, response) => {
      let body = "";
      request.setEncoding("utf8");
      request.on("data", (chunk) => { body += chunk; });
      request.on("end", () => {
        const payload = JSON.parse(body) as {
          messages: Array<{ role: string; content: string }>;
        };
        const system = payload.messages.find((message) => message.role === "system")
          ?.content ?? "";
        const prompt = payload.messages.find((message) => message.role === "user")
          ?.content ?? "";
        if (system.includes("semantic graph")) {
          response.writeHead(200, { "content-type": "application/json" });
          response.end(JSON.stringify({ choices: [{ message: { content: JSON.stringify({
            files: sourcePaths(prompt).map((path) => ({
              path,
              summary: `${path} semantic summary`,
              tags: ["source-backed"],
              complexity: "moderate",
              symbols: [],
            })),
            relationships: [],
            warnings: [],
          }) } }] }));
          return;
        }
        const citedPath = "src/main/java/example/OrderService.java";
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({ choices: [{ message: { content: JSON.stringify({
          documents: [{
            knowledgeType: "BUSINESS_RULE",
            facetKey: "order.submit.permission",
            title: "订单提交权限校验",
            questionPatterns: ["为什么订单无法提交？", "订单提交需要什么权限？"],
            conclusion: "订单提交前必须通过权限校验。",
            applicability: "通过 OrderService.submit 提交订单时。",
            details: ["权限校验发生在持久化之前。"],
            exceptions: [],
            sourceRefs: [{
              path: citedPath,
              symbol: "submit",
              lineStart: 4,
              lineEnd: 9,
            }],
            keywords: ["订单", "提交", "权限"],
            evidenceType: "DIRECT",
          }],
          exclusions: sourcePaths(prompt)
            .filter((path) => path !== citedPath)
            .map((path) => ({ path, reason: "NON_BUSINESS_UTILITY" })),
          warnings: [],
        }) } }] }));
      });
    });
    servers.push(server);
    await new Promise<void>((resolvePromise) =>
      server.listen(0, "127.0.0.1", resolvePromise));
    const { port } = server.address() as AddressInfo;

    const result = await runProcess(process.execPath, [
      cliPath,
      "generate",
      "--source", source,
      "--output", output,
      "--project-key", "order-service",
      "--source-id", "1001",
      "--repo-type", "BACKEND",
      "--commit", commit,
    ], {
      ...process.env,
      OPENAI_API_KEY: "fixture-key",
      OPENAI_BASE_URL: `http://127.0.0.1:${port}/v1`,
      OPENAI_MODEL: "fixture-model",
    });

    expect(result.code).toBe(0);
    const events = result.stdout.trim().split("\n").map((line) => JSON.parse(line));
    expect(events.at(-1)).toMatchObject({
      type: "completed",
      status: "COMPLETED",
      documents: 2,
    });
    expect(result.stderr).toBe("");
    await expect(readFile(resolve(output, "knowledge-documents.json"), "utf8"))
      .resolves.toContain('"knowledgeType": "BUSINESS_RULE"');
  });

  it.skipIf(process.platform === "win32")(
    "terminates an active model request on SIGTERM with exit code 70", async () => {
    const root = await mkdtemp(resolve(tmpdir(), "understand-headless-cli-cancel-"));
    temporaryDirectories.push(root);
    const source = resolve(root, "source");
    const output = resolve(root, "output");
    await cp(fixtureRoot, source, { recursive: true });
    await execFileAsync("git", ["init", source]);
    await execFileAsync("git", ["-C", source, "config", "user.email", "fixture@example.com"]);
    await execFileAsync("git", ["-C", source, "config", "user.name", "Fixture"]);
    await execFileAsync("git", ["-C", source, "add", "."]);
    await execFileAsync("git", ["-C", source, "commit", "-m", "fixture"]);
    const commit = (await execFileAsync(
      "git", ["-C", source, "rev-parse", "HEAD"])).stdout.trim();
    const server = createServer((request) => request.resume());
    servers.push(server);
    await new Promise<void>((resolvePromise) =>
      server.listen(0, "127.0.0.1", resolvePromise));
    const { port } = server.address() as AddressInfo;
    const child = spawn(process.execPath, [
      cliPath,
      "generate",
      "--source", source,
      "--output", output,
      "--project-key", "order-service",
      "--source-id", "1001",
      "--repo-type", "BACKEND",
      "--commit", commit,
    ], {
      env: {
        ...process.env,
        OPENAI_API_KEY: "fixture-key",
        OPENAI_BASE_URL: `http://127.0.0.1:${port}/v1`,
        OPENAI_MODEL: "fixture-model",
      },
    });
    const exit = new Promise<number | null>((resolvePromise, reject) => {
      child.on("error", reject);
      child.on("exit", resolvePromise);
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    await waitFor(() => stdout.includes('"phase":"semantic_graph"'), 5_000);
    child.kill("SIGTERM");
    const code = await exit;

    expect(code).toBe(70);
    expect(JSON.parse(stderr)).toMatchObject({
      status: "FAILED",
      code: "HEADLESS_CANCELLED",
    });
    },
  );
});

async function waitFor(predicate: () => boolean, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("timed out waiting for child process state");
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 20));
  }
}

function sourcePaths(prompt: string): string[] {
  return [...prompt.matchAll(/^--- FILE (.+) ---$/gm)]
    .map((match) => match[1] as string)
    .filter((path, index, paths) => paths.indexOf(path) === index);
}

function runProcess(
  command: string,
  arguments_: string[],
  env: NodeJS.ProcessEnv,
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, arguments_, { env });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("exit", (code) => resolvePromise({ code, stdout, stderr }));
  });
}
