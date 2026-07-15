import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { afterEach, describe, expect, it } from "vitest";

import {
  assertEligibleStructureCoverage,
  deduplicateDocuments,
  generateKnowledge,
} from "./generator.js";
import type { JsonCompletionRequest, JsonCompletionResult } from "./llm-client.js";
import { deriveStableKey, type KnowledgeDocumentInput } from "./schema.js";

const temporaryDirectories: string[] = [];
const here = dirname(fileURLToPath(import.meta.url));
const fixtureRoot = resolve(here, "fixtures/order-service");
const frontendFixtureRoot = resolve(here, "fixtures/frontend-app");
const commitSha = "b".repeat(40);
const execFileAsync = promisify(execFile);

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) =>
    rm(path, { recursive: true, force: true })));
});

class FixtureLlmClient {
  readonly modelRef = "fixture-model";
  readonly requests: JsonCompletionRequest[] = [];
  batchScoped = false;

  constructor(private readonly sourcePath =
    "src/main/java/example/OrderService.java", private readonly sourceSymbol = "submit",
  private readonly lineStart = 4, private readonly lineEnd = 9) {}

  async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
    this.requests.push(request);
    const emitDocument = !this.batchScoped || requestContainsReference(
      request,
      this.sourcePath,
      this.lineStart,
      this.lineEnd,
    );
    return {
      value: {
        documents: emitDocument ? [{
          knowledgeType: "BUSINESS_RULE",
          facetKey: "order.submit.permission",
          title: "订单提交权限校验",
          questionPatterns: [
            "为什么订单无法提交？",
            "订单提交需要什么权限？",
          ],
          conclusion: "订单提交前必须通过 ORDER_SUBMIT 权限校验。",
          applicability: "通过 OrderService.submit 提交订单时。",
          details: ["权限校验发生在持久化之前。"],
          exceptions: [],
          sourceRefs: [{
            path: this.sourcePath,
            symbol: this.sourceSymbol,
            lineStart: this.lineStart,
            lineEnd: this.lineEnd,
          }],
          keywords: ["订单", "提交", "权限", "ORDER_SUBMIT"],
          evidenceType: "DIRECT",
        }] : [],
        exclusions: exclusionsForRequest(
          request,
          emitDocument ? [this.sourcePath] : [],
        ),
        warnings: [],
      },
      usage: { promptTokens: 100, completionTokens: 80 },
    };
  }
}

class FixtureSemanticLlmClient {
  readonly modelRef = "fixture-semantic-model";
  readonly requests: JsonCompletionRequest[] = [];

  async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
    this.requests.push(request);
    const paths = sourcePaths(request.prompt);
    const controller = "src/main/java/example/OrderController.java";
    const service = "src/main/java/example/OrderService.java";
    return {
      value: {
        files: paths.map((path) => ({
          path,
          summary: `${path} 的语义摘要`,
          tags: ["source-backed"],
          complexity: "moderate",
          symbols: [],
        })),
        relationships: paths.includes(controller) && paths.includes(service) ? [{
          sourcePath: controller,
          targetPath: service,
          type: "calls",
          description: "控制器调用订单服务提交订单",
        }] : [],
        warnings: [],
      },
      usage: { promptTokens: 30, completionTokens: 20 },
    };
  }
}

interface TestLlmClient {
  modelRef?: string;
  completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult>;
}

function testDependencies(
  llm: TestLlmClient,
  semanticLlm: TestLlmClient = new FixtureSemanticLlmClient(),
) {
  return { llm, semanticLlm };
}

function sourcePaths(prompt: string): string[] {
  return [...prompt.matchAll(/^--- FILE (.+) ---$/gm)]
    .map((match) => match[1] as string)
    .filter((path, index, paths) => paths.indexOf(path) === index);
}

function requestContainsReference(
  request: JsonCompletionRequest,
  sourcePath: string,
  lineStart: number,
  lineEnd: number,
): boolean {
  const marker = `--- FILE ${sourcePath} ---`;
  let offset = request.prompt.indexOf(marker);
  while (offset >= 0) {
    const nextFile = request.prompt.indexOf("--- FILE ", offset + marker.length);
    const section = request.prompt.slice(offset, nextFile < 0 ? undefined : nextFile);
    const match = section.match(/SOURCE LINES (\d+)-(\d+):/);
    if (match && lineStart >= Number(match[1]) && lineEnd <= Number(match[2])) {
      return true;
    }
    offset = request.prompt.indexOf(marker, offset + marker.length);
  }
  return false;
}

function exclusionsForRequest(
  request: JsonCompletionRequest,
  citedPaths: string[] = [],
): Array<{ path: string; reason: "NON_BUSINESS_UTILITY" }> {
  const cited = new Set(citedPaths);
  return [...request.prompt.matchAll(/^--- FILE (.+) ---$/gm)]
    .map((match) => match[1] as string)
    .filter((path, index, paths) => !cited.has(path) && paths.indexOf(path) === index)
    .map((path) => ({ path, reason: "NON_BUSINESS_UTILITY" as const }));
}

describe("generateKnowledge", () => {
  it("rejects conflicting documents with the same stable identity", () => {
    const input: KnowledgeDocumentInput = {
      knowledgeType: "BUSINESS_RULE",
      facetKey: "order.submit.permission",
      title: "订单提交权限校验",
      questionPatterns: ["为什么不能提交订单？", "提交订单需要什么权限？"],
      conclusion: "提交前需要权限。",
      applicability: "订单提交。",
      details: ["先校验权限。"],
      exceptions: [],
      sourceRefs: [{
        path: "src/OrderService.java",
        symbol: "submit",
        lineStart: 10,
        lineEnd: 20,
      }],
      keywords: ["订单", "权限"],
      evidenceType: "DIRECT",
    };
    const document = { ...input, stableKey: deriveStableKey(input) };

    expect(() => deduplicateDocuments([
      document,
      { ...document, conclusion: "提交前不需要权限。" },
    ])).toThrow(/DUPLICATE_DOCUMENT_CONFLICT/);
    expect(deduplicateDocuments([document, document])).toEqual([document]);
  });

  it("generates source-backed page interaction knowledge from TypeScript", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-frontend-"));
    temporaryDirectories.push(output);
    const llm = {
      modelRef: "fixture-model",
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [{
              knowledgeType: "PAGE_INTERACTION",
              facetKey: "order.submit.enabled",
              title: "订单提交可用条件",
              questionPatterns: ["为什么订单不能提交？", "提交按钮什么时候可用？"],
              conclusion: "只有待提交状态且用户有权限时才允许提交订单。",
              applicability: "订单页面提交操作。",
              details: ["校验通过后调用订单提交 API。"],
              exceptions: ["接口失败时抛出 ORDER_SUBMIT_FAILED。"],
              sourceRefs: [{
                path: "src/pages/OrderPage.tsx",
                symbol: "canSubmit",
                lineStart: 3,
                lineEnd: 5,
              }, {
                path: "src/api/orders.ts",
                symbol: "submitOrder",
                lineStart: 1,
                lineEnd: 6,
              }],
              keywords: ["订单", "提交", "PENDING", "权限"],
              evidenceType: "DIRECT",
            }],
            exclusions: exclusionsForRequest(request, [
              "src/pages/OrderPage.tsx",
              "src/api/orders.ts",
            ]),
            warnings: [],
          },
          usage: { promptTokens: 20, completionTokens: 20 },
        };
      },
    };

    const semanticLlm = new FixtureSemanticLlmClient();
    const result = await generateKnowledge({
      sourceRoot: frontendFixtureRoot,
      outputRoot: output,
      projectKey: "frontend-app",
      sourceId: "2001",
      repoType: "FRONTEND",
      commitSha,
      config: {},
    }, testDependencies(llm, semanticLlm));

    expect(result.envelope.documents).toHaveLength(2);
    expect(result.envelope.documents[1]).toMatchObject({
      knowledgeType: "PAGE_INTERACTION",
      evidenceType: "DIRECT",
    });
    expect(result.graph.nodes.some((node) => node.name === "canSubmit")).toBe(true);
  });

  it("runs the official structure pipeline and writes a validated bounded artifact", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    const llm = new FixtureLlmClient();
    const semanticLlm = new FixtureSemanticLlmClient();
    const progress: Array<{ phase: string; status: string; detail?: Record<string, number | string> }> = [];

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
      onProgress: (event) => progress.push(event),
    }, testDependencies(llm, semanticLlm));

    expect(result.envelope.documents).toHaveLength(2);
    expect(result.envelope.documents[0]?.knowledgeType).toBe("REPOSITORY_OVERVIEW");
    expect(result.envelope.documents[1]).toMatchObject({
      knowledgeType: "BUSINESS_RULE",
      evidenceType: "DIRECT",
    });
    expect(result.report).toMatchObject({
      status: "COMPLETED",
      scannedFiles: 5,
      analyzedFiles: 5,
      functionalDocuments: 1,
      totalDocuments: 2,
      modelRef: "fixture-model",
      batchCount: 1,
      llmRequestCount: 2,
      schemaRepairCount: 0,
      promptTokens: 130,
      completionTokens: 100,
      documentsByType: {
        REPOSITORY_OVERVIEW: 1,
        BUSINESS_RULE: 1,
      },
      limits: expect.objectContaining({ maxDocuments: 100 }),
      phaseTimingsMs: expect.objectContaining({ write: expect.any(Number) }),
    });
    expect(result.graph.nodes.some((node) => node.name === "OrderService")).toBe(true);
    expect(result.graph.nodes.find((node) =>
      node.id === "file:src/main/java/example/OrderService.java")?.summary)
      .toContain("语义摘要");
    expect(result.graph.edges).toContainEqual(expect.objectContaining({
      source: "file:src/main/java/example/OrderController.java",
      target: "file:src/main/java/example/OrderService.java",
      type: "calls",
      description: "控制器调用订单服务提交订单",
    }));
    expect(new Set(result.graph.nodes.map((node) => node.id)).size)
      .toBe(result.graph.nodes.length);
    expect(llm.requests).toHaveLength(1);
    expect(semanticLlm.requests).toHaveLength(1);
    expect(llm.requests[0]?.prompt).toContain("Validated semantic files");
    expect(llm.requests[0]?.prompt).toContain("database.password=[REDACTED]");
    expect(llm.requests[0]?.prompt).not.toContain("fixture-super-secret");
    expect(llm.requests[0]?.prompt).not.toContain("fixture-bearer-secret");
    expect(llm.requests[0]?.prompt).not.toContain("fixture-url-secret");
    expect(llm.requests[0]?.prompt).not.toContain("fixture-signature-value");
    expect(llm.requests[0]?.prompt)
      .not.toContain("Zml4dHVyZS11c2VyOmZpeHR1cmUtcGFzc3dvcmQ=");
    expect(progress).toContainEqual({
      phase: "generate",
      status: "progress",
      detail: { completed: 1, total: 1 },
    });

    const written = JSON.parse(await readFile(
      resolve(output, "knowledge-documents.json"), "utf8"));
    expect(written).toEqual(result.envelope);
    const graphJson = await readFile(resolve(output, "knowledge-graph.json"), "utf8");
    const reportJson = await readFile(resolve(output, "generation-report.json"), "utf8");
    expect(graphJson).toContain("OrderController");
    expect(reportJson).toContain('"status": "COMPLETED"');
    expect(result.report.artifactChecksums).toEqual({
      knowledgeDocumentsSha256: createHash("sha256")
        .update(await readFile(resolve(output, "knowledge-documents.json"), "utf8"))
        .digest("hex"),
      knowledgeGraphSha256: createHash("sha256").update(graphJson).digest("hex"),
    });
  });

  it("drops semantic relationships outside the current batch and records a warning", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    const semanticLlm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        const paths = sourcePaths(request.prompt);
        return {
          value: {
            files: paths.map((path) => ({
              path,
              summary: `${path} 的语义摘要`,
              tags: ["source-backed"],
              complexity: "moderate",
              symbols: path === paths[0] ? [{
                name: "MissingStructuralSymbol",
                summary: "模型生成的未知符号",
              }] : [],
            })),
            relationships: [{
              sourcePath: paths[0],
              targetPath: "src/main/java/example/OutsideCurrentBatch.java",
              type: "calls",
              description: "模型生成的批外关系",
            }],
            warnings: [],
          },
          usage: { promptTokens: 10, completionTokens: 10 },
        };
      },
    };

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {},
    }, testDependencies(new FixtureLlmClient(), semanticLlm));

    expect(result.report.warnings).toContain(
      "dropped 1 semantic relationship(s) outside the current batch",
    );
    expect(result.report.warnings).toContain(
      "dropped 1 semantic symbol(s) not found by structural extraction",
    );
    expect(result.graph.edges.some((edge) =>
      edge.target === "file:src/main/java/example/OutsideCurrentBatch.java")).toBe(false);
  });

  it("rejects a model document that cites a file outside the scanned source set", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(new FixtureLlmClient("../../outside.java"))))
      .rejects.toThrow(/LLM_SCHEMA_REPAIR_EXHAUSTED/);
  });

  it("accepts a non-node symbol only when it occurs inside the cited source lines", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(new FixtureLlmClient(undefined, "ORDER_SUBMIT")));

    expect(result.envelope.documents[1]?.sourceRefs[0]?.symbol).toBe("ORDER_SUBMIT");
  });

  it("repairs a document whose cited symbol is not supported by the source", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    const invalid = new FixtureLlmClient(undefined, "SQL_TEMPLATE");
    const invalidRepair = new FixtureLlmClient("../../outside.java");
    const valid = new FixtureLlmClient();
    let attempts = 0;
    const llm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        attempts += 1;
        if (attempts === 2) {
          expect(request.prompt).toContain("source reference symbol was not found");
        } else if (attempts === 3) {
          expect(request.prompt).toContain("source reference is outside the current model batch");
        }
        if (attempts === 1) return invalid.completeJson(request);
        if (attempts === 2) return invalidRepair.completeJson(request);
        return valid.completeJson(request);
      },
    };

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {},
    }, testDependencies(llm));

    expect(attempts).toBe(3);
    expect(result.report.schemaRepairCount).toBe(2);
    expect(result.envelope.documents[1]?.sourceRefs[0]?.symbol).toBe("submit");
  });

  it("accepts a class-qualified method symbol when the extracted method and lines match", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(new FixtureLlmClient(undefined, "OrderService.submit()")));

    expect(result.envelope.documents[1]?.sourceRefs[0]?.symbol)
      .toBe("OrderService.submit()");
  });

  it("rejects the reserved file symbol in model-authored documents", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(new FixtureLlmClient(undefined, "file", 1, 1))))
      .rejects.toThrow(/reserved for the repository overview/);
  });

  it("accepts a cited line inside the matching MyBatis statement element", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(new FixtureLlmClient(
      "src/main/resources/mappers/OrderMapper.xml",
      "findSubmittedOrders",
      3,
      3,
    )));

    expect(result.envelope.documents[1]?.sourceRefs[0]?.symbol)
      .toBe("findSubmittedOrders");
  });

  it("accepts a basename file symbol only for non-code configuration", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(new FixtureLlmClient(
      "src/main/resources/application.properties",
      "application.properties",
      1,
      5,
    )));

    expect(result.envelope.documents[1]?.sourceRefs[0]?.symbol)
      .toBe("application.properties");
  });

  it("accepts a dotted YAML key when the cited range contains its hierarchy", async () => {
    const root = await mkdtemp(resolve(tmpdir(), "understand-headless-yaml-"));
    temporaryDirectories.push(root);
    const source = resolve(root, "source");
    const output = resolve(root, "output");
    await mkdir(resolve(source, "config"), { recursive: true });
    await writeFile(resolve(source, "config/application.yaml"), [
      "feishu:",
      "  message:",
      "    app-id: fixture",
      "    gateway-url: http://example.invalid/send",
      "",
    ].join("\n"));
    const llm = {
      modelRef: "fixture-model",
      async completeJson(): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [{
              knowledgeType: "CONFIGURATION",
              facetKey: "feishu.message",
              title: "飞书消息配置",
              questionPatterns: ["飞书消息如何配置？", "消息网关地址在哪里配置？"],
              conclusion: "飞书消息通过 feishu.message 配置应用和网关。",
              applicability: "启用飞书消息发送时。",
              details: ["app-id 和 gateway-url 位于同一配置层级。"],
              exceptions: [],
              sourceRefs: [{
                path: "config/application.yaml",
                symbol: "feishu.message",
                lineStart: 1,
                lineEnd: 4,
              }],
              keywords: ["飞书", "message", "gateway-url"],
              evidenceType: "DIRECT",
            }],
            exclusions: [],
            warnings: [],
          },
          usage: { promptTokens: 10, completionTokens: 10 },
        };
      },
    };

    const result = await generateKnowledge({
      sourceRoot: source,
      outputRoot: output,
      projectKey: "yaml-config",
      sourceId: "3001",
      repoType: "BACKEND",
      commitSha,
      config: {},
    }, testDependencies(llm));

    expect(result.envelope.documents[1]?.sourceRefs[0]?.symbol).toBe("feishu.message");
  });

  it("fails when an eligible source file has no structural extraction result", () => {
    expect(() => assertEligibleStructureCoverage(
      [{ path: "src/Missing.java", language: "java", sizeLines: 3, fileCategory: "code" }],
      { filesAnalyzed: 0, filesSkipped: ["src/Missing.java"], results: [] },
    )).toThrow(/STRUCTURE_COVERAGE_INCOMPLETE.*src\/Missing\.java/);
  });

  it("splits an oversized source file into bounded line-preserving contexts", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    const llm = new FixtureLlmClient(undefined, undefined, 4, 4);
    llm.batchScoped = true;

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 700,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(llm));

    expect(llm.requests.length).toBeGreaterThan(1);
    expect(llm.requests.some((request) => request.prompt.includes("SOURCE LINES")))
      .toBe(true);
    expect(result.envelope.documents).toHaveLength(2);
  });

  it("fails instead of pruning when a model batch exceeds its assigned document budget", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);

    const llm = new FixtureLlmClient(undefined, undefined, 4, 4);
    llm.batchScoped = true;
    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 700,
        maxLlmConcurrency: 1,
        maxDocuments: 2,
      },
    }, testDependencies(llm)))
      .rejects.toThrow(/batch document count exceeds assigned budget/);
  });

  it("allows one focused schema repair before accepting a model batch", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    let attempts = 0;
    const repairingLlm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        attempts += 1;
        if (attempts === 1) {
          return {
            value: {
              documents: [],
              exclusions: [{ path: "OrderService.java", reason: "LOW_VALUE" }],
              warnings: [],
            },
            usage: { promptTokens: 10, completionTokens: 10 },
          };
        }
        expect(request.prompt).toContain("did not match the required schema");
        return {
          value: {
            documents: [],
            exclusions: exclusionsForRequest(request),
            warnings: [],
          },
          usage: { promptTokens: 10, completionTokens: 10 },
        };
      },
    };

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(repairingLlm));

    expect(attempts).toBe(2);
    expect(result.envelope.documents).toHaveLength(1);
  });

  it("rejects a hallucinated exclusion path", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    const llm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [],
            exclusions: [
              ...exclusionsForRequest(request),
              {
                path: "src/main/java/example/MissingService.java",
                reason: "NON_BUSINESS_UTILITY",
              },
            ],
            warnings: [],
          },
          usage: { promptTokens: 10, completionTokens: 10 },
        };
      },
    };

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(llm))).rejects.toThrow(/LLM_SCHEMA_REPAIR_EXHAUSTED/);
  });

  it("fails when analyzed source files are neither cited nor explicitly excluded", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    const llm = {
      async completeJson(): Promise<JsonCompletionResult> {
        return {
          value: { documents: [], exclusions: [], warnings: [] },
          usage: { promptTokens: 10, completionTokens: 10 },
        };
      },
    };

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(llm)))
      .rejects.toThrow(/batch source path is neither cited nor excluded/);
  });

  it("stops scheduling new batches after the first concurrent failure", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    let calls = 0;
    const llm = {
      async completeJson(): Promise<JsonCompletionResult> {
        calls += 1;
        if (calls === 1) throw new Error("first batch failed");
        await new Promise((resolvePromise) => setTimeout(resolvePromise, 25));
        return {
          value: { documents: [], exclusions: [], warnings: [] },
          usage: { promptTokens: 0, completionTokens: 0 },
        };
      },
    };

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 700,
        maxLlmConcurrency: 2,
        maxDocuments: 100,
      },
    }, testDependencies(llm))).rejects.toThrow(/first batch failed/);
    const callsAtFailure = calls;
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 150));

    expect(calls).toBe(callsAtFailure);
  });

  it("removes promoted artifacts when an artifact-set commit fails", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-output-"));
    temporaryDirectories.push(output);
    await mkdir(resolve(output, "knowledge-graph.json"));
    const llm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [],
            exclusions: exclusionsForRequest(request),
            warnings: [],
          },
          usage: { promptTokens: 0, completionTokens: 0 },
        };
      },
    };

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {
        maxSourceFiles: 100,
        maxSourceBytes: 1_000_000,
        maxSingleFileBytes: 100_000,
        maxBatchInputCharacters: 100_000,
        maxLlmConcurrency: 1,
        maxDocuments: 100,
      },
    }, testDependencies(llm))).rejects.toThrow();

    expect(await readdir(output)).toEqual(["knowledge-graph.json"]);
  });

  it("rejects a tracked symlink that escapes the source root before calling the model", async () => {
    const root = await mkdtemp(resolve(tmpdir(), "understand-headless-symlink-"));
    temporaryDirectories.push(root);
    const source = resolve(root, "source");
    const output = resolve(root, "output");
    const outside = resolve(root, "outside.java");
    await mkdir(resolve(source, "src"), { recursive: true });
    await writeFile(outside, 'class Outside { String value = "OUTSIDE_BOUNDARY"; }\n');
    await symlink(outside, resolve(source, "src/Leaked.java"));
    await execFileAsync("git", ["init", "-q", source]);
    await execFileAsync("git", ["-C", source, "config", "user.email", "fixture@example.com"]);
    await execFileAsync("git", ["-C", source, "config", "user.name", "Fixture"]);
    await execFileAsync("git", ["-C", source, "add", "."]);
    await execFileAsync("git", ["-C", source, "commit", "-qm", "fixture"]);
    let calls = 0;

    await expect(generateKnowledge({
      sourceRoot: source,
      outputRoot: output,
      projectKey: "symlink-probe",
      sourceId: "1",
      repoType: "BACKEND",
      commitSha,
      config: {},
    }, testDependencies({
      async completeJson(): Promise<JsonCompletionResult> {
        calls += 1;
        throw new Error("model must not be called");
      },
    }))).rejects.toThrow(/SOURCE_SYMLINK_ESCAPE/);
    expect(calls).toBe(0);
  });

  it("rejects references and exclusions for files outside the current model batch", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-batch-scope-"));
    temporaryDirectories.push(output);
    const allPaths = [
      "src/main/java/example/OrderController.java",
      "src/main/java/example/OrderService.java",
      "src/main/resources/application.properties",
      "src/main/resources/mappers/OrderMapper.xml",
    ];
    const llm = {
      async completeJson(): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [],
            exclusions: allPaths.map((path) => ({
              path,
              reason: "NON_BUSINESS_UTILITY",
            })),
            warnings: [],
          },
          usage: { promptTokens: 0, completionTokens: 0 },
        };
      },
    };

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: { maxBatchInputCharacters: 700, maxLlmConcurrency: 1 },
    }, testDependencies(llm))).rejects.toThrow(/LLM_SCHEMA_REPAIR_EXHAUSTED/);
  });

  it("does not promote artifacts when cancellation arrives during the write phase", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-write-cancel-"));
    temporaryDirectories.push(output);
    const controller = new AbortController();
    const llm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [],
            exclusions: exclusionsForRequest(request),
            warnings: [],
          },
          usage: { promptTokens: 0, completionTokens: 0 },
        };
      },
    };

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {},
      signal: controller.signal,
      onProgress: (event) => {
        if (event.phase === "write" && event.status === "started") {
          controller.abort("cancel during write");
        }
      },
    }, testDependencies(llm))).rejects.toMatchObject({ name: "AbortError" });
    expect(await readdir(output)).toEqual([]);
  });

  it("rejects a document whose rendered body exceeds the configured hard limit", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-doc-size-"));
    temporaryDirectories.push(output);
    const llm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [{
              knowledgeType: "BUSINESS_RULE",
              facetKey: "oversized.document",
              title: "超大知识",
              questionPatterns: ["问题一", "问题二"],
              conclusion: "结".repeat(3_500),
              applicability: "订单提交",
              details: ["详".repeat(1_000)],
              exceptions: [],
              sourceRefs: [{
                path: "src/main/java/example/OrderService.java",
                symbol: "submit",
                lineStart: 4,
                lineEnd: 9,
              }],
              keywords: ["订单"],
              evidenceType: "DIRECT",
            }],
            exclusions: exclusionsForRequest(request, [
              "src/main/java/example/OrderService.java",
            ]),
            warnings: [],
          },
          usage: { promptTokens: 0, completionTokens: 0 },
        };
      },
    };

    await expect(generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: { maxDocumentCharacters: 4_000 },
    }, testDependencies(llm))).rejects.toThrow(/DOCUMENT_CONTENT_TOO_LARGE/);
  });

  it("uses the selected overview symbol's actual source range", async () => {
    const root = await mkdtemp(resolve(tmpdir(), "understand-headless-overview-range-"));
    temporaryDirectories.push(root);
    const source = resolve(root, "source");
    const output = resolve(root, "output");
    await mkdir(source);
    await writeFile(
      resolve(source, "LateClass.java"),
      `${Array.from({ length: 25 }, (_, index) => `// preface ${index}`).join("\n")}\n` +
        "public class LateClass {}\n",
    );
    const llm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [],
            exclusions: exclusionsForRequest(request),
            warnings: [],
          },
          usage: { promptTokens: 0, completionTokens: 0 },
        };
      },
    };

    const result = await generateKnowledge({
      sourceRoot: source,
      outputRoot: output,
      projectKey: "overview-probe",
      sourceId: "1",
      repoType: "BACKEND",
      commitSha,
      config: {},
    }, testDependencies(llm));

    expect(result.envelope.documents[0]?.sourceRefs[0]).toEqual({
      path: "LateClass.java",
      symbol: "LateClass",
      lineStart: 26,
      lineEnd: 26,
    });
  });

  it("reports cited and excluded source files as disjoint coverage counts", async () => {
    const output = await mkdtemp(resolve(tmpdir(), "understand-headless-coverage-count-"));
    temporaryDirectories.push(output);
    const llm = {
      async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
        return {
          value: {
            documents: [],
            exclusions: exclusionsForRequest(request),
            warnings: [],
          },
          usage: { promptTokens: 0, completionTokens: 0 },
        };
      },
    };

    const result = await generateKnowledge({
      sourceRoot: fixtureRoot,
      outputRoot: output,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND",
      commitSha,
      config: {},
    }, testDependencies(llm));

    expect(result.report.coveredSourceFiles).toBe(0);
    expect(result.report.excludedSourceFiles).toBe(result.report.analyzedFiles);
  });
});
