import { createHash, randomUUID } from "node:crypto";
import { execFile, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import {
  mkdir,
  mkdtemp,
  open,
  readFile,
  realpath,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import {
  validateGraph,
  type GraphEdge,
  type GraphNode,
  type KnowledgeGraph,
} from "@understand-anything/core";
import { z } from "zod";

import type { JsonCompletionRequest, JsonCompletionResult } from "./llm-client.js";
import { HeadlessConfigSchema, type HeadlessConfig, type HeadlessConfigInput } from "./config.js";
import {
  ExclusionSchema,
  GenerationReportSchema,
  KnowledgeDocumentSchema,
  KnowledgeDocumentsEnvelopeSchema,
  deriveStableKey,
  type GeneratedKnowledgeDocument,
  type GenerationReport,
  type KnowledgeDocumentsEnvelope,
} from "./schema.js";

export const GENERATOR_VERSION = "0.4.8";
export const UNDERSTAND_ANYTHING_VERSION = "2.8.1";
export const PROMPT_VERSION = "1.1.0";
export const OFFICIAL_BASE_COMMIT = "7f5a717694d3a94f19f523b375c777eb21548ff5";
const SCAN_SCRIPT = resolveRuntimeScript("scan-project.mjs");
const COMPUTE_BATCHES_SCRIPT = resolveRuntimeScript("compute-batches.mjs");
const IMPORT_SCRIPT = resolveRuntimeScript("extract-import-map.mjs");
const STRUCTURE_SCRIPT = resolveRuntimeScript("extract-structure.mjs");
const execFileAsync = promisify(execFile);

function resolveRuntimeScript(name: string): string {
  const bundled = fileURLToPath(new URL(`./scripts/${name}`, import.meta.url));
  if (existsSync(bundled)) return bundled;
  return fileURLToPath(new URL(`../../../skills/understand/${name}`, import.meta.url));
}

export interface JsonLlmClient {
  readonly modelRef?: string;
  completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult>;
}

export interface GenerateKnowledgeOptions {
  sourceRoot: string;
  outputRoot: string;
  projectKey: string;
  sourceId: string;
  repoType: "FRONTEND" | "BACKEND" | "COMMON" | "DOCS";
  commitSha: string;
  config: HeadlessConfigInput;
  signal?: AbortSignal;
  onProgress?: (event: ProgressEvent) => void;
}

export interface ProgressEvent {
  phase: "scan" | "structure" | "semantic_graph" | "candidates" |
    "generate" | "validate" | "write";
  status: "started" | "progress" | "completed";
  detail?: Record<string, string | number>;
}

export interface GenerateKnowledgeDependencies {
  llm: JsonLlmClient;
  semanticLlm: JsonLlmClient;
}

export interface GenerateKnowledgeResult {
  envelope: KnowledgeDocumentsEnvelope;
  report: GenerationReport;
  graph: KnowledgeGraph;
}

export interface ScannedFile {
  path: string;
  language: string;
  sizeLines: number;
  fileCategory: string;
}

interface ScanResult {
  files: ScannedFile[];
  totalFiles: number;
  stats: {
    byLanguage: Record<string, number>;
    byCategory: Record<string, number>;
  };
}

interface StructureEntry {
  path: string;
  language: string;
  fileCategory: string;
  totalLines: number;
  functions?: Array<{ name: string; startLine: number; endLine: number }>;
  classes?: Array<{
    name: string;
    startLine: number;
    endLine: number;
    methods?: Array<string | { name?: string }>;
  }>;
  callGraph?: Array<{ caller: string; callee: string; lineNumber: number }>;
  metrics?: Record<string, number>;
}

export interface StructureResult {
  filesAnalyzed: number;
  filesSkipped: string[];
  results: StructureEntry[];
}

interface ImportResult {
  importMap: Record<string, string[]>;
}

interface ComputedBatchPlan {
  algorithm: string;
  batches: Array<{
    batchIndex: number;
    files: ScannedFile[];
    batchImportData: Record<string, string[]>;
    neighborMap: Record<string, Array<{
      path: string;
      batchIndex: number;
      symbols: string[];
    }>>;
  }>;
}

interface SourceContext {
  file: ScannedFile;
  structure: StructureEntry;
  content: string;
  lineStart: number;
  lineEnd: number;
}

const FunctionalKnowledgeDocumentSchema = KnowledgeDocumentSchema.refine(
  (document) => document.knowledgeType !== "REPOSITORY_OVERVIEW",
  { message: "model batches cannot emit REPOSITORY_OVERVIEW" },
);

const BatchResponseSchema = z.object({
  documents: z.array(FunctionalKnowledgeDocumentSchema).max(30),
  exclusions: z.array(ExclusionSchema).max(2_000).default([]),
  warnings: z.array(z.string().trim().min(1).max(1_000)).max(100).default([]),
}).strict();

interface ValidatedBatchResult {
  response: z.infer<typeof BatchResponseSchema>;
  promptTokens: number;
  completionTokens: number;
  requestCount: number;
  schemaRepairCount: number;
}

const SemanticRelationshipTypeSchema = z.enum([
  "calls",
  "validates",
  "configures",
  "routes",
  "depends_on",
  "related",
  "transforms",
  "triggers",
]);

const SemanticBatchResponseSchema = z.object({
  files: z.array(z.object({
    path: z.string().trim().min(1),
    summary: z.string().trim().min(1).max(600),
    tags: z.array(z.string().trim().min(1).max(120)).max(12),
    complexity: z.enum(["simple", "moderate", "complex"]),
    symbols: z.array(z.object({
      name: z.string().trim().min(1).max(240),
      summary: z.string().trim().min(1).max(500),
    })).max(50),
  })).max(100),
  relationships: z.array(z.object({
    sourcePath: z.string().trim().min(1),
    targetPath: z.string().trim().min(1),
    type: SemanticRelationshipTypeSchema,
    description: z.string().trim().min(1).max(500),
  })).max(100),
  warnings: z.array(z.string().trim().min(1).max(1_000)).max(100).default([]),
}).strict();

type SemanticBatchResponse = z.infer<typeof SemanticBatchResponseSchema>;

interface ValidatedSemanticBatchResult {
  response: SemanticBatchResponse;
  promptTokens: number;
  completionTokens: number;
  requestCount: number;
  schemaRepairCount: number;
}

interface FunctionalCandidate {
  id: string;
  anchorPaths: string[];
  batch: SourceContext[];
  semantic: SemanticBatchResponse;
}

export async function generateKnowledge(
  options: GenerateKnowledgeOptions,
  dependencies: GenerateKnowledgeDependencies,
): Promise<GenerateKnowledgeResult> {
  const config = HeadlessConfigSchema.parse(options.config);
  validateOptions(options, config);
  throwIfAborted(options.signal);
  const operationController = new AbortController();
  const forwardCancellation = (): void =>
    operationController.abort(options.signal?.reason);
  options.signal?.addEventListener("abort", forwardCancellation, { once: true });
  const sourceRoot = await realpath(options.sourceRoot);
  await assertSourceFilesContained(sourceRoot, operationController.signal);
  assertDisjoint(sourceRoot, resolve(options.outputRoot));
  await mkdir(options.outputRoot, { recursive: true });
  const outputRoot = await realpath(options.outputRoot);
  assertDisjoint(sourceRoot, outputRoot);

  const phaseTimingsMs: Record<string, number> = {};
  const workRoot = await mkdtemp(join(tmpdir(), "understand-headless-"));
  try {
    options.onProgress?.({ phase: "scan", status: "started" });
    const scanStart = Date.now();
    const scanPath = join(workRoot, "scan-result.json");
    await runScript(SCAN_SCRIPT, [sourceRoot, scanPath], operationController.signal);
    const scan = JSON.parse(await readFile(scanPath, "utf8")) as ScanResult;
    throwIfAborted(operationController.signal);
    phaseTimingsMs.scan = Date.now() - scanStart;
    if (scan.files.length > config.maxSourceFiles) {
      throw new Error(
        `SOURCE_FILE_LIMIT_EXCEEDED: ${scan.files.length} > ${config.maxSourceFiles}`);
    }
    options.onProgress?.({
      phase: "scan",
      status: "completed",
      detail: { files: scan.files.length },
    });

    const eligibleFiles = scan.files.filter((file) =>
      isEligibleSourceFile(file, config, options.repoType));
    const deterministicExclusions = scan.files
      .filter((file) => !eligibleFiles.includes(file))
      .map((file) => ({
        path: file.path,
        reason: file.path.includes("/test/") || file.path.includes("/tests/")
          ? "TEST_ONLY_HELPER" as const
          : "GENERATED_OR_BUILD_OUTPUT" as const,
      }));

    options.onProgress?.({ phase: "structure", status: "started" });
    const structureStart = Date.now();
    const importInputPath = join(workRoot, "import-input.json");
    const importOutputPath = join(workRoot, "import-result.json");
    await writeFile(importInputPath, JSON.stringify({
      projectRoot: sourceRoot,
      files: eligibleFiles,
    }), "utf8");
    await runScript(
      IMPORT_SCRIPT,
      [importInputPath, importOutputPath],
      operationController.signal,
    );
    const imports = JSON.parse(await readFile(importOutputPath, "utf8")) as ImportResult;

    const batchScanPath = join(workRoot, "batch-scan-result.json");
    const batchOutputPath = join(workRoot, "batches.json");
    await writeFile(batchScanPath, JSON.stringify({
      ...scan,
      files: eligibleFiles,
      importMap: imports.importMap,
    }), "utf8");
    await runScript(
      COMPUTE_BATCHES_SCRIPT,
      [
        sourceRoot,
        `--scan-result=${batchScanPath}`,
        `--output=${batchOutputPath}`,
      ],
      operationController.signal,
    );
    const batchPlan = JSON.parse(
      await readFile(batchOutputPath, "utf8")) as ComputedBatchPlan;

    const structureInputPath = join(workRoot, "structure-input.json");
    const structureOutputPath = join(workRoot, "structure-result.json");
    await writeFile(structureInputPath, JSON.stringify({
      projectRoot: sourceRoot,
      batchFiles: eligibleFiles,
      batchImportData: imports.importMap,
    }), "utf8");
    await runScript(
      STRUCTURE_SCRIPT,
      [structureInputPath, structureOutputPath],
      operationController.signal,
    );
    const structure = JSON.parse(
      await readFile(structureOutputPath, "utf8")) as StructureResult;
    assertEligibleStructureCoverage(eligibleFiles, structure);
    const structureByPath = new Map(
      structure.results.map((entry) => [entry.path, entry]));
    const sourceFileCache = new Map<string, string[]>();
    phaseTimingsMs.structure = Date.now() - structureStart;
    options.onProgress?.({
      phase: "structure",
      status: "completed",
      detail: { files: structure.filesAnalyzed },
    });

    const sourceContextLimit = Math.max(
      1,
      Math.floor(config.maxBatchInputCharacters * 0.7),
    );
    const contexts = await buildSourceContexts(
      sourceRoot,
      eligibleFiles,
      structure,
      config,
      sourceContextLimit,
      operationController.signal,
    );
    const batches = buildBatchesFromPlan(
      contexts,
      batchPlan,
      sourceContextLimit,
    );

    options.onProgress?.({ phase: "semantic_graph", status: "started" });
    const semanticStart = Date.now();
    let completedSemanticBatches = 0;
    const semanticResults = await mapConcurrent(
      batches,
      config.maxLlmConcurrency,
      async (batch) => {
        try {
          throwIfAborted(operationController.signal);
          const semanticPrompt = buildSemanticBatchPrompt(options, batch, config.language);
          assertPromptSize(semanticPrompt, config.maxLlmPromptCharacters, "semantic graph");
          const result = await completeValidatedSemanticBatch(
            dependencies.semanticLlm,
            buildSemanticSystemPrompt(),
            semanticPrompt,
            batch,
            operationController.signal,
            config.maxLlmPromptCharacters,
          );
          completedSemanticBatches += 1;
          options.onProgress?.({
            phase: "semantic_graph",
            status: "progress",
            detail: { completed: completedSemanticBatches, total: batches.length },
          });
          return result;
        } catch (error) {
          operationController.abort(error);
          throw error;
        }
      },
    );
    phaseTimingsMs.semanticGraph = Date.now() - semanticStart;
    const graph = buildSemanticGraph(
      options,
      scan,
      structure,
      imports,
      semanticResults.map((result) => result.response),
    );
    options.onProgress?.({
      phase: "semantic_graph",
      status: "completed",
      detail: { batches: batches.length },
    });

    options.onProgress?.({ phase: "candidates", status: "started" });
    const candidateStart = Date.now();
    const candidates = buildFunctionalCandidates(
      batches,
      semanticResults.map((result) => result.response),
    );
    phaseTimingsMs.candidates = Date.now() - candidateStart;
    options.onProgress?.({
      phase: "candidates",
      status: "completed",
      detail: { candidates: candidates.length },
    });

    options.onProgress?.({ phase: "generate", status: "started" });
    const generateStart = Date.now();
    const functionalLimit = Math.min(99, config.maxDocuments - 1);
    const batchBudgets = allocateDocumentBudgets(candidates.length, functionalLimit);
    let completedBatches = 0;
    const batchResults = await mapConcurrent(
      candidates,
      config.maxLlmConcurrency,
      async (candidate, index) => {
        try {
          throwIfAborted(operationController.signal);
          const budget = batchBudgets[index] ?? 0;
          const system = buildSystemPrompt();
          const prompt = buildBatchPrompt(
            options, candidate, budget, config.language);
          assertPromptSize(prompt, config.maxLlmPromptCharacters, "knowledge document");
          const result = await completeValidatedBatch(
            dependencies.llm,
            system,
            prompt,
            candidate.batch,
            operationController.signal,
            config.maxLlmPromptCharacters,
            structureByPath,
            sourceRoot,
            sourceFileCache,
            budget,
            config,
          );
          completedBatches += 1;
          options.onProgress?.({
            phase: "generate",
            status: "progress",
            detail: { completed: completedBatches, total: candidates.length },
          });
          return result;
        } catch (error) {
          operationController.abort(error);
          throw error;
        }
      },
    );
    phaseTimingsMs.generate = Date.now() - generateStart;
    options.onProgress?.({
      phase: "generate",
      status: "completed",
      detail: { batches: candidates.length },
    });

    options.onProgress?.({ phase: "validate", status: "started" });
    const validateStart = Date.now();
    const modelExclusions: Array<z.infer<typeof ExclusionSchema>> = [];
    const invalidExclusionWarnings: string[] = [];
    const responses = batchResults.map((result) => result.response);
    for (const exclusion of responses.flatMap((response) => response.exclusions)) {
      if (!structureByPath.has(exclusion.path)) {
        invalidExclusionWarnings.push(
          `ignored exclusion with unknown source: ${exclusion.path}`);
      } else {
        modelExclusions.push(exclusion);
      }
    }
    const generatedCandidates: GeneratedKnowledgeDocument[] = [];
    for (const response of responses) {
      for (const document of response.documents) {
        throwIfAborted(operationController.signal);
        await validateSourceReferences(
          document, structureByPath, sourceRoot, sourceFileCache);
        generatedCandidates.push({
          ...document,
          stableKey: deriveStableKey(document),
        });
      }
    }
    const citedPaths = new Set(generatedCandidates.flatMap((document) =>
      document.sourceRefs.map((ref) => ref.path)));
    const excludedPaths = new Set(modelExclusions
      .map((exclusion) => exclusion.path)
      .filter((path) => !citedPaths.has(path)));
    const accountedPaths = new Set([...citedPaths, ...excludedPaths]);
    const unaccountedPaths = [...new Set(contexts.map((context) => context.file.path))]
      .filter((path) => !accountedPaths.has(path))
      .sort();
    if (unaccountedPaths.length > 0) {
      throw new Error(
        `SOURCE_COVERAGE_INCOMPLETE: ${unaccountedPaths.length} analyzed files ` +
        `were neither cited nor excluded: ${unaccountedPaths.slice(0, 20).join(", ")}`);
    }
    const generated = deduplicateDocuments(generatedCandidates);
    if (generated.length > functionalLimit) {
      throw new Error(
        `DOCUMENT_LIMIT_EXCEEDED: ${generated.length} > ${functionalLimit}`);
    }

    const overview = buildOverview(options, scan, structure);
    await validateSourceReferences(
      overview, structureByPath, sourceRoot, sourceFileCache);
    validateDocumentSizes([overview, ...generated], config);
    const readinessMarker = `project-code-ready:v1:${options.sourceId}:${options.commitSha}`;
    const build = buildProvenance();
    const envelope = KnowledgeDocumentsEnvelopeSchema.parse({
      schemaVersion: "1.0",
      projectKey: options.projectKey,
      sourceId: options.sourceId,
      repoType: options.repoType,
      branch: "master",
      commitSha: options.commitSha,
      readinessMarker,
      generator: {
        name: "understand-headless",
        version: GENERATOR_VERSION,
        understandAnythingVersion: UNDERSTAND_ANYTHING_VERSION,
        promptVersion: PROMPT_VERSION,
        modelRef: dependencies.llm.modelRef ?? "unspecified",
        build,
      },
      documents: [overview, ...generated],
    });
    const warnings = [
      ...structure.filesSkipped.map((path) => `structure extraction skipped ${path}`),
      ...invalidExclusionWarnings,
      ...semanticResults.flatMap((result) => result.response.warnings),
      ...responses.flatMap((response) => response.warnings),
    ];
    phaseTimingsMs.validate = Date.now() - validateStart;
    phaseTimingsMs.write = 0;
    const documentsByType = Object.fromEntries(
      [...KnowledgeDocumentSchema.shape.knowledgeType.options].map((type) => [
        type,
        envelope.documents.filter((document) => document.knowledgeType === type).length,
      ]),
    );
    const report = GenerationReportSchema.parse({
      schemaVersion: "1.0",
      status: "COMPLETED",
      sourceCommitSha: options.commitSha,
      generatorVersion: GENERATOR_VERSION,
      understandAnythingVersion: UNDERSTAND_ANYTHING_VERSION,
      promptVersion: PROMPT_VERSION,
      modelRef: dependencies.llm.modelRef ?? "unspecified",
      build,
      scannedFiles: scan.files.length,
      analyzedFiles: structure.filesAnalyzed,
      coveredSourceFiles: citedPaths.size,
      excludedSourceFiles: excludedPaths.size,
      batchCount: candidates.length,
      llmRequestCount: [...semanticResults, ...batchResults]
        .reduce((sum, item) => sum + item.requestCount, 0),
      schemaRepairCount: batchResults.reduce(
        (sum, item) => sum + item.schemaRepairCount, 0) + semanticResults.reduce(
        (sum, item) => sum + item.schemaRepairCount, 0),
      promptTokens: [...semanticResults, ...batchResults]
        .reduce((sum, item) => sum + item.promptTokens, 0),
      completionTokens: [...semanticResults, ...batchResults].reduce(
        (sum, item) => sum + item.completionTokens, 0),
      functionalDocuments: generated.length,
      totalDocuments: envelope.documents.length,
      documentsByType,
      exclusions: [
        ...deterministicExclusions,
        ...modelExclusions,
      ],
      warnings,
      limits: config,
      artifactChecksums: {
        knowledgeDocumentsSha256: sha256Json(envelope),
        knowledgeGraphSha256: sha256Json(graph),
      },
      phaseTimingsMs,
      generatedAt: new Date().toISOString(),
    });
    options.onProgress?.({ phase: "validate", status: "completed" });

    options.onProgress?.({ phase: "write", status: "started" });
    throwIfAborted(operationController.signal);
    const writeStart = Date.now();
    await writeArtifactSetAtomic(
      outputRoot,
      [
        ["knowledge-graph.json", graph],
        ["generation-report.json", report],
        ["knowledge-documents.json", envelope],
      ],
      () => { report.phaseTimingsMs.write = Date.now() - writeStart; },
      operationController.signal,
    );
    throwIfAborted(operationController.signal);
    phaseTimingsMs.write = Date.now() - writeStart;
    options.onProgress?.({ phase: "write", status: "completed" });

    return { envelope, report, graph };
  } finally {
    options.signal?.removeEventListener("abort", forwardCancellation);
    await rm(workRoot, { recursive: true, force: true });
  }
}

function buildProvenance(): {
  officialBaseCommit: string;
  forkCommit: string;
  nodeVersion: string;
  packageSha256: string;
} {
  return {
    officialBaseCommit: OFFICIAL_BASE_COMMIT,
    forkCommit: process.env.UNDERSTAND_HEADLESS_FORK_COMMIT ?? "development",
    nodeVersion: process.version,
    packageSha256: process.env.UNDERSTAND_HEADLESS_PACKAGE_SHA256 ?? "development",
  };
}

function validateOptions(options: GenerateKnowledgeOptions, config: HeadlessConfig): void {
  if (!/^[a-f0-9]{40,64}$/.test(options.commitSha)) {
    throw new Error("commitSha must be a full lowercase hexadecimal Git SHA");
  }
  for (const [name, value] of Object.entries(config)) {
    if (typeof value === "number" && (!Number.isInteger(value) || value <= 0)) {
      throw new Error(`${name} must be a positive integer`);
    }
  }
  if (config.maxDocuments > 100) {
    throw new Error("maxDocuments cannot exceed 100");
  }
}

function assertDisjoint(sourceRoot: string, outputRoot: string): void {
  const sourcePrefix = `${sourceRoot}${sep}`;
  const outputPrefix = `${outputRoot}${sep}`;
  if (sourceRoot === outputRoot || sourceRoot.startsWith(outputPrefix) ||
      outputRoot.startsWith(sourcePrefix)) {
    throw new Error("source and output directories must be disjoint");
  }
}

async function assertSourceFilesContained(
  sourceRoot: string,
  signal: AbortSignal,
): Promise<void> {
  let stdout: string;
  try {
    const result = await execFileAsync(
      "git",
      ["-C", sourceRoot, "ls-files", "-z", "-co", "--exclude-standard"],
      { encoding: "utf8", maxBuffer: 256 * 1024 * 1024, signal },
    );
    stdout = result.stdout;
  } catch (_error) {
    if (signal.aborted) throwIfAborted(signal);
    return;
  }
  for (const relativePath of stdout.split("\0").filter(Boolean)) {
    throwIfAborted(signal);
    const absolutePath = resolve(sourceRoot, relativePath);
    if (!absolutePath.startsWith(`${sourceRoot}${sep}`)) {
      throw new Error(`SOURCE_PATH_ESCAPE: ${relativePath}`);
    }
    const resolvedPath = await realpath(absolutePath);
    if (!resolvedPath.startsWith(`${sourceRoot}${sep}`)) {
      throw new Error(`SOURCE_SYMLINK_ESCAPE: ${relativePath}`);
    }
  }
}

export function assertEligibleStructureCoverage(
  eligibleFiles: ScannedFile[],
  structure: StructureResult,
): void {
  const structuredPaths = new Set(structure.results.map((entry) => entry.path));
  const missing = eligibleFiles
    .map((file) => file.path)
    .filter((path) => !structuredPaths.has(path))
    .sort();
  if (missing.length > 0) {
    throw new Error(
      `STRUCTURE_COVERAGE_INCOMPLETE: ${missing.length} eligible files have no ` +
      `structural result: ${missing.slice(0, 20).join(", ")}`,
    );
  }
}

function isEligibleSourceFile(
  file: ScannedFile,
  config: HeadlessConfig,
  repoType: GenerateKnowledgeOptions["repoType"],
): boolean {
  const path = `/${file.path.toLowerCase()}`;
  if (config.includePaths.length > 0 &&
      !config.includePaths.some((rule) => matchesPathRule(file.path, rule))) return false;
  if (config.excludePaths.some((rule) => matchesPathRule(file.path, rule))) return false;
  if (/(^|\/)(test|tests|__tests__|fixtures)(\/|$)/.test(path)) return false;
  if (/(^|\/)(dist|build|target|node_modules|generated)(\/|$)/.test(path)) return false;
  if (/\.(lock|min\.js|min\.css|map)$/.test(path)) return false;
  if (/\.(xlsx?|docx?|pptx?|pdf|zip|jar|war|class|png|jpe?g|gif|webp|ico|woff2?|ttf|mp[34]|mov|avi)$/.test(path)) {
    return false;
  }
  if (["code", "script", "config", "schema", "data", "infra"]
    .includes(file.fileCategory)) return true;
  if (file.fileCategory === "markup") {
    return repoType === "FRONTEND" || repoType === "DOCS";
  }
  return file.fileCategory === "docs" && repoType === "DOCS";
}

async function buildSourceContexts(
  sourceRoot: string,
  files: ScannedFile[],
  structure: StructureResult,
  config: HeadlessConfig,
  maximumContextCharacters: number,
  signal: AbortSignal,
): Promise<SourceContext[]> {
  const structureByPath = new Map(structure.results.map((entry) => [entry.path, entry]));
  let totalBytes = 0;
  const contexts: SourceContext[] = [];
  for (const file of files) {
    throwIfAborted(signal);
    const absolutePath = resolve(sourceRoot, file.path);
    if (!absolutePath.startsWith(`${sourceRoot}${sep}`)) {
      throw new Error(`source path escaped project root: ${file.path}`);
    }
    const fileStat = await stat(absolutePath);
    if (fileStat.size > config.maxSingleFileBytes) {
      throw new Error(
        `SOURCE_FILE_SIZE_EXCEEDED: ${file.path} is ${fileStat.size} bytes`);
    }
    totalBytes += fileStat.size;
    if (totalBytes > config.maxSourceBytes) {
      throw new Error(`SOURCE_SIZE_EXCEEDED: ${totalBytes} > ${config.maxSourceBytes}`);
    }
    const entry = structureByPath.get(file.path);
    if (!entry) continue;
    const content = await readFile(absolutePath, "utf8");
    throwIfAborted(signal);
    contexts.push(...splitSourceContext({
      file,
      structure: entry,
      content: redactSensitiveContent(content),
      lineStart: 1,
      lineEnd: entry.totalLines,
    }, maximumContextCharacters));
  }
  return contexts;
}

function splitSourceContext(
  context: SourceContext,
  maximumCharacters: number,
): SourceContext[] {
  if (serializeContext(context).length <= maximumCharacters) return [context];

  const lines = context.content.split("\n");
  const chunks: SourceContext[] = [];
  let startIndex = 0;
  while (startIndex < lines.length) {
    let endIndex = startIndex;
    let accepted: SourceContext | undefined;
    while (endIndex < lines.length) {
      const candidate: SourceContext = {
        ...context,
        content: lines.slice(startIndex, endIndex + 1).join("\n"),
        lineStart: startIndex + 1,
        lineEnd: endIndex + 1,
      };
      if (serializeContext(candidate).length > maximumCharacters) break;
      accepted = candidate;
      endIndex += 1;
    }
    if (!accepted) {
      throw new Error(
        `SOURCE_LINE_LIMIT_EXCEEDED: ${context.file.path}:${startIndex + 1}`);
    }
    chunks.push(accepted);
    startIndex = accepted.lineEnd;
  }
  return chunks;
}

function buildBatches(contexts: SourceContext[], maximumCharacters: number): SourceContext[][] {
  const batches: SourceContext[][] = [];
  let current: SourceContext[] = [];
  let currentSize = 0;
  for (const context of contexts) {
    const size = serializeContext(context).length;
    if (size > maximumCharacters) {
      throw new Error(`BATCH_INPUT_LIMIT_EXCEEDED: ${context.file.path}`);
    }
    if (current.length > 0 && currentSize + size > maximumCharacters) {
      batches.push(current);
      current = [];
      currentSize = 0;
    }
    current.push(context);
    currentSize += size;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

function buildBatchesFromPlan(
  contexts: SourceContext[],
  plan: ComputedBatchPlan,
  maximumCharacters: number,
): SourceContext[][] {
  const contextsByPath = new Map<string, SourceContext[]>();
  for (const context of contexts) {
    const entries = contextsByPath.get(context.file.path) ?? [];
    entries.push(context);
    contextsByPath.set(context.file.path, entries);
  }
  const plannedPaths = new Set<string>();
  const result: SourceContext[][] = [];
  for (const plannedBatch of [...plan.batches].sort((left, right) =>
    left.batchIndex - right.batchIndex)) {
    const plannedContexts: SourceContext[] = [];
    for (const file of plannedBatch.files) {
      plannedPaths.add(file.path);
      plannedContexts.push(...(contextsByPath.get(file.path) ?? []));
    }
    result.push(...buildBatches(plannedContexts, maximumCharacters));
  }
  const unplannedContexts = contexts.filter((context) =>
    !plannedPaths.has(context.file.path));
  result.push(...buildBatches(unplannedContexts, maximumCharacters));
  return result;
}

function buildSemanticSystemPrompt(): string {
  return [
    "You produce a strict source-backed semantic graph for one bounded code batch.",
    "Return exactly one semantic file record for every provided FILE and only those files.",
    "Summaries and relationships must describe behavior visible in the supplied source.",
    "Use only supplied project-relative paths and extracted symbol names.",
    "Do not generate knowledge documents, exclusions, commands, or filesystem requests.",
  ].join("\n");
}

function buildSemanticBatchPrompt(
  options: GenerateKnowledgeOptions,
  batch: SourceContext[],
  language: string,
): string {
  const schema = {
    files: [{
      path: "project/relative/path",
      summary: "behavioral summary",
      tags: ["api", "validation"],
      complexity: "simple|moderate|complex",
      symbols: [{ name: "exact extracted symbol", summary: "behavioral summary" }],
    }],
    relationships: [{
      sourcePath: "project/relative/path",
      targetPath: "project/relative/path",
      type: "calls|validates|configures|routes|depends_on|related|transforms|triggers",
      description: "source-backed relationship",
    }],
    warnings: ["string"],
  };
  return [
    `Project: ${options.projectKey}`,
    `Repository type: ${options.repoType}`,
    `Authoritative branch: master`,
    `Commit: ${options.commitSha}`,
    `Write summaries in ${language}; preserve symbols, paths, API routes, keys, and status values.`,
    `Required JSON shape: ${JSON.stringify(schema)}`,
    ...batch.map(serializeContext),
  ].join("\n\n");
}

async function completeValidatedSemanticBatch(
  llm: JsonLlmClient,
  system: string,
  prompt: string,
  batch: SourceContext[],
  signal: AbortSignal,
  maximumPromptCharacters: number,
): Promise<ValidatedSemanticBatchResult> {
  const schema = buildScopedSemanticBatchSchema(batch);
  const firstCompletion = await llm.completeJson({ system, prompt, signal });
  const firstValue = sanitizeSemanticBatchValue(firstCompletion.value, batch);
  const first = schema.safeParse(firstValue);
  if (first.success) {
    return {
      response: first.data,
      promptTokens: firstCompletion.usage.promptTokens,
      completionTokens: firstCompletion.usage.completionTokens,
      requestCount: 1,
      schemaRepairCount: 0,
    };
  }
  const repairPrompt = buildRepairPrompt(
    "semantic graph",
    first.error.issues,
    prompt,
    maximumPromptCharacters,
  );
  const repairedCompletion = await llm.completeJson({
    system,
    signal,
    prompt: repairPrompt,
  });
  const repairedValue = sanitizeSemanticBatchValue(repairedCompletion.value, batch);
  const repaired = schema.safeParse(repairedValue);
  if (!repaired.success) {
    throw new Error(
      `SEMANTIC_GRAPH_SCHEMA_REPAIR_EXHAUSTED: ${JSON.stringify(repaired.error.issues)}`,
    );
  }
  return {
    response: repaired.data,
    promptTokens: firstCompletion.usage.promptTokens + repairedCompletion.usage.promptTokens,
    completionTokens: firstCompletion.usage.completionTokens +
      repairedCompletion.usage.completionTokens,
    requestCount: 2,
    schemaRepairCount: 1,
  };
}

function sanitizeSemanticBatchValue(
  value: Record<string, unknown>,
  batch: SourceContext[],
): Record<string, unknown> {
  const allowedPaths = new Set(batch.map((context) => context.file.path));
  const allowedSymbols = buildAllowedSemanticSymbols(batch);
  let droppedSymbols = 0;
  const files = Array.isArray(value.files) ? value.files.map((file) => {
    if (!file || typeof file !== "object") return file;
    const candidate = file as Record<string, unknown>;
    if (typeof candidate.path !== "string" || !Array.isArray(candidate.symbols)) return file;
    const symbolsForFile = allowedSymbols.get(candidate.path);
    if (!symbolsForFile) return file;
    const symbols = candidate.symbols.filter((symbol) => {
      if (!symbol || typeof symbol !== "object") return true;
      const name = (symbol as Record<string, unknown>).name;
      if (typeof name !== "string" || symbolsForFile.has(name)) return true;
      droppedSymbols += 1;
      return false;
    });
    return { ...candidate, symbols };
  }) : value.files;
  let droppedRelationships = 0;
  const relationships = Array.isArray(value.relationships) ?
    value.relationships.filter((relationship) => {
    if (!relationship || typeof relationship !== "object") return true;
    const candidate = relationship as Record<string, unknown>;
    const keep = typeof candidate.sourcePath !== "string" ||
      typeof candidate.targetPath !== "string" ||
      allowedPaths.has(candidate.sourcePath) && allowedPaths.has(candidate.targetPath);
    if (!keep) droppedRelationships += 1;
    return keep;
  }) : value.relationships;
  if (droppedSymbols === 0 && droppedRelationships === 0) return value;
  const warnings = Array.isArray(value.warnings) ? value.warnings : [];
  return {
    ...value,
    files,
    relationships,
    warnings: [
      ...warnings,
      ...(droppedRelationships > 0 ? [
        `dropped ${droppedRelationships} semantic relationship(s) outside the current batch`,
      ] : []),
      ...(droppedSymbols > 0 ? [
        `dropped ${droppedSymbols} semantic symbol(s) not found by structural extraction`,
      ] : []),
    ],
  };
}

function buildAllowedSemanticSymbols(batch: SourceContext[]): Map<string, Set<string>> {
  const allowedSymbols = new Map<string, Set<string>>();
  for (const context of batch) {
    const symbols = allowedSymbols.get(context.file.path) ?? new Set<string>();
    for (const fn of context.structure.functions ?? []) symbols.add(fn.name);
    for (const cls of context.structure.classes ?? []) {
      symbols.add(cls.name);
      for (const method of cls.methods ?? []) {
        const name = typeof method === "string" ? method : method.name;
        if (name) symbols.add(name);
      }
    }
    allowedSymbols.set(context.file.path, symbols);
  }
  return allowedSymbols;
}

function buildScopedSemanticBatchSchema(batch: SourceContext[]): typeof SemanticBatchResponseSchema {
  const expectedPaths = [...new Set(batch.map((context) => context.file.path))].sort();
  const allowedSymbols = buildAllowedSemanticSymbols(batch);
  const expectedSet = new Set(expectedPaths);
  return SemanticBatchResponseSchema.superRefine((response, validation) => {
    const actualPaths = response.files.map((file) => file.path).sort();
    if (actualPaths.length !== expectedPaths.length ||
        actualPaths.some((path, index) => path !== expectedPaths[index])) {
      validation.addIssue({
        code: "custom",
        path: ["files"],
        message: "semantic files must cover exactly the current batch paths",
      });
    }
    for (const [fileIndex, file] of response.files.entries()) {
      const symbols = allowedSymbols.get(file.path);
      if (!symbols) continue;
      for (const [symbolIndex, symbol] of file.symbols.entries()) {
        if (!symbols.has(symbol.name)) {
          validation.addIssue({
            code: "custom",
            path: ["files", fileIndex, "symbols", symbolIndex, "name"],
            message: "semantic symbol is not extracted from the current batch",
          });
        }
      }
    }
    for (const [index, relationship] of response.relationships.entries()) {
      if (!expectedSet.has(relationship.sourcePath) ||
          !expectedSet.has(relationship.targetPath)) {
        validation.addIssue({
          code: "custom",
          path: ["relationships", index],
          message: "semantic relationship path is outside the current batch",
        });
      }
    }
  }) as typeof SemanticBatchResponseSchema;
}

function buildFunctionalCandidates(
  batches: SourceContext[][],
  semanticResponses: SemanticBatchResponse[],
): FunctionalCandidate[] {
  if (batches.length !== semanticResponses.length) {
    throw new Error("SEMANTIC_CANDIDATE_BATCH_MISMATCH");
  }
  return batches.map((batch, index) => {
    const semantic = semanticResponses[index] as SemanticBatchResponse;
    const paths = [...new Set(batch.map((context) => context.file.path))].sort();
    const anchors = semantic.files
      .filter((file) => isFunctionalAnchor(file.path, file.tags))
      .map((file) => file.path)
      .sort();
    const anchorPaths = anchors.length > 0 ? anchors : paths.slice(0, 1);
    return {
      id: createHash("sha256").update(paths.join("\n")).digest("hex").slice(0, 16),
      anchorPaths,
      batch,
      semantic,
    };
  });
}

function isFunctionalAnchor(path: string, tags: string[]): boolean {
  const normalized = path.toLowerCase();
  return /(?:page|controller|service|handler|endpoint|api|config|permission|validator|exception)/
    .test(normalized) || tags.some((tag) =>
    /(?:page|api|flow|business|validation|permission|config|exception|endpoint)/i.test(tag));
}

function buildSystemPrompt(): string {
  return [
    "You generate source-backed functional code knowledge as strict JSON.",
    "Return an object with documents, exclusions, and warnings.",
    "Documents explain user-visible behavior, business rules, feature flows, APIs, configuration, or exception paths.",
    "Do not emit generic inventories, plain accessors, mechanical DTO access, non-business utilities, generated output, duplicate wrappers, test-only helpers, or internal methods fully covered by a higher-level document.",
    "Use only the provided project-relative paths, symbols, and line ranges.",
    "Every document requires 2-5 natural-language questionPatterns and at least one sourceRef.",
    "Mark synthesis as INFERRED; use DIRECT only when the cited code directly supports the conclusion.",
    "Do not include stableKey or repository overview; the caller creates them.",
  ].join("\n");
}

function buildBatchPrompt(
  options: GenerateKnowledgeOptions,
  candidate: FunctionalCandidate,
  documentBudget: number,
  language: string,
): string {
  const schema = {
    documents: [{
      knowledgeType: "BUSINESS_RULE|FEATURE_FLOW|PAGE_INTERACTION|API_CONTRACT|CONFIGURATION|EXCEPTION_PATH|SYMBOL_REFERENCE",
      facetKey: "stable.semantic.facet",
      title: "string",
      questionPatterns: ["string", "string"],
      conclusion: "string",
      applicability: "string",
      details: ["string"],
      exceptions: ["string"],
      sourceRefs: [{ path: "relative/path", symbol: "symbol", lineStart: 1, lineEnd: 2 }],
      keywords: ["string"],
      evidenceType: "DIRECT|INFERRED",
    }],
    exclusions: [{
      path: "relative/path",
      symbol: "optional",
      reason: "PLAIN_ACCESSOR|MECHANICAL_DTO_ACCESS|NON_BUSINESS_UTILITY|GENERATED_OR_BUILD_OUTPUT|DUPLICATE_WRAPPER|TEST_ONLY_HELPER|COVERED_BY_HIGHER_LEVEL_DOCUMENT",
    }],
    warnings: ["string"],
  };
  return [
    `Project: ${options.projectKey}`,
    `Repository type: ${options.repoType}`,
    `Authoritative branch: master`,
    `Commit: ${options.commitSha}`,
    `Write all user-facing fields in ${language}; preserve code symbols, API paths, configuration keys, and status values exactly.`,
    `This batch may return at most ${documentBudget} documents. Return an empty documents array when no high-value functional knowledge fits the budget.`,
    "Every FILE in this batch must be accounted for: cite it from at least one document or emit one exclusion using an allowed reason. Do not leave files unclassified.",
    `Functional candidate: ${candidate.id}`,
    `Candidate anchors: ${candidate.anchorPaths.join(", ")}`,
    `Validated semantic files: ${JSON.stringify(candidate.semantic.files)}`,
    `Validated semantic relationships: ${JSON.stringify(candidate.semantic.relationships)}`,
    `Required JSON shape: ${JSON.stringify(schema)}`,
    ...candidate.batch.map(serializeContext),
  ].join("\n\n");
}

function matchesPathRule(filePath: string, rawRule: string): boolean {
  const normalizedPath = filePath.replaceAll("\\", "/");
  const normalizedRule = rawRule.replace(/^\.\//, "").replaceAll("\\", "/");
  const escaped = normalizedRule.replace(/[.+?^${}()|[\]\\]/g, "\\$&")
    .replaceAll("**", "\u0000")
    .replaceAll("*", "[^/]*")
    .replaceAll("\u0000", ".*");
  return new RegExp(`^${escaped}(?:/.*)?$`).test(normalizedPath);
}

async function completeValidatedBatch(
  llm: JsonLlmClient,
  system: string,
  prompt: string,
  batch: SourceContext[],
  signal: AbortSignal,
  maximumPromptCharacters: number,
  structureByPath: Map<string, StructureEntry>,
  sourceRoot: string,
  sourceFileCache: Map<string, string[]>,
  documentBudget: number,
  config: HeadlessConfig,
): Promise<ValidatedBatchResult> {
  throwIfAborted(signal);
  const scopedSchema = buildScopedBatchResponseSchema(batch, documentBudget);
  const firstCompletion = await llm.completeJson({ system, prompt, signal });
  const firstValue = firstCompletion.value;
  const first = scopedSchema.safeParse(firstValue);
  if (first.success) {
    try {
      await validateBatchResponseEvidence(
        first.data, structureByPath, sourceRoot, sourceFileCache, config);
      return {
        response: first.data,
        promptTokens: firstCompletion.usage.promptTokens,
        completionTokens: firstCompletion.usage.completionTokens,
        requestCount: 1,
        schemaRepairCount: 0,
      };
    } catch (error) {
      return repairBatchResponse(
        llm,
        system,
        prompt,
        batch,
        signal,
        maximumPromptCharacters,
        structureByPath,
        sourceRoot,
        sourceFileCache,
        documentBudget,
        config,
        firstCompletion,
        [{ message: error instanceof Error ? error.message : String(error) }],
      );
    }
  }

  return repairBatchResponse(
    llm,
    system,
    prompt,
    batch,
    signal,
    maximumPromptCharacters,
    structureByPath,
    sourceRoot,
    sourceFileCache,
    documentBudget,
    config,
    firstCompletion,
    first.error.issues,
  );
}

async function repairBatchResponse(
  llm: JsonLlmClient,
  system: string,
  prompt: string,
  batch: SourceContext[],
  signal: AbortSignal,
  maximumPromptCharacters: number,
  structureByPath: Map<string, StructureEntry>,
  sourceRoot: string,
  sourceFileCache: Map<string, string[]>,
  documentBudget: number,
  config: HeadlessConfig,
  firstCompletion: JsonCompletionResult,
  issues: unknown,
): Promise<ValidatedBatchResult> {
  const scopedSchema = buildScopedBatchResponseSchema(batch, documentBudget);
  let currentIssues = issues;
  let promptTokens = firstCompletion.usage.promptTokens;
  let completionTokens = firstCompletion.usage.completionTokens;
  for (let repairAttempt = 1; repairAttempt <= 2; repairAttempt += 1) {
    throwIfAborted(signal);
    const repairPrompt = buildRepairPrompt(
      "knowledge document",
      currentIssues,
      prompt,
      maximumPromptCharacters,
    );
    const completion = await llm.completeJson({
      system,
      prompt: repairPrompt,
      signal,
    });
    promptTokens += completion.usage.promptTokens;
    completionTokens += completion.usage.completionTokens;
    const repaired = scopedSchema.safeParse(completion.value);
    if (!repaired.success) {
      currentIssues = repaired.error.issues;
      continue;
    }
    try {
      await validateBatchResponseEvidence(
        repaired.data, structureByPath, sourceRoot, sourceFileCache, config);
      return {
        response: repaired.data,
        promptTokens,
        completionTokens,
        requestCount: repairAttempt + 1,
        schemaRepairCount: repairAttempt,
      };
    } catch (error) {
      currentIssues = [{
        message: error instanceof Error ? error.message : String(error),
      }];
    }
  }
  throw new Error(`LLM_SCHEMA_REPAIR_EXHAUSTED: ${JSON.stringify(currentIssues)}`);
}

async function validateBatchResponseEvidence(
  response: z.infer<typeof BatchResponseSchema>,
  structureByPath: Map<string, StructureEntry>,
  sourceRoot: string,
  sourceFileCache: Map<string, string[]>,
  config: HeadlessConfig,
): Promise<void> {
  for (const document of response.documents) {
    await validateSourceReferences(
      document, structureByPath, sourceRoot, sourceFileCache);
  }
  validateDocumentSizes(response.documents.map((document) => ({
    ...document,
    stableKey: deriveStableKey(document),
  })), config);
}

function buildRepairPrompt(
  phase: string,
  issues: unknown,
  originalPrompt: string,
  maximumCharacters: number,
): string {
  const prefix = [
    `Your previous ${phase} JSON did not match the required schema.`,
    "Return a corrected JSON object only. Do not add paths, symbols, or source claims.",
    "Validation errors: ",
  ].join("\n");
  const suffix = `\nOriginal task and source context:\n${originalPrompt}`;
  const availableDiagnostics = maximumCharacters - prefix.length - suffix.length;
  if (availableDiagnostics < 0) {
    throw new Error(
      `BATCH_INPUT_LIMIT_EXCEEDED: ${phase} repair prompt exceeds ` +
      `${maximumCharacters} characters`,
    );
  }
  const diagnostics = JSON.stringify(issues).slice(0, availableDiagnostics);
  const repairPrompt = `${prefix}${diagnostics}${suffix}`;
  assertPromptSize(repairPrompt, maximumCharacters, `${phase} repair`);
  return repairPrompt;
}

function buildScopedBatchResponseSchema(
  batch: SourceContext[],
  documentBudget: number,
): typeof BatchResponseSchema {
  const rangesByPath = new Map<string, Array<[number, number]>>();
  for (const context of batch) {
    const ranges = rangesByPath.get(context.file.path) ?? [];
    ranges.push([context.lineStart, context.lineEnd]);
    rangesByPath.set(context.file.path, ranges);
  }
  return BatchResponseSchema.superRefine((response, validation) => {
    if (response.documents.length > documentBudget) {
      validation.addIssue({
        code: "custom",
        path: ["documents"],
        message: `batch document count exceeds assigned budget ${documentBudget}`,
      });
    }
    for (const [documentIndex, document] of response.documents.entries()) {
      for (const [referenceIndex, reference] of document.sourceRefs.entries()) {
        const visibleRanges = rangesByPath.get(reference.path);
        const isVisible = visibleRanges?.some(([start, end]) =>
          reference.lineStart >= start && reference.lineEnd <= end) ?? false;
        if (!isVisible) {
          validation.addIssue({
            code: "custom",
            path: ["documents", documentIndex, "sourceRefs", referenceIndex],
            message: "source reference is outside the current model batch",
          });
        }
      }
    }
    for (const [exclusionIndex, exclusion] of response.exclusions.entries()) {
      if (!rangesByPath.has(exclusion.path)) {
        validation.addIssue({
          code: "custom",
          path: ["exclusions", exclusionIndex, "path"],
          message: "exclusion path is outside the current model batch",
        });
      }
    }
    const accountedPaths = new Set([
      ...response.documents.flatMap((document) =>
        document.sourceRefs.map((reference) => reference.path)),
      ...response.exclusions.map((exclusion) => exclusion.path),
    ]);
    for (const path of rangesByPath.keys()) {
      if (!accountedPaths.has(path)) {
        validation.addIssue({
          code: "custom",
          path: ["documents"],
          message: `batch source path is neither cited nor excluded: ${path}`,
        });
      }
    }
  }) as typeof BatchResponseSchema;
}

function allocateDocumentBudgets(batchCount: number, totalBudget: number): number[] {
  if (batchCount === 0) return [];
  const base = Math.floor(totalBudget / batchCount);
  const remainder = totalBudget % batchCount;
  return Array.from({ length: batchCount }, (_, index) =>
    base + (index < remainder ? 1 : 0));
}

function serializeContext(context: SourceContext): string {
  const compactStructure = {
    path: context.structure.path,
    language: context.structure.language,
    totalLines: context.structure.totalLines,
    sourceRange: [context.lineStart, context.lineEnd],
    functions: (context.structure.functions ?? []).filter((fn) =>
      rangesOverlap(context.lineStart, context.lineEnd, fn.startLine, fn.endLine)),
    classes: (context.structure.classes ?? [])
      .filter((cls) => rangesOverlap(
        context.lineStart, context.lineEnd, cls.startLine, cls.endLine))
      .map((cls) => ({ name: cls.name, startLine: cls.startLine, endLine: cls.endLine })),
  };
  const numberedSource = context.content.split("\n")
    .map((line, index) => `${String(context.lineStart + index).padStart(6, "0")} | ${line}`)
    .join("\n");
  return [
    `--- FILE ${context.file.path} ---`,
    `language=${context.file.language} category=${context.file.fileCategory}`,
    `structure=${JSON.stringify(compactStructure)}`,
    `SOURCE LINES ${context.lineStart}-${context.lineEnd}:`,
    numberedSource,
  ].join("\n");
}

function rangesOverlap(
  leftStart: number,
  leftEnd: number,
  rightStart: number,
  rightEnd: number,
): boolean {
  return leftStart <= rightEnd && rightStart <= leftEnd;
}

function assertPromptSize(prompt: string, maximumCharacters: number, phase: string): void {
  if (prompt.length > maximumCharacters) {
    throw new Error(
      `BATCH_INPUT_LIMIT_EXCEEDED: ${phase} prompt has ${prompt.length} characters > ` +
      maximumCharacters,
    );
  }
}

function redactSensitiveContent(content: string): string {
  let insidePrivateKey = false;
  return content.replaceAll("\r\n", "\n").replaceAll("\r", "\n").split("\n").map((line) => {
    if (/-----BEGIN [^-]*PRIVATE KEY[^-]*-----/.test(line)) {
      insidePrivateKey = true;
      return "[REDACTED PRIVATE KEY]";
    }
    if (insidePrivateKey) {
      if (/-----END [^-]*PRIVATE KEY[^-]*-----/.test(line)) {
        insidePrivateKey = false;
      }
      return "[REDACTED PRIVATE KEY]";
    }
    const assignment = line.match(
      /^(\s*["']?[^:=\s][^:=]*(?:password|passwd|secret|token|api[-_.]?key|access[-_.]?key|private[-_.]?key)[^:=]*["']?\s*[:=]\s*)(.*)$/i,
    );
    if (assignment) return `${assignment[1]}[REDACTED]`;
    return line
      .replace(/(\bBearer\s+)[A-Za-z0-9._~+/-]{8,}={0,2}/gi, "$1[REDACTED]")
      .replace(/(\bBasic\s+)[A-Za-z0-9+/]{8,}={0,2}/gi, "$1[REDACTED]")
      .replace(
        /(\b[a-z][a-z0-9+.-]*:\/\/[^\s/:@]+:)[^\s/@]+(@)/gi,
        "$1[REDACTED]$2",
      )
      .replace(
        /\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b/g,
        "[REDACTED JWT]",
      )
      .replace(/\b(?:glpat-|gh[pousr]_)[A-Za-z0-9_-]{12,}\b/g, "[REDACTED TOKEN]")
      .replace(
        /([?&;,]\s*(?:password|passwd|token|secret|api[_-]?key)=)[^\s&;,]+/gi,
        "$1[REDACTED]",
      );
  }).join("\n");
}

async function validateSourceReferences(
  document: z.infer<typeof KnowledgeDocumentSchema>,
  structureByPath: Map<string, StructureEntry>,
  sourceRoot: string,
  sourceFileCache: Map<string, string[]>,
): Promise<void> {
  for (const ref of document.sourceRefs) {
    const structure = structureByPath.get(ref.path);
    if (!structure) {
      throw new Error(
        `source reference is not in the scanned source set: ${ref.path}`);
    }
    if (ref.lineEnd > structure.totalLines) {
      throw new Error(
        `source reference line is outside file: ${ref.path}:${ref.lineEnd}`);
    }
    const symbolRanges = new Map<string, Array<[number, number]>>();
    const addRange = (name: string, range: [number, number]): void => {
      const ranges = symbolRanges.get(name) ?? [];
      ranges.push(range);
      symbolRanges.set(name, ranges);
    };
    for (const fn of structure.functions ?? []) {
      addRange(fn.name, [fn.startLine, fn.endLine]);
    }
    for (const cls of structure.classes ?? []) {
      addRange(cls.name, [cls.startLine, cls.endLine]);
      for (const method of cls.methods ?? []) {
        const name = typeof method === "string" ? method : method.name;
        if (name && !symbolRanges.has(name)) {
          addRange(name, [cls.startLine, cls.endLine]);
        }
      }
    }
    if (ref.symbol === "file") {
      throw new Error("source reference symbol 'file' is reserved for the repository overview");
    }
    if (structure.fileCategory !== "code" && structure.fileCategory !== "script" &&
        ref.symbol === basename(ref.path)) {
      continue;
    }
    const symbolCandidates = sourceSymbolCandidates(ref.symbol, symbolRanges);
    const extractedMatch = symbolCandidates.some((candidate) =>
      symbolRanges.get(candidate)?.some((range) => rangesOverlap(
        ref.lineStart, ref.lineEnd, range[0], range[1])));
    if (extractedMatch) continue;

    let sourceLines = sourceFileCache.get(ref.path);
    if (!sourceLines) {
      sourceLines = (await readFile(resolve(sourceRoot, ref.path), "utf8")).split("\n");
      sourceFileCache.set(ref.path, sourceLines);
    }
    const citedSource = sourceLines.slice(ref.lineStart - 1, ref.lineEnd).join("\n");
    if (symbolCandidates.some((candidate) => citedSource.includes(candidate))) continue;
    const yamlPathMatch = /\.ya?ml$/i.test(ref.path) &&
      symbolCandidates.some((candidate) => {
        const range = findYamlPathRange(sourceLines, candidate);
        return range !== undefined && rangesOverlap(
          ref.lineStart, ref.lineEnd, range[0], range[1]);
      });
    if (yamlPathMatch) continue;
    const xmlElementMatch = ref.path.toLowerCase().endsWith(".xml") &&
      symbolCandidates.some((candidate) => {
        const range = findNamedXmlElementRange(sourceLines, candidate);
        return range !== undefined && rangesOverlap(
          ref.lineStart, ref.lineEnd, range[0], range[1]);
      });
    if (xmlElementMatch) continue;
    throw new Error(
      `source reference symbol was not found in cited lines: ` +
      `${ref.path}#${ref.symbol}:${ref.lineStart}-${ref.lineEnd}`);
  }
}

function findYamlPathRange(
  lines: string[],
  dottedPath: string,
): [number, number] | undefined {
  const expected = dottedPath.split(".").map((part) => part.trim()).filter(Boolean);
  if (expected.length === 0) return undefined;
  const stack: Array<{ indent: number; key: string; line: number }> = [];
  for (let index = 0; index < lines.length; index += 1) {
    const raw = lines[index] ?? "";
    if (/^\s*(?:#|$)/.test(raw)) continue;
    const match = raw.match(/^(\s*)(?:-\s+)?(["']?)([^:"']+?)\2\s*:\s*(?:.*)$/);
    if (!match) continue;
    const indent = (match[1] ?? "").replaceAll("\t", "  ").length;
    const key = (match[3] ?? "").trim();
    while (stack.length > 0 && (stack.at(-1)?.indent ?? -1) >= indent) stack.pop();
    stack.push({ indent, key, line: index + 1 });
    const path = stack.map((item) => item.key);
    if (path.length !== expected.length ||
        !path.every((part, pathIndex) => part === expected[pathIndex])) continue;
    let endLine = lines.length;
    for (let end = index + 1; end < lines.length; end += 1) {
      const next = lines[end] ?? "";
      if (/^\s*(?:#|$)/.test(next)) continue;
      const nextIndent = (next.match(/^\s*/)?.[0] ?? "").replaceAll("\t", "  ").length;
      if (nextIndent <= indent) {
        endLine = end;
        break;
      }
    }
    return [stack[0]?.line ?? index + 1, Math.max(index + 1, endLine)];
  }
  return undefined;
}

function findNamedXmlElementRange(
  lines: string[],
  symbol: string,
): [number, number] | undefined {
  const escapedSymbol = symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const idPattern = new RegExp(`\\bid\\s*=\\s*["']${escapedSymbol}["']`);
  for (let index = 0; index < lines.length; index += 1) {
    const opening = lines[index]?.match(
      /<(select|insert|update|delete|sql|resultMap)\b/i);
    if (!opening) continue;
    let declaration = lines[index] ?? "";
    let declarationEnd = index;
    while (!declaration.includes(">") && declarationEnd + 1 < lines.length &&
      declarationEnd - index < 10) {
      declarationEnd += 1;
      declaration += `\n${lines[declarationEnd] ?? ""}`;
    }
    if (!idPattern.test(declaration)) continue;
    const tag = opening[1] as string;
    const closingPattern = new RegExp(`</${tag}\\s*>`, "i");
    for (let end = declarationEnd; end < lines.length; end += 1) {
      if (closingPattern.test(lines[end] ?? "")) return [index + 1, end + 1];
    }
    return [index + 1, declarationEnd + 1];
  }
  return undefined;
}

function sourceSymbolCandidates(
  rawSymbol: string,
  extractedSymbols: Map<string, Array<[number, number]>>,
): string[] {
  const candidates = new Set([rawSymbol]);
  const withoutSignature = rawSymbol.replace(/\s*\([^)]*\)\s*$/, "");
  candidates.add(withoutSignature);
  const parts = withoutSignature.split(/::|[.#]/).filter(Boolean);
  if (parts.length > 1) {
    const owner = parts.at(-2);
    const leaf = parts.at(-1);
    if (owner && leaf && extractedSymbols.has(owner)) candidates.add(leaf);
  }
  return [...candidates];
}

export function deduplicateDocuments(
  documents: GeneratedKnowledgeDocument[],
): GeneratedKnowledgeDocument[] {
  const byKey = new Map<string, GeneratedKnowledgeDocument>();
  for (const document of documents) {
    const existing = byKey.get(document.stableKey);
    if (!existing) {
      byKey.set(document.stableKey, document);
      continue;
    }
    if (serializeJson(existing) !== serializeJson(document)) {
      throw new Error(`DUPLICATE_DOCUMENT_CONFLICT: ${document.stableKey}`);
    }
  }
  return [...byKey.values()].sort((left, right) =>
    left.stableKey.localeCompare(right.stableKey));
}

function validateDocumentSizes(
  documents: GeneratedKnowledgeDocument[],
  config: HeadlessConfig,
): void {
  for (const document of documents) {
    const body = [
      document.title,
      ...document.questionPatterns,
      document.conclusion,
      document.applicability,
      ...document.details,
      ...document.exceptions,
      ...document.sourceRefs.map((reference) =>
        `${reference.path}#${reference.symbol}:${reference.lineStart}-${reference.lineEnd}`),
      ...document.keywords,
    ].join("\n");
    if (body.length > config.maxDocumentCharacters) {
      throw new Error(
        `DOCUMENT_CONTENT_TOO_LARGE: ${document.stableKey} has ${body.length} characters`);
    }
    const serializedBytes = new TextEncoder().encode(JSON.stringify(document)).byteLength;
    if (serializedBytes > config.maxDocumentBytes) {
      throw new Error(
        `DOCUMENT_BYTES_TOO_LARGE: ${document.stableKey} has ${serializedBytes} bytes`);
    }
  }
}

function buildOverview(
  options: GenerateKnowledgeOptions,
  scan: ScanResult,
  structure: StructureResult,
): GeneratedKnowledgeDocument {
  const sortedEntries = [...structure.results].sort((left, right) =>
    left.path.localeCompare(right.path));
  const evidence = sortedEntries.map((entry) => {
    const cls = entry.classes?.[0];
    if (cls) {
      return {
        path: entry.path,
        symbol: cls.name,
        lineStart: cls.startLine,
        lineEnd: Math.min(cls.endLine, cls.startLine + 19),
      };
    }
    const fn = entry.functions?.[0];
    if (fn) {
      return {
        path: entry.path,
        symbol: fn.name,
        lineStart: fn.startLine,
        lineEnd: Math.min(fn.endLine, fn.startLine + 19),
      };
    }
    if (entry.fileCategory !== "code" && entry.fileCategory !== "script") {
      return {
        path: entry.path,
        symbol: basename(entry.path),
        lineStart: 1,
        lineEnd: Math.max(1, Math.min(entry.totalLines, 20)),
      };
    }
    return undefined;
  }).find((item) => item !== undefined);
  if (!evidence) throw new Error("NO_ANALYZABLE_SOURCE_FILES");
  const document = KnowledgeDocumentSchema.parse({
    knowledgeType: "REPOSITORY_OVERVIEW",
    facetKey: "repository.overview",
    title: `${options.projectKey} 仓库概览`,
    questionPatterns: [
      `${options.projectKey} 项目包含哪些代码？`,
      `${options.projectKey} 当前 master 知识对应哪个版本？`,
    ],
    conclusion: `${options.projectKey} 的代码知识来自 master ${options.commitSha}。`,
    applicability: `仅适用于 ${options.projectKey} 的 ${options.repoType} 仓库当前 master。`,
    details: [
      `扫描 ${scan.totalFiles} 个文件，完成 ${structure.filesAnalyzed} 个文件的结构分析。`,
      `主要语言：${Object.entries(scan.stats.byLanguage)
        .sort((left, right) => right[1] - left[1])
        .slice(0, 5)
        .map(([language, count]) => `${language}(${count})`).join("、") || "unknown"}。`,
      `readinessMarker=project-code-ready:v1:${options.sourceId}:${options.commitSha}`,
    ],
    exceptions: ["不代表 feature branch、tag 或未合入 master 的代码。"],
    sourceRefs: [evidence],
    keywords: [options.projectKey, options.repoType, "master", options.commitSha],
    evidenceType: "DIRECT",
  });
  return { ...document, stableKey: deriveStableKey(document) };
}

function buildSemanticGraph(
  options: GenerateKnowledgeOptions,
  scan: ScanResult,
  structure: StructureResult,
  imports: ImportResult,
  semanticResponses: SemanticBatchResponse[],
): KnowledgeGraph {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeIds = new Set<string>();
  for (const entry of structure.results) {
    const fileId = `file:${entry.path}`;
    nodeIds.add(fileId);
    nodes.push({
      id: fileId,
      type: "file",
      name: basename(entry.path),
      filePath: entry.path,
      summary: `Structural source file with ${entry.totalLines} lines`,
      tags: [entry.language, entry.fileCategory],
      complexity: "moderate",
    });
    for (const cls of entry.classes ?? []) {
      const id = `class:${entry.path}:${cls.name}:${cls.startLine}-${cls.endLine}`;
      nodeIds.add(id);
      nodes.push({
        id,
        type: "class",
        name: cls.name,
        filePath: entry.path,
        lineRange: [cls.startLine, cls.endLine],
        summary: `Class ${cls.name}`,
        tags: [],
        complexity: "moderate",
      });
      edges.push({
        source: fileId,
        target: id,
        type: "contains",
        direction: "forward",
        weight: 1,
      });
    }
    for (const fn of entry.functions ?? []) {
      const id = `function:${entry.path}:${fn.name}:${fn.startLine}-${fn.endLine}`;
      nodeIds.add(id);
      nodes.push({
        id,
        type: "function",
        name: fn.name,
        filePath: entry.path,
        lineRange: [fn.startLine, fn.endLine],
        summary: `Function ${fn.name}`,
        tags: [],
        complexity: "moderate",
      });
      edges.push({
        source: fileId,
        target: id,
        type: "contains",
        direction: "forward",
        weight: 1,
      });
    }
  }
  for (const [source, targets] of Object.entries(imports.importMap)) {
    for (const target of targets) {
      if (!nodeIds.has(`file:${source}`) || !nodeIds.has(`file:${target}`)) continue;
      edges.push({
        source: `file:${source}`,
        target: `file:${target}`,
        type: "imports",
        direction: "forward",
        weight: 0.7,
      });
    }
  }
  const graph: KnowledgeGraph = {
    version: "1.0",
    kind: "codebase",
    project: {
      name: options.projectKey,
      languages: Object.keys(scan.stats.byLanguage).sort(),
      frameworks: [],
      description: `Structural graph for ${options.projectKey}`,
      analyzedAt: "1970-01-01T00:00:00.000Z",
      gitCommitHash: options.commitSha,
    },
    nodes,
    edges,
    layers: [],
    tour: [],
  };
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  for (const semantic of semanticResponses) {
    for (const file of semantic.files) {
      const fileNode = nodeById.get(`file:${file.path}`);
      if (!fileNode) throw new Error(`SEMANTIC_GRAPH_UNKNOWN_FILE: ${file.path}`);
      fileNode.summary = file.summary;
      fileNode.tags = [...new Set([...fileNode.tags, ...file.tags])].sort();
      fileNode.complexity = file.complexity;
      for (const symbol of file.symbols) {
        const candidates = graph.nodes.filter((node) =>
          node.filePath === file.path && node.name === symbol.name && node.type !== "file");
        for (const node of candidates) node.summary = symbol.summary;
      }
    }
    for (const relationship of semantic.relationships) {
      const edge: GraphEdge = {
        source: `file:${relationship.sourcePath}`,
        target: `file:${relationship.targetPath}`,
        type: relationship.type,
        direction: "forward",
        description: relationship.description,
        weight: 0.9,
      };
      if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) {
        throw new Error(
          `SEMANTIC_GRAPH_UNKNOWN_RELATIONSHIP: ${relationship.sourcePath} -> ` +
          relationship.targetPath,
        );
      }
      const duplicate = graph.edges.some((existing) =>
        existing.source === edge.source && existing.target === edge.target &&
        existing.type === edge.type && existing.description === edge.description);
      if (!duplicate) graph.edges.push(edge);
    }
  }
  graph.edges.sort((left, right) => [left.source, left.target, left.type, left.description ?? ""]
    .join("\n").localeCompare(
      [right.source, right.target, right.type, right.description ?? ""].join("\n")));
  const validation = validateGraph(graph);
  if (!validation.success || !validation.data) {
    throw new Error(
      `SEMANTIC_GRAPH_INVALID: ${validation.fatal ?? validation.errors?.join("; ") ?? "unknown"}`,
    );
  }
  return validation.data as KnowledgeGraph;
}

async function runScript(
  script: string,
  arguments_: string[],
  signal: AbortSignal,
): Promise<void> {
  throwIfAborted(signal);
  await new Promise<void>((resolvePromise, reject) => {
    const child = spawn(process.execPath, [script, ...arguments_], {
      stdio: ["ignore", "ignore", "pipe"],
      env: analysisScriptEnvironment(),
      signal,
      killSignal: "SIGTERM",
    });
    let stderr = "";
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      if (stderr.length < 16_384) stderr += chunk;
    });
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) resolvePromise();
      else reject(new Error(
        `official analysis script failed (${code ?? signal}): ${stderr.trim()}`));
    });
  });
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw new DOMException(
    signal.reason instanceof Error ? signal.reason.message : "operation cancelled",
    "AbortError",
  );
}

function analysisScriptEnvironment(): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {};
  for (const name of ["PATH", "HOME", "TMPDIR", "SystemRoot"]) {
    if (process.env[name]) environment[name] = process.env[name];
  }
  return environment;
}

function serializeJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function sha256Json(value: unknown): string {
  return createHash("sha256").update(serializeJson(value)).digest("hex");
}

async function writeArtifactSetAtomic(
  root: string,
  artifacts: Array<readonly [name: string, value: unknown]>,
  beforePromote?: () => void,
  signal?: AbortSignal,
): Promise<void> {
  const pending = artifacts.map(([name, value]) => ({
    name,
    value,
    temporary: join(root, `.${name}.${randomUUID()}.tmp`),
    target: join(root, name),
  }));
  const promoted: string[] = [];
  try {
    throwIfAborted(signal);
    await Promise.all(pending.map((artifact) =>
      writeAndSync(artifact.temporary, serializeJson(artifact.value))));
    throwIfAborted(signal);
    beforePromote?.();
    await Promise.all(pending.map((artifact) =>
      writeAndSync(artifact.temporary, serializeJson(artifact.value))));
    throwIfAborted(signal);
    for (const artifact of pending) {
      throwIfAborted(signal);
      await rename(artifact.temporary, artifact.target);
      promoted.push(artifact.target);
    }
    throwIfAborted(signal);
    // Windows supports syncing files but rejects fsync on directory handles.
    if (process.platform !== "win32") {
      const directory = await open(root, "r");
      try {
        await directory.sync();
      } finally {
        await directory.close();
      }
    }
  } catch (error) {
    await Promise.all([
      ...pending.map((artifact) => rm(artifact.temporary, { force: true })),
      ...promoted.map((target) => rm(target, { force: true })),
    ]);
    throw error;
  }
}

async function writeAndSync(path: string, content: string): Promise<void> {
  const handle = await open(path, "w");
  try {
    await handle.writeFile(content, "utf8");
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function mapConcurrent<T, R>(
  values: T[],
  concurrency: number,
  mapper: (value: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let next = 0;
  let stopped = false;
  async function worker(): Promise<void> {
    while (!stopped) {
      const index = next;
      next += 1;
      if (index >= values.length) return;
      try {
        results[index] = await mapper(values[index] as T, index);
      } catch (error) {
        stopped = true;
        throw error;
      }
    }
  }
  await Promise.all(Array.from(
    { length: Math.min(concurrency, Math.max(1, values.length)) },
    () => worker()));
  return results;
}
