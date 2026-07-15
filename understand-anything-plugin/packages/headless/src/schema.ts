import { createHash } from "node:crypto";

import { z } from "zod";

const BuildProvenanceSchema = z.object({
  officialBaseCommit: z.string().regex(/^[a-f0-9]{40,64}$/),
  forkCommit: z.union([z.string().regex(/^[a-f0-9]{40,64}$/), z.literal("development")]),
  nodeVersion: z.string().regex(/^v\d+\.\d+\.\d+$/),
  packageSha256: z.union([z.string().regex(/^[a-f0-9]{64}$/), z.literal("development")]),
});

export const KnowledgeTypeSchema = z.enum([
  "REPOSITORY_OVERVIEW",
  "BUSINESS_RULE",
  "FEATURE_FLOW",
  "PAGE_INTERACTION",
  "API_CONTRACT",
  "CONFIGURATION",
  "EXCEPTION_PATH",
  "SYMBOL_REFERENCE",
]);

export const SourceRefSchema = z.object({
  path: z.string().trim().min(1).refine((path) => !path.startsWith("/"), {
    message: "source path must be project-relative",
  }),
  symbol: z.string().trim().min(1),
  lineStart: z.number().int().positive(),
  lineEnd: z.number().int().positive(),
}).refine((ref) => ref.lineEnd >= ref.lineStart, {
  message: "source lineEnd must be greater than or equal to lineStart",
});

export const KnowledgeDocumentSchema = z.object({
  knowledgeType: KnowledgeTypeSchema,
  facetKey: z.string().trim().min(1).max(160),
  title: z.string().trim().min(1).max(240),
  questionPatterns: z.array(z.string().trim().min(1).max(500)).min(2).max(5),
  conclusion: z.string().trim().min(1).max(4_000),
  applicability: z.string().trim().min(1).max(2_000),
  details: z.array(z.string().trim().min(1).max(2_000)).min(1).max(12),
  exceptions: z.array(z.string().trim().min(1).max(2_000)).max(8),
  sourceRefs: z.array(SourceRefSchema).min(1).max(20),
  keywords: z.array(z.string().trim().min(1).max(120)).min(1).max(30),
  evidenceType: z.enum(["DIRECT", "INFERRED"]),
});

export type KnowledgeDocumentInput = z.infer<typeof KnowledgeDocumentSchema>;

export function deriveStableKey(document: KnowledgeDocumentInput): string {
  const normalizedRefs = [...new Set(document.sourceRefs
    .map((ref) => `${ref.path}#${ref.symbol}`))].sort();
  const identity = [
    document.knowledgeType,
    document.facetKey.trim().toLowerCase(),
    ...normalizedRefs,
  ].join("\n");
  return createHash("sha256").update(identity).digest("hex").slice(0, 24);
}

export const GeneratedKnowledgeDocumentSchema = KnowledgeDocumentSchema.extend({
  stableKey: z.string().regex(/^[a-f0-9]{24}$/),
}).superRefine((document, context) => {
  if (document.stableKey !== deriveStableKey(document)) {
    context.addIssue({
      code: "custom",
      path: ["stableKey"],
      message: "stableKey does not match the document identity",
    });
  }
});

export const KnowledgeDocumentsEnvelopeSchema = z.object({
  schemaVersion: z.literal("1.0"),
  projectKey: z.string().min(1).max(128),
  sourceId: z.string().min(1).max(64),
  repoType: z.enum(["FRONTEND", "BACKEND", "COMMON", "DOCS"]),
  branch: z.literal("master"),
  commitSha: z.string().regex(/^[a-f0-9]{40,64}$/),
  readinessMarker: z.string().min(1).max(128),
  generator: z.object({
    name: z.literal("understand-headless"),
    version: z.string().min(1).max(64),
    understandAnythingVersion: z.string().min(1).max(64),
    promptVersion: z.string().min(1).max(64),
    modelRef: z.string().min(1).max(160),
    build: BuildProvenanceSchema,
  }),
  documents: z.array(GeneratedKnowledgeDocumentSchema).min(1).max(100),
}).superRefine((envelope, context) => {
  const overviewCount = envelope.documents.filter((document) =>
    document.knowledgeType === "REPOSITORY_OVERVIEW").length;
  if (overviewCount !== 1) {
    context.addIssue({
      code: "custom",
      path: ["documents"],
      message: "documents must contain exactly one REPOSITORY_OVERVIEW",
    });
  }
  const keys = new Set<string>();
  for (const [index, document] of envelope.documents.entries()) {
    if (keys.has(document.stableKey)) {
      context.addIssue({
        code: "custom",
        path: ["documents", index, "stableKey"],
        message: "stableKey must be unique",
      });
    }
    keys.add(document.stableKey);
  }
});

export type GeneratedKnowledgeDocument = z.infer<typeof GeneratedKnowledgeDocumentSchema>;
export type KnowledgeDocumentsEnvelope = z.infer<typeof KnowledgeDocumentsEnvelopeSchema>;

export const ExclusionSchema = z.object({
  path: z.string().trim().min(1),
  symbol: z.string().trim().min(1).optional(),
  reason: z.enum([
    "PLAIN_ACCESSOR",
    "MECHANICAL_DTO_ACCESS",
    "NON_BUSINESS_UTILITY",
    "GENERATED_OR_BUILD_OUTPUT",
    "DUPLICATE_WRAPPER",
    "TEST_ONLY_HELPER",
    "COVERED_BY_HIGHER_LEVEL_DOCUMENT",
  ]),
});

export const GenerationReportSchema = z.object({
  schemaVersion: z.literal("1.0"),
  status: z.enum(["COMPLETED", "FAILED"]),
  sourceCommitSha: z.string().regex(/^[a-f0-9]{40,64}$/),
  generatorVersion: z.string().min(1),
  understandAnythingVersion: z.string().min(1),
  promptVersion: z.string().min(1),
  modelRef: z.string().min(1),
  build: BuildProvenanceSchema,
  scannedFiles: z.number().int().nonnegative(),
  analyzedFiles: z.number().int().nonnegative(),
  coveredSourceFiles: z.number().int().nonnegative(),
  excludedSourceFiles: z.number().int().nonnegative(),
  batchCount: z.number().int().nonnegative(),
  llmRequestCount: z.number().int().nonnegative(),
  schemaRepairCount: z.number().int().nonnegative(),
  promptTokens: z.number().int().nonnegative(),
  completionTokens: z.number().int().nonnegative(),
  functionalDocuments: z.number().int().nonnegative().max(99),
  totalDocuments: z.number().int().positive().max(100),
  documentsByType: z.record(KnowledgeTypeSchema, z.number().int().nonnegative()),
  exclusions: z.array(ExclusionSchema),
  warnings: z.array(z.string()),
  limits: z.object({
    maxSourceFiles: z.number().int().positive(),
    maxSourceBytes: z.number().int().positive(),
    maxSingleFileBytes: z.number().int().positive(),
    maxBatchInputCharacters: z.number().int().positive(),
    maxLlmPromptCharacters: z.number().int().positive(),
    maxLlmConcurrency: z.number().int().positive(),
    maxDocuments: z.number().int().positive(),
    maxDocumentCharacters: z.number().int().positive(),
    maxDocumentBytes: z.number().int().positive(),
    language: z.string().min(1),
    includePaths: z.array(z.string()),
    excludePaths: z.array(z.string()),
  }),
  artifactChecksums: z.object({
    knowledgeDocumentsSha256: z.string().regex(/^[a-f0-9]{64}$/),
    knowledgeGraphSha256: z.string().regex(/^[a-f0-9]{64}$/),
  }),
  phaseTimingsMs: z.record(z.string(), z.number().int().nonnegative()),
  generatedAt: z.string().datetime(),
});

export type GenerationReport = z.infer<typeof GenerationReportSchema>;
