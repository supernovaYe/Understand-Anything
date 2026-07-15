import { describe, expect, it } from "vitest";

import {
  KnowledgeDocumentSchema,
  KnowledgeDocumentsEnvelopeSchema,
  deriveStableKey,
} from "./schema.js";

const validDocument = {
  knowledgeType: "BUSINESS_RULE" as const,
  facetKey: "order.submit.permission",
  title: "Order submission permission",
  questionPatterns: [
    "Why can an order not be submitted?",
    "Which permission controls order submission?",
  ],
  conclusion: "Submission requires the ORDER_SUBMIT permission.",
  applicability: "Order submission requests handled by OrderService.",
  details: ["The service checks permission before persisting the order."],
  exceptions: ["Administrative repair does not use this entry point."],
  sourceRefs: [
    {
      path: "src/main/java/example/OrderService.java",
      symbol: "submit",
      lineStart: 12,
      lineEnd: 28,
    },
  ],
  keywords: ["order", "submit", "permission"],
  evidenceType: "DIRECT" as const,
};

describe("KnowledgeDocumentSchema", () => {
  it("accepts an answer-oriented document and derives a wording-independent stable key", () => {
    const parsed = KnowledgeDocumentSchema.parse(validDocument);
    const key = deriveStableKey(parsed);
    const reworded = KnowledgeDocumentSchema.parse({
      ...validDocument,
      title: "Can the current user submit this order?",
      conclusion: "The caller needs ORDER_SUBMIT.",
    });

    expect(key).toMatch(/^[a-f0-9]{24}$/);
    expect(deriveStableKey(reworded)).toBe(key);
  });

  it("keeps identity stable when only source line numbers move", () => {
    const parsed = KnowledgeDocumentSchema.parse(validDocument);
    const shifted = KnowledgeDocumentSchema.parse({
      ...validDocument,
      sourceRefs: validDocument.sourceRefs.map((ref) => ({
        ...ref,
        lineStart: ref.lineStart + 10,
        lineEnd: ref.lineEnd + 10,
      })),
    });

    expect(deriveStableKey(shifted)).toBe(deriveStableKey(parsed));
  });

  it("rejects documents without two natural-language question patterns", () => {
    expect(() => KnowledgeDocumentSchema.parse({
      ...validDocument,
      questionPatterns: ["Why is submission blocked?"],
    })).toThrow();
  });

  it("rejects whitespace-only answer fields", () => {
    expect(() => KnowledgeDocumentSchema.parse({
      ...validDocument,
      conclusion: "   ",
    })).toThrow();
  });
});

describe("KnowledgeDocumentsEnvelopeSchema", () => {
  it("requires exactly one repository overview and no more than 100 documents", () => {
    const overview = {
      ...validDocument,
      knowledgeType: "REPOSITORY_OVERVIEW" as const,
      facetKey: "repository.overview",
      title: "Repository overview",
    };
    const envelope = {
      schemaVersion: "1.0" as const,
      projectKey: "order-service",
      sourceId: "1001",
      repoType: "BACKEND" as const,
      branch: "master" as const,
      commitSha: "a".repeat(40),
      readinessMarker: `project-code-ready:v1:1001:${"a".repeat(40)}`,
      generator: {
        name: "understand-headless" as const,
        version: "0.1.0",
        understandAnythingVersion: "2.8.1",
        promptVersion: "1.0.0",
        modelRef: "fixture-model",
        build: {
          officialBaseCommit: "b".repeat(40),
          forkCommit: "c".repeat(40),
          nodeVersion: "v22.0.0",
          packageSha256: "d".repeat(64),
        },
      },
      documents: [
        { ...overview, stableKey: deriveStableKey(overview) },
        { ...validDocument, stableKey: deriveStableKey(validDocument) },
      ],
    };

    expect(KnowledgeDocumentsEnvelopeSchema.parse(envelope).documents).toHaveLength(2);
    expect(() => KnowledgeDocumentsEnvelopeSchema.parse({
      ...envelope,
      documents: envelope.documents.filter((document) =>
        document.knowledgeType !== "REPOSITORY_OVERVIEW"),
    })).toThrow(/REPOSITORY_OVERVIEW/);
  });
});
