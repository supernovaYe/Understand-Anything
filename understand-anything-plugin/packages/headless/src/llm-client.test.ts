import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

import { afterEach, describe, expect, it } from "vitest";

import { OpenAiCompatibleLlmClient } from "./llm-client.js";

const servers: ReturnType<typeof createServer>[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) =>
    new Promise<void>((resolve) => server.close(() => resolve()))));
});

describe("OpenAiCompatibleLlmClient", () => {
  it("rejects invalid timeout and retry settings", () => {
    expect(() => new OpenAiCompatibleLlmClient({
      baseUrl: "http://127.0.0.1/v1",
      apiKey: "test-secret",
      model: "qwen-test",
      timeoutMs: Number.NaN,
      maxRetries: 0,
    })).toThrow(/timeoutMs/);
    expect(() => new OpenAiCompatibleLlmClient({
      baseUrl: "http://127.0.0.1/v1",
      apiKey: "test-secret",
      model: "qwen-test",
      timeoutMs: 1_000,
      maxRetries: -1,
    })).toThrow(/maxRetries/);
  });

  it("sends an authenticated JSON completion request and parses object content", async () => {
    let receivedHeaders: Record<string, string | string[] | undefined> = {};
    let receivedBody: unknown;
    const server = createServer((request, response) => {
      receivedHeaders = request.headers;
      let body = "";
      request.setEncoding("utf8");
      request.on("data", (chunk) => { body += chunk; });
      request.on("end", () => {
        receivedBody = JSON.parse(body);
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({
          choices: [{ message: { content: "```json\n{\"documents\":[]}\n```" } }],
          usage: { prompt_tokens: 12, completion_tokens: 3 },
        }));
      });
    });
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;

    const client = new OpenAiCompatibleLlmClient({
      baseUrl: `http://127.0.0.1:${port}/v1`,
      apiKey: "test-secret",
      model: "qwen-test",
      timeoutMs: 2_000,
      maxRetries: 0,
    });
    const result = await client.completeJson({
      system: "Return JSON.",
      prompt: "Analyze this batch.",
    });

    expect(result.value).toEqual({ documents: [] });
    expect(result.usage).toEqual({ promptTokens: 12, completionTokens: 3 });
    expect(receivedHeaders.authorization).toBe("Bearer test-secret");
    expect(receivedBody).toMatchObject({
      model: "qwen-test",
      messages: [
        { role: "system", content: "Return JSON." },
        { role: "user", content: "Analyze this batch." },
      ],
    });
    expect(receivedBody).not.toHaveProperty("response_format");
  });

  it("retries one transient response and never includes the key in its error", async () => {
    let attempts = 0;
    const server = createServer((_request, response) => {
      attempts += 1;
      if (attempts === 1) {
        response.writeHead(503, { "content-type": "application/json" });
        response.end('{"message":"temporary"}');
        return;
      }
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        choices: [{ message: { content: '{"ok":true}' } }],
      }));
    });
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;

    const client = new OpenAiCompatibleLlmClient({
      baseUrl: `http://127.0.0.1:${port}/v1`,
      apiKey: "must-not-leak",
      model: "qwen-test",
      timeoutMs: 2_000,
      maxRetries: 1,
      retryDelayMs: 1,
    });

    await expect(client.completeJson({ system: "json", prompt: "retry" }))
      .resolves.toMatchObject({ value: { ok: true } });
    expect(attempts).toBe(2);
  });

  it("retries a successful HTTP response with a missing completion choice", async () => {
    let attempts = 0;
    const server = createServer((_request, response) => {
      attempts += 1;
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(attempts === 1
        ? { choices: [] }
        : { choices: [{ message: { content: '{"ok":true}' } }] }));
    });
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;
    const client = new OpenAiCompatibleLlmClient({
      baseUrl: `http://127.0.0.1:${port}/v1`,
      apiKey: "test-secret",
      model: "qwen-test",
      timeoutMs: 2_000,
      maxRetries: 1,
      retryDelayMs: 1,
    });

    await expect(client.completeJson({ system: "json", prompt: "retry" }))
      .resolves.toMatchObject({ value: { ok: true } });
    expect(attempts).toBe(2);
  });

  it("aborts an active completion without retrying", async () => {
    let attempts = 0;
    const server = createServer((request) => {
      attempts += 1;
      request.resume();
    });
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;
    const client = new OpenAiCompatibleLlmClient({
      baseUrl: `http://127.0.0.1:${port}/v1`,
      apiKey: "test-secret",
      model: "qwen-test",
      timeoutMs: 10_000,
      maxRetries: 2,
      retryDelayMs: 1,
    });
    const controller = new AbortController();
    const completion = client.completeJson({
      system: "json",
      prompt: "cancel",
      signal: controller.signal,
    });
    setTimeout(() => controller.abort("test cancellation"), 20);

    await expect(completion).rejects.toMatchObject({ name: "AbortError" });
    expect(attempts).toBe(1);
  });

  it("times out while a successful response body is still streaming", async () => {
    const server = createServer((_request, response) => {
      response.writeHead(200, { "content-type": "application/json" });
      response.flushHeaders();
      response.write('{"choices":[');
    });
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;
    const client = new OpenAiCompatibleLlmClient({
      baseUrl: `http://127.0.0.1:${port}/v1`,
      apiKey: "test-secret",
      model: "qwen-test",
      timeoutMs: 30,
      maxRetries: 0,
    });

    await expect(client.completeJson({ system: "json", prompt: "timeout" }))
      .rejects.toThrow(/LLM completion failed/);
  });
});
