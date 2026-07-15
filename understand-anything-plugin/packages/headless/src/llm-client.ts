export interface LlmClientConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  timeoutMs: number;
  maxRetries: number;
  retryDelayMs?: number;
  jsonResponseFormat?: boolean;
  completionsPath?: string;
  maxResponseBytes?: number;
  maxCompletionTokens?: number;
}

export interface JsonCompletionRequest {
  system: string;
  prompt: string;
  signal?: AbortSignal;
}

export interface JsonCompletionResult {
  value: Record<string, unknown>;
  usage: {
    promptTokens: number;
    completionTokens: number;
  };
}

interface ChatCompletionResponse {
  choices?: Array<{ message?: { content?: string } }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
  };
}

export class OpenAiCompatibleLlmClient {
  readonly modelRef: string;
  private readonly endpoint: string;
  private readonly retryDelayMs: number;
  private readonly maxResponseBytes: number;
  private readonly maxCompletionTokens: number;

  constructor(private readonly config: LlmClientConfig) {
    if (!config.baseUrl || !config.apiKey || !config.model) {
      throw new Error("LLM baseUrl, apiKey, and model are required");
    }
    if (!Number.isInteger(config.timeoutMs) || config.timeoutMs <= 0) {
      throw new Error("LLM timeoutMs must be a positive integer");
    }
    if (!Number.isInteger(config.maxRetries) || config.maxRetries < 0 ||
        config.maxRetries > 10) {
      throw new Error("LLM maxRetries must be an integer between 0 and 10");
    }
    const completionsPath = config.completionsPath ?? "/chat/completions";
    if (!completionsPath.startsWith("/") || completionsPath.includes("://")) {
      throw new Error("LLM completionsPath must be an absolute URL path");
    }
    this.maxResponseBytes = config.maxResponseBytes ?? 4_194_304;
    if (!Number.isInteger(this.maxResponseBytes) || this.maxResponseBytes <= 0) {
      throw new Error("LLM maxResponseBytes must be a positive integer");
    }
    this.maxCompletionTokens = config.maxCompletionTokens ?? 8_192;
    if (!Number.isInteger(this.maxCompletionTokens) || this.maxCompletionTokens <= 0) {
      throw new Error("LLM maxCompletionTokens must be a positive integer");
    }
    this.modelRef = config.model;
    this.endpoint = `${config.baseUrl.replace(/\/+$/, "")}${completionsPath}`;
    this.retryDelayMs = config.retryDelayMs ?? 500;
  }

  async completeJson(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
    let lastError: Error | undefined;
    for (let attempt = 0; attempt <= this.config.maxRetries; attempt += 1) {
      try {
        return await this.execute(request);
      } catch (error) {
        lastError = toError(error);
        if (request.signal?.aborted) throw abortError(request.signal.reason);
        if (attempt >= this.config.maxRetries || !isRetryable(error)) {
          break;
        }
        await delay(this.retryDelayMs * (attempt + 1), request.signal);
      }
    }
    throw new Error(`LLM completion failed: ${lastError?.message ?? "unknown error"}`);
  }

  private async execute(request: JsonCompletionRequest): Promise<JsonCompletionResult> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.timeoutMs);
    try {
      const response = await fetch(this.endpoint, {
        method: "POST",
        headers: {
          authorization: `Bearer ${this.config.apiKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: this.config.model,
          temperature: 0.1,
          max_tokens: this.maxCompletionTokens,
          ...(this.config.jsonResponseFormat
            ? { response_format: { type: "json_object" } }
            : {}),
          messages: [
            { role: "system", content: request.system },
            { role: "user", content: request.prompt },
          ],
        }),
        signal: request.signal
          ? AbortSignal.any([controller.signal, request.signal])
          : controller.signal,
      });
      if (!response.ok) {
        throw new LlmHttpError(response.status);
      }

      const responseText = await readBoundedResponse(response, this.maxResponseBytes);
      let payload: ChatCompletionResponse;
      try {
        payload = JSON.parse(responseText) as ChatCompletionResponse;
      } catch {
        throw new LlmProtocolError("response body was not valid JSON");
      }
      const content = payload.choices?.[0]?.message?.content;
      if (!content) {
        throw new LlmProtocolError(
          "response did not contain choices[0].message.content");
      }
      let value: Record<string, unknown>;
      try {
        value = parseJsonObject(content);
      } catch {
        throw new LlmProtocolError("response content was not a JSON object");
      }
      return {
        value,
        usage: {
          promptTokens: payload.usage?.prompt_tokens ?? 0,
          completionTokens: payload.usage?.completion_tokens ?? 0,
        },
      };
    } finally {
      clearTimeout(timeout);
    }
  }
}

class LlmHttpError extends Error {
  constructor(readonly status: number) {
    super(`HTTP ${status}`);
  }
}

class LlmProtocolError extends Error {}

function isRetryable(error: unknown): boolean {
  if (error instanceof LlmHttpError) {
    return error.status === 408 || error.status === 429 || error.status >= 500;
  }
  if (error instanceof LlmProtocolError) return true;
  return error instanceof TypeError ||
    (error instanceof DOMException &&
      (error.name === "TimeoutError" || error.name === "AbortError"));
}

function parseJsonObject(content: string): Record<string, unknown> {
  const fenced = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const text = (fenced?.[1] ?? content).trim();
  const parsed = JSON.parse(text) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("response content must be a JSON object");
  }
  return parsed as Record<string, unknown>;
}

async function readBoundedResponse(response: Response, maximumBytes: number): Promise<string> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    totalBytes += value.byteLength;
    if (totalBytes > maximumBytes) {
      await reader.cancel().catch(() => undefined);
      throw new LlmProtocolError(
        `response exceeded maximum size of ${maximumBytes} bytes`);
    }
    chunks.push(value);
  }
  const merged = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(merged);
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function delay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(abortError(signal.reason));
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    const onAbort = (): void => {
      clearTimeout(timeout);
      reject(abortError(signal?.reason));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function abortError(reason: unknown): DOMException {
  return new DOMException(
    reason instanceof Error ? reason.message : "operation cancelled",
    "AbortError",
  );
}
