/**
 * DeepSeek V4.1 Flash profile, 32-image provider cap, and /workers-ai/ routing.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { transformOpenAIChatCompletions } from '../src/core/openai.js';
import { resolveGptProfile, isMisresolvedModelId } from '../src/core/gpt-model-profiles.js';
import { createProxy } from '../src/core/proxy.js';

const enc = new TextEncoder();
const dec = new TextDecoder();
const BIG_SLAB = 'You are a coding agent with detailed instructions. '.repeat(80);
const OPENING_PROMPT_MARKER = 'OPENING_PROMPT_SHOULD_BE_HISTORY';
const LIVE_PROMPT_MARKER = 'LIVE_CURRENT_PROMPT_SHOULD_STAY_TEXT';

function buildChatMessages(turns: number): Array<Record<string, unknown>> {
  const msgs: Array<Record<string, unknown>> = [
    { role: 'system', content: BIG_SLAB },
    { role: 'user', content: `${OPENING_PROMPT_MARKER} `.repeat(40) },
  ];
  for (let i = 0; i < turns; i++) {
    const id = `call_${i}`;
    msgs.push({
      role: 'assistant',
      content: `Working on step ${i}. `.repeat(30),
      tool_calls: [{ id, type: 'function', function: { name: 'read', arguments: `{"path":"f${i}"}` } }],
    });
    msgs.push({ role: 'tool', tool_call_id: id, content: `result ${i} `.repeat(50) });
    msgs.push({
      role: 'user',
      content: i === turns - 1
        ? `${LIVE_PROMPT_MARKER} `.repeat(20)
        : `Continue with ${i}. `.repeat(20),
    });
  }
  return msgs;
}

let ambientPxpipeModels: string | undefined;
const realFetch = globalThis.fetch;

beforeEach(() => {
  ambientPxpipeModels = process.env.PXPIPE_MODELS;
  delete process.env.PXPIPE_MODELS;
});

afterEach(() => {
  globalThis.fetch = realFetch;
  if (ambientPxpipeModels === undefined) delete process.env.PXPIPE_MODELS;
  else process.env.PXPIPE_MODELS = ambientPxpipeModels;
});

describe('resolveGptProfile (DeepSeek V4.1)', () => {
  it('uses native 14px packing, maxH 768, and the 32-image provider cap', () => {
    for (const id of [
      '@cf/deepseek-ai/deepseek-v4.1-flash-1017',
      'workers-ai/@cf/deepseek-ai/deepseek-v4.1-flash-1017',
      'deepseek-v4.1-flash-1017',
      'deepseek-v4.1-flash',
      'deepseek/deepseek-v4.1-flash',
    ]) {
      const p = resolveGptProfile(id);
      expect(p.stripCols).toBe(84);
      expect(p.maxHeightPx).toBe(768);
      expect(p.style.font).toBe('jetbrains-mono-14');
      expect(p.history.maxImages).toBe(32);
      expect(p.history.keepTail).toBe(1);
      expect(p.history.minCollapseTokens).toBe(0);
      expect(p.history.minCollapsePrefix).toBe(1);
      expect(p.history.responsesMode).toBe('mixed');
      expect(p.providerImageCap).toBe(32);
      expect(isMisresolvedModelId(id)).toBe(false);
    }
  });

  it('refuses text-only or unmeasured DeepSeek variants instead of applying OpenAI tile math', () => {
    for (const id of [
      '@cf/deepseek-ai/deepseek-v4-flash',
      '@cf/deepseek-ai/deepseek-v4-flash-0731',
      '@cf/deepseek-ai/deepseek-v4-pro',
      '@cf/deepseek-ai/deepseek-v4-pro-0813',
      'deepseek-v3.2',
      'deepseek-r1',
    ]) {
      expect(resolveGptProfile(id).stripCols).not.toBe(84);
      expect(isMisresolvedModelId(id)).toBe(true);
    }
  });
});

describe('DeepSeek V4.1 provider image cap and gateway routing', () => {
  it('budgets history against 32 minus the images already in the request', async () => {
    const tinyPng = { url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==' };
    const messages = buildChatMessages(20);
    const lastMsg = messages.at(-1);
    if (!lastMsg) throw new Error('messages must not be empty');
    lastMsg.content = [
      { type: 'text', text: lastMsg.content },
      ...Array.from({ length: 24 }, () => ({ type: 'image_url', image_url: tinyPng })),
    ];
    const result = await transformOpenAIChatCompletions(
      enc.encode(JSON.stringify({ model: '@cf/deepseek-ai/deepseek-v4.1-flash-1017', messages })),
      { charsPerToken: 1, minCompressChars: 1 },
    );
    expect(result.info.compressed).toBe(true);
    expect(result.info.historyReason).toBe('collapsed');
    expect(result.info.collapsedImages ?? 0).toBeGreaterThan(0);

    const out = JSON.parse(dec.decode(result.body)) as { messages: Array<{ role: string; content: unknown }> };
    let totalImages = 0;
    for (const m of out.messages) {
      if (Array.isArray(m.content)) {
        totalImages += (m.content as Array<{ type?: string }>).filter((p) => p.type === 'image_url').length;
      }
    }
    expect(totalImages).toBeLessThanOrEqual(32);
  });

  it('forwards /workers-ai/v1/chat/completions without injecting Anthropic credentials', async () => {
    let capturedUrl: string | undefined;
    let capturedHeaders: Headers | undefined;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      capturedUrl = String(input);
      capturedHeaders = new Headers(init?.headers);
      return new Response(
        JSON.stringify({
          id: 'chatcmpl_ds',
          choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
        }),
        { headers: { 'content-type': 'application/json' } },
      );
    }) as typeof fetch;

    await createProxy({
      upstream: 'http://gateway.test',
      apiKey: 'sk-anthropic-test',
    })(
      new Request('http://localhost/workers-ai/v1/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: 'Bearer local-token' },
        body: JSON.stringify({
          model: '@cf/deepseek-ai/deepseek-v4.1-flash-1017',
          messages: [{ role: 'user', content: 'hi' }],
        }),
      }),
    );

    expect(capturedUrl).toBe('http://gateway.test/workers-ai/v1/chat/completions');
    expect(capturedHeaders?.get('authorization')).toBe('Bearer local-token');
    expect(capturedHeaders?.get('x-api-key')).toBeNull();
  });
});
