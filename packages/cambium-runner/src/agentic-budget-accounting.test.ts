/**
 * A gen whose model requests several tool calls per turn must be able to run
 * to its configured max_tool_calls and finalize. It used to hard-fail with the
 * step's output discarded, because runGen charged each dispatch once in the
 * loop and again when it walked the loop's traceSteps afterwards.
 *
 * These tests reproduce runGen's post-loop walk (see `budgetTrack`) rather
 * than standing up runGen itself.
 */
import { describe, it, expect } from 'vitest';
import { join } from 'node:path';
import { handleAgenticGenerate } from './step-handlers.js';
import { ToolRegistry } from './tools/registry.js';
import { testOverrideHandlers } from './tools/index.js';
import { Budget, trackBudgetFromTraceStep } from './budget.js';

const BUILTINS = join(process.cwd(), 'packages/cambium-runner/src/builtin-tools');
const APP_TOOLS = join(process.cwd(), 'packages/cambium/app/tools');

const registry = new ToolRegistry();
await registry.loadFromDir(BUILTINS);
await registry.loadFromDir(APP_TOOLS);

let dispatched = 0;
(registry as any).defs.set('probe', {
  name: 'probe',
  description: 'probe tool',
  permissions: { pure: true },
  inputSchema: {},
  outputSchema: {},
});
testOverrideHandlers['probe'] = async (input: any) => {
  dispatched += 1;
  return { value: input.q };
};

const SCHEMA = {
  $id: 'ProbeOut',
  type: 'object',
  properties: { answer: { type: 'string' } },
} as any;

const STEP = { prompt: 'Do the thing.' } as any;
const IR = {
  model: { id: 'test:model', max_tokens: 512, temperature: 0 },
  system: 'You are a test agent.',
  context: {},
  policies: {},
} as any;

/** Two tool calls per turn until `turnsWithCalls` is exhausted, then final JSON. */
function twoCallsPerTurn(turnsWithCalls: number) {
  let turn = 0;
  return async (opts: any) => {
    turn += 1;
    // A forced-final turn is offered no tools; answer with content, as a real
    // model would.
    const toolsOffered = (opts?.tools?.length ?? 0) > 0;
    if (toolsOffered && turn <= turnsWithCalls) {
      return {
        message: {
          content: null,
          tool_calls: [
            { id: `a${turn}`, type: 'function', function: { name: 'probe', arguments: `{"q":"a${turn}"}` } },
            { id: `b${turn}`, type: 'function', function: { name: 'probe', arguments: `{"q":"b${turn}"}` } },
          ],
        },
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      };
    }
    return {
      message: { content: '{"answer":"done"}', tool_calls: null },
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    };
  };
}

/** What runGen does with the loop's traceSteps once it returns. */
function walkTraceSteps(budget: Budget, traceSteps: any[]) {
  for (const ts of traceSteps) {
    trackBudgetFromTraceStep(budget, ts);
    const violation = budget.check();
    if (violation) throw new Error(violation.message);
  }
}

async function run(maxCalls: number, turnsWithCalls: number) {
  dispatched = 0;
  const budget = new Budget({ max_tool_calls: maxCalls }, {});
  const toolsOpenAI = registry.toOpenAIFormat(['probe']);
  const result = await handleAgenticGenerate(
    STEP, IR, SCHEMA, toolsOpenAI, registry, ['probe'],
    twoCallsPerTurn(turnsWithCalls) as any,
    (raw: string) => JSON.parse(raw),
    maxCalls,
    { budget } as any,
    { documents: [], groundingTextByKey: {} } as any,
  );
  return { budget, result };
}

describe('agentic tool-call budget accounting', () => {
  it('charges each dispatched call exactly once', async () => {
    const { budget } = await run(10, 3);

    expect(dispatched).toBe(6);
    expect(budget.toolCallsUsed).toBe(6);
  });

  it('finalizes instead of throwing when a multi-call-per-turn run reaches its cap', async () => {
    // 4 turns x 2 calls = 8 dispatched against a cap of 8: legitimately AT the
    // cap, never over it.
    const { budget, result } = await run(8, 4);

    expect(dispatched).toBe(8);
    expect(budget.toolCallsUsed).toBe(8);
    expect(budget.check()).toBeNull();
    expect(result.parsed).toEqual({ answer: 'done' });

    // runGen's post-loop walk must not push an at-cap run over.
    expect(() => walkTraceSteps(budget, result.traceSteps)).not.toThrow();
    expect(budget.toolCallsUsed).toBe(8);
  });

  it('refuses the call that would cross the cap and still returns output', async () => {
    // Model wants 10 calls, cap is 5. The pre-call gate refuses the 6th, which
    // ends the loop early and forces a final turn.
    const { budget, result } = await run(5, 5);

    expect(dispatched).toBe(5);
    expect(budget.toolCallsUsed).toBe(5);
    expect(result.parsed).toEqual({ answer: 'done' });
    expect(() => walkTraceSteps(budget, result.traceSteps)).not.toThrow();
  });
});
