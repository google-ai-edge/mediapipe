/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {WasmModule} from '../../../../web/graph_runner/graph_runner';
import 'jasmine';
import {
  BooleanResult,
  ChoiceResult,
  DecisionMaker,
  ScoreResult,
} from './decision_maker';

class DecisionMakerFake extends DecisionMaker {
  mockModule: Record<string, jasmine.Spy>;

  static async create(
    backendType = 0,
    maxNumTokens = 4096,
  ): Promise<DecisionMakerFake> {
    const fake = new DecisionMakerFake();
    await fake.init(backendType, maxNumTokens);
    return fake;
  }

  constructor() {
    const mockPredicateResult: BooleanResult = {
      value: true,
      probabilityTrue: 0.88,
      confidence: 0.95,
    };

    const mockChoiceResult: ChoiceResult = {
      selectedKey: 'device',
      probabilities: {'cloud': 0.1, 'device': 0.85, 'search': 0.05},
      confidence: 0.91,
    };

    const mockScoreResult: ScoreResult = {
      expectedScore: 3.8,
      probabilities: [0.01, 0.04, 0.15, 0.7, 0.1],
      confidence: 0.85,
    };

    const module = {
      'FS_createDataFile': jasmine.createSpy('FS_createDataFile'),
      'FS_unlink': jasmine.createSpy('FS_unlink'),
      'createDecision': jasmine
        .createSpy('createDecision')
        .and.returnValue(Promise.resolve(54321)),
      'deleteDecision': jasmine.createSpy('deleteDecision'),
      'evaluateBoolean': jasmine
        .createSpy('evaluateBoolean')
        .and.returnValue(Promise.resolve(mockPredicateResult)),
      'evaluateChoice': jasmine
        .createSpy('evaluateChoice')
        .and.returnValue(Promise.resolve(mockChoiceResult)),
      'evaluateScore': jasmine
        .createSpy('evaluateScore')
        .and.returnValue(Promise.resolve(mockScoreResult)),
      'evaluateBooleanBatch': jasmine
        .createSpy('evaluateBooleanBatch')
        .and.callFake((_ptr: number, texts: string[]) =>
          Promise.resolve(texts.map(() => mockPredicateResult)),
        ),
      'evaluateChoiceBatch': jasmine
        .createSpy('evaluateChoiceBatch')
        .and.callFake((_ptr: number, texts: string[]) =>
          Promise.resolve(texts.map(() => mockChoiceResult)),
        ),
      'evaluateScoreBatch': jasmine
        .createSpy('evaluateScoreBatch')
        .and.callFake((_ptr: number, texts: string[]) =>
          Promise.resolve(texts.map(() => mockScoreResult)),
        ),
      '_setAutoRenderToScreen': jasmine.createSpy('_setAutoRenderToScreen'),
      '_closeGraph': jasmine.createSpy('_closeGraph'),
    };
    super(module as unknown as WasmModule, null);
    this.mockModule = module as Record<string, jasmine.Spy>;
  }

  async init(backendType = 0, maxNumTokens = 4096): Promise<void> {
    await this.setOptions({backendType, maxNumTokens});
  }
}

describe('DecisionMaker', () => {
  let decisionMaker: DecisionMakerFake;
  let mockModule: Record<string, jasmine.Spy>;

  beforeEach(async () => {
    decisionMaker = await DecisionMakerFake.create();
    mockModule = decisionMaker.mockModule;
  });

  afterEach(() => {
    if (decisionMaker) {
      decisionMaker.close();
    }
  });

  it('initializes with default backend and max tokens', () => {
    expect(decisionMaker).toBeDefined();
    expect(mockModule['createDecision']).toHaveBeenCalledWith(
      '/model.litertlm',
      0,
      4096,
    );
  });

  it('initializes with custom backend and max tokens', async () => {
    const customMaker = await DecisionMakerFake.create(2, 2048);
    expect(customMaker).toBeDefined();
    expect(customMaker.mockModule['createDecision']).toHaveBeenCalledWith(
      '/model.litertlm',
      2,
      2048,
    );
  });

  it('evaluates boolean predicate', async () => {
    const result = await decisionMaker.evaluateBoolean('Play jazz music', {
      condition: 'The user wants to play media',
      threshold: 0.6,
      temperature: 0.8,
    });
    expect(mockModule['evaluateBoolean']).toHaveBeenCalledWith(
      54321,
      'Play jazz music',
      'The user wants to play media',
      0.6,
      0.8,
    );
    expect(result.value).toBeTrue();
    expect(result.probabilityTrue).toBe(0.88);
  });

  it('evaluates categorical choice', async () => {
    const criteria: Record<string, string> = {
      'cloud': 'Complex cloud query',
      'device': 'Fast on-device task',
      'search': 'Live web search',
    };
    const result = await decisionMaker.evaluateChoice('Turn off living room', {
      criteria,
      temperature: 0.7,
    });
    expect(mockModule['evaluateChoice']).toHaveBeenCalledWith(
      54321,
      'Turn off living room',
      criteria,
      0.7,
      '',
    );
    expect(result.selectedKey).toBe('device');
    expect(result.probabilities['device']).toBe(0.85);
  });

  it('evaluates ordinal score', async () => {
    const rubric = ['Poor', 'Fair', 'Good', 'Excellent'];
    const result = await decisionMaker.evaluateScore('Great service and food', {
      rubric,
      temperature: 1.0,
    });
    expect(mockModule['evaluateScore']).toHaveBeenCalledWith(
      54321,
      'Great service and food',
      rubric,
      1.0,
      '',
    );
    expect(result.expectedScore).toBe(3.8);
  });

  it('deletes maker pointer on close', () => {
    decisionMaker.close();
    expect(mockModule['deleteDecision']).toHaveBeenCalledWith(54321);
  });

  it('evaluates batch and shared-prefix candidates in a single Wasm call', async () => {
    const criteria: Record<string, string> = {
      'cloud': 'Complex cloud query',
      'device': 'Fast on-device task',
    };
    const queries = ['Set a timer', 'Explain quantum mechanics'];
    const batchResults = await decisionMaker.evaluateChoiceBatch(
      queries,
      {criteria, temperature: 0.9, instructions: 'Route query'},
      'Shared System Context',
    );
    expect(mockModule['evaluateChoiceBatch']).toHaveBeenCalledWith(
      54321,
      queries,
      criteria,
      0.9,
      'Route query',
      'Shared System Context',
    );
    expect(batchResults.length).toBe(2);
    expect(batchResults[0].selectedKey).toBe('device');
  });

  it('evaluates unified multi-question ClassifierSchema and numeric ordinal levels', async () => {
    const schema = {
      context: 'SaaS support desk triage.',
      questions: [
        {
          id: 'is_urgent',
          type: 'binary' as const,
          prompt: 'Does this require immediate attention?',
        },
        {
          id: 'route',
          type: 'categorical' as const,
          prompt: 'Which queue should handle this?',
          options: [
            {label: 'cloud', description: 'Complex cloud query'},
            {label: 'device', description: 'Fast on-device task'},
            {label: 'search', description: 'Live web search'},
          ],
        },
        {
          id: 'severity',
          type: 'ordinal' as const,
          prompt: 'Rate severity from 1 to 5.',
          options: [
            {label: '1', description: 'Cosmetic'},
            {label: '2', description: 'Minor'},
            {label: '3', description: 'Moderate'},
            {label: '4', description: 'Major'},
            {label: '5', description: 'Critical'},
          ],
        },
      ],
    };
    const result = await decisionMaker.evaluate('Production outage', schema, {
      context: 'Enterprise tier',
    });
    expect(result['is_urgent'].label).toBe('true');
    expect(result['is_urgent'].probability).toBe(0.88);
    expect(result['route'].label).toBe('device');
    expect(result['severity'].label).toBe('4');
    // 1*0.01 + 2*0.04 + 3*0.15 + 4*0.70 + 5*0.10 = 3.84
    expect(result['severity'].expectedScore).toBeCloseTo(3.84, 2);
  });

  it('throws error when evaluating after close', () => {
    decisionMaker.close();
    expect(() =>
      decisionMaker.evaluateBoolean('test', {condition: 'is test'}),
    ).toThrowError('Decision is already closed.');
  });

  it('streams model and companion per-layer embedder into Wasm heap with zero-copy pointers and frees on close', async () => {
    const fakeHeap = new Uint8Array(1024);
    let nextPtr = 64;
    const mallocSpy = jasmine
      .createSpy('_malloc')
      .and.callFake((size: number) => {
        const ptr = nextPtr;
        nextPtr += size;
        return ptr;
      });
    const freeSpy = jasmine.createSpy('_free');
    const createFromBuffersSpy = jasmine
      .createSpy('createDecisionFromBuffers')
      .and.returnValue(Promise.resolve(99999));
    const deleteSdmSpy = jasmine.createSpy('deleteDecision');
    const streamingModule: Record<string, unknown> = {};
    streamingModule['HEAPU8'] = fakeHeap;
    streamingModule['_malloc'] = mallocSpy;
    streamingModule['_free'] = freeSpy;
    streamingModule['createDecisionFromBuffers'] = createFromBuffersSpy;
    streamingModule['deleteDecision'] = deleteSdmSpy;
    streamingModule['_setAutoRenderToScreen'] = jasmine.createSpy(
      '_setAutoRenderToScreen',
    );
    streamingModule['_closeGraph'] = jasmine.createSpy('_closeGraph');
    const streamingMaker = new DecisionMaker(
      streamingModule as unknown as WasmModule,
      null,
    );
    const modelBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const pleBytes = new Uint8Array([10, 20, 30, 40]);
    await streamingMaker.setOptions({
      baseOptions: {
        modelAssetBuffer: modelBytes,
      },
      companionPleAssetBuffer: pleBytes,
      backendType: 2,
      maxNumTokens: 4096,
    });
    expect(mallocSpy).toHaveBeenCalledWith(8);
    expect(mallocSpy).toHaveBeenCalledWith(4);
    expect(createFromBuffersSpy).toHaveBeenCalledWith(64, 8, 72, 4, 2, 4096);
    expect(Array.from(fakeHeap.slice(64, 72))).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8,
    ]);
    expect(Array.from(fakeHeap.slice(72, 76))).toEqual([10, 20, 30, 40]);

    streamingMaker.close();
    expect(deleteSdmSpy).toHaveBeenCalledWith(99999);
    expect(freeSpy).toHaveBeenCalledWith(64);
    expect(freeSpy).toHaveBeenCalledWith(72);
  });
});
