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

import {TaskLogger} from '../../../../tasks/web/core/task_logger';
import {WasmModule} from '../../../../web/graph_runner/graph_runner';
import 'jasmine';
import {
  BooleanResult,
  ChoiceResult,
  DecisionMaker,
  ScoreResult,
} from './decision_maker';

// Example arguments that the Wasm functions get for questions without examples.
const NO_BOOLEAN_EXAMPLES = {
  positiveExamples: [],
  negativeExamples: [],
  inputPrompt: 'CLASSIFICATION',
};
const NO_CHOICE_EXAMPLES = {
  examples: {},
  inputPrompt: 'CLASSIFICATION',
  balanceClassWeights: true,
};

interface DecisionMakerInternals {
  logger?: TaskLogger;
}

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

    const mockChoiceResult: Omit<ChoiceResult, 'predictionSet'> = {
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
      'evaluateContextBoolean': jasmine
        .createSpy('evaluateContextBoolean')
        .and.returnValue(Promise.resolve(mockPredicateResult)),
      'evaluateContextChoice': jasmine
        .createSpy('evaluateContextChoice')
        .and.returnValue(Promise.resolve(mockChoiceResult)),
      'evaluateContextScore': jasmine
        .createSpy('evaluateContextScore')
        .and.returnValue(Promise.resolve(mockScoreResult)),
      'evaluateSchema': jasmine.createSpy('evaluateSchema').and.returnValue(
        Promise.resolve({
          'is_urgent': {
            id: 'is_urgent',
            label: 'true',
            probability: 0.88,
            probabilities: [
              {label: 'false', probability: 0.12},
              {label: 'true', probability: 0.88},
            ],
            confidence: 0.95,
          },
          'route': {
            id: 'route',
            label: 'device',
            probabilities: [
              {label: 'cloud', probability: 0.1},
              {label: 'device', probability: 0.85},
              {label: 'search', probability: 0.05},
            ],
            confidence: 0.91,
          },
          'severity': {
            id: 'severity',
            label: '4',
            probabilities: [
              {label: '1', probability: 0.01},
              {label: '2', probability: 0.04},
              {label: '3', probability: 0.15},
              {label: '4', probability: 0.7},
              {label: '5', probability: 0.1},
            ],
            confidence: 0.85,
            expectedScore: 3.84,
          },
        }),
      ),
      'evaluateContextSchema': jasmine
        .createSpy('evaluateContextSchema')
        .and.returnValue(
          Promise.resolve({
            'is_damaged': {
              id: 'is_damaged',
              label: 'true',
              probability: 0.92,
              probabilities: [
                {label: 'false', probability: 0.08},
                {label: 'true', probability: 0.92},
              ],
              confidence: 0.94,
            },
            'resolution': {
              id: 'resolution',
              label: 'instant_refund',
              probabilities: [
                {label: 'instant_refund', probability: 0.87},
                {label: 'human_audit', probability: 0.1},
                {label: 'reject_claim', probability: 0.03},
              ],
              confidence: 0.91,
            },
            'severity': {
              id: 'severity',
              label: '4',
              probabilities: [
                {label: '1', probability: 0.01},
                {label: '2', probability: 0.04},
                {label: '3', probability: 0.1},
                {label: '4', probability: 0.75},
                {label: '5', probability: 0.1},
              ],
              confidence: 0.89,
              expectedScore: 3.89,
            },
          }),
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
      false,
      NO_BOOLEAN_EXAMPLES,
    );
    expect(result.value).toBeTrue();
    expect(result.probabilityTrue).toBe(0.88);
  });

  it('evaluates categorical choice and populates conformal predictionSet', async () => {
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
      false,
      NO_CHOICE_EXAMPLES,
    );
    expect(result.selectedKey).toBe('device');
    expect(result.probabilities['device']).toBe(0.85);
    expect(result.predictionSet).toEqual(['device', 'cloud']);
  });

  it('forwards normalizePrior and supports prewarmChoice and custom boolean options', async () => {
    mockModule['prewarmChoice'] = jasmine
      .createSpy('prewarmChoice')
      .and.returnValue(Promise.resolve(true));

    await decisionMaker.prewarmChoice({
      instructions: 'Categorize support ticket',
      criteria: {
        'refund': 'Refunds, invoice updates, billing issues',
        'bug': 'Software errors, crashes, broken UI',
      },
      normalizePrior: true,
    });
    expect(mockModule['prewarmChoice']).toHaveBeenCalledWith(
      54321,
      {
        'refund': 'Refunds, invoice updates, billing issues',
        'bug': 'Software errors, crashes, broken UI',
      },
      [],
      0,
      'Categorize support ticket',
      true,
      NO_CHOICE_EXAMPLES,
    );

    await decisionMaker.evaluateBoolean(
      'Please cancel my subscription immediately.',
      {
        condition: 'The user wants to cancel their subscription.',
        threshold: 0.5,
        normalizePrior: true,
      },
    );
    expect(mockModule['evaluateBoolean']).toHaveBeenCalledWith(
      54321,
      'Please cancel my subscription immediately.',
      'The user wants to cancel their subscription.',
      0.5,
      0,
      true,
      NO_BOOLEAN_EXAMPLES,
    );

    mockModule['evaluateChoice'].and.returnValue(
      Promise.resolve({
        selectedKey: 'true',
        probabilities: {'false': 0.15, 'true': 0.85},
        confidence: 0.82,
      }),
    );
    const customBool = await decisionMaker.evaluateBoolean(
      'Cancel my monthly plan.',
      {
        condition: 'The user wants to cancel their subscription.',
        normalizePrior: true,
        options: [
          {label: 'false', description: 'Keep subscription active'},
          {label: 'true', description: 'Cancel subscription'},
        ],
      },
    );
    expect(customBool.value).toBeTrue();
    expect(customBool.probabilityTrue).toBeCloseTo(0.85, 5);

    mockModule['evaluateChoice'].and.returnValue(
      Promise.resolve({
        selectedKey: 'allow',
        probabilities: {'allow': 0.8, 'no': 0.2},
        confidence: 0.78,
      }),
    );
    const affirmFirstBool = await decisionMaker.evaluateBoolean(
      'Approve safe request.',
      {
        condition: 'Should the request be allowed?',
        threshold: 0,
        options: [
          {label: 'allow', description: 'Safe request'},
          {label: 'no', description: 'Unsafe request'},
        ],
      },
    );
    expect(affirmFirstBool.value).toBeTrue();
    expect(affirmFirstBool.probabilityTrue).toBeCloseTo(0.8, 5);
    expect(affirmFirstBool.confidence).toBeCloseTo((0.8 - 0.5) / 0.5, 5);

    (mockModule['prewarmChoice'] as jasmine.Spy).calls.reset();
    await decisionMaker.prewarmBoolean({
      condition: 'Numeric options',
      normalizePrior: true,
      options: [
        {label: '1', description: 'Satisfied'},
        {label: '0', description: 'Unsatisfied'},
      ],
    });
    expect(mockModule['prewarmChoice']).toHaveBeenCalledWith(
      54321,
      {'0': 'Unsatisfied', '1': 'Satisfied'},
      [
        {label: '1', description: 'Satisfied'},
        {label: '0', description: 'Unsatisfied'},
      ],
      0,
      'Numeric options',
      true,
      NO_CHOICE_EXAMPLES,
    );

    mockModule['prewarmSchema'] = jasmine
      .createSpy('prewarmSchema')
      .and.returnValue(Promise.resolve(true));
    await decisionMaker.prewarmScore({
      instructions: 'Rate severity',
      options: [
        {label: '1', description: 'Low'},
        {label: '2', description: 'High'},
      ],
      temperature: 0.7,
    });
    expect(mockModule['prewarmSchema']).toHaveBeenCalledWith(54321, '', [
      {
        id: '__score__',
        type: 'score',
        prompt: 'Rate severity',
        options: [
          {label: '', description: '1: Low'},
          {label: '', description: '2: High'},
        ],
        temperature: 0.7,
      },
    ]);

    // Fallback when `wasmModule.prewarmChoice` is absent: numeric labels must
    // follow `Object.keys(criteria)` order (`'0'`, `'1'`) while preserving
    // attached option media so cache fingerprints match `evaluateChoice`.
    delete mockModule['prewarmChoice'];
    (mockModule['prewarmSchema'] as jasmine.Spy).calls.reset();
    const imgOne = new Uint8Array([1, 2]);
    const imgZero = new Uint8Array([3, 4]);
    await decisionMaker.prewarmChoice({
      instructions: 'Numeric multimodal fallback',
      options: [
        {label: '1', description: 'Satisfied', imageBytes: imgOne},
        {label: '0', description: 'Unsatisfied', imageBytes: imgZero},
      ],
      normalizePrior: true,
    });
    expect(mockModule['prewarmSchema']).toHaveBeenCalledWith(54321, '', [
      {
        id: '__choice__',
        type: 'choice',
        prompt: 'Numeric multimodal fallback',
        options: [
          {label: '0', description: 'Unsatisfied', imageBytes: imgZero},
          {label: '1', description: 'Satisfied', imageBytes: imgOne},
        ],
        temperature: undefined,
        normalizePrior: true,
      },
    ]);
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

    const customResult = await decisionMaker.evaluateScore(
      'Great service and food',
      {
        options: [
          {label: '1', description: 'Cosmetic'},
          {label: '2', description: 'Minor'},
          {label: '3', description: 'Moderate'},
          {label: '4', description: 'Major'},
          {label: '5', description: 'Critical'},
        ],
      },
    );
    expect(customResult.selectedKey).toBe('4');
    // 1*0.01 + 2*0.04 + 3*0.15 + 4*0.70 + 5*0.10 = 3.84
    expect(customResult.expectedScore).toBeCloseTo(3.84, 2);
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
      false,
      NO_CHOICE_EXAMPLES,
    );
    expect(batchResults.length).toBe(2);
    expect(batchResults[0].selectedKey).toBe('device');
    expect(batchResults[0].predictionSet).toEqual(['device', 'cloud']);
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
    expect(result['route'].predictionSet).toEqual(['device', 'cloud']);
    expect(result['severity'].label).toBe('4');
    // 1*0.01 + 2*0.04 + 3*0.15 + 4*0.70 + 5*0.10 = 3.84
    expect(result['severity'].expectedScore).toBeCloseTo(3.84, 2);
  });

  it('evaluates multimodal context (imageBytes and audioSamples) across boolean, choice, score, schema, and classify', async () => {
    const sampleImage = new Uint8Array([137, 80, 78, 71]);
    const sampleAudio = new Float32Array([0.1, -0.2, 0.3, -0.1]);

    const boolResult = await decisionMaker.evaluateBoolean(
      {
        text: 'Customer claims mug arrived shattered',
        imageBytes: sampleImage,
      },
      {
        condition: 'The item in the photo is visibly broken or shattered',
        threshold: 0.5,
        temperature: 0.9,
      },
    );
    expect(mockModule['evaluateContextBoolean']).toHaveBeenCalledWith(
      54321,
      'Customer claims mug arrived shattered',
      'The item in the photo is visibly broken or shattered',
      0.5,
      0.9,
      {
        imageBytes: sampleImage,
        audioSamples: undefined,
      },
      false,
      NO_BOOLEAN_EXAMPLES,
    );
    expect(boolResult.value).toBeTrue();

    const choiceResult = await decisionMaker.evaluateChoice(
      {
        text: 'Doorbell camera alert',
        imageBytes: sampleImage,
        audioSamples: sampleAudio,
      },
      {
        criteria: {
          'unlock': 'Courier delivering package',
          'alert': 'Intruder or glass break',
        },
        temperature: 0.8,
        instructions: 'Select edge action',
      },
    );
    expect(mockModule['evaluateContextChoice']).toHaveBeenCalledWith(
      54321,
      'Doorbell camera alert',
      {
        'unlock': 'Courier delivering package',
        'alert': 'Intruder or glass break',
      },
      [],
      0.8,
      'Select edge action',
      {
        imageBytes: sampleImage,
        audioSamples: sampleAudio,
      },
      false,
      NO_CHOICE_EXAMPLES,
    );
    expect(choiceResult.selectedKey).toBe('device');

    const scoreResult = await decisionMaker.evaluateScore(
      {
        text: 'Assess bumper dent severity',
        imageBytes: sampleImage,
      },
      {
        rubric: ['Scratch', 'Minor dent', 'Moderate', 'Structural'],
        temperature: 1.0,
        instructions: 'Rate damage',
      },
    );
    expect(mockModule['evaluateContextScore']).toHaveBeenCalledWith(
      54321,
      'Assess bumper dent severity',
      ['Scratch', 'Minor dent', 'Moderate', 'Structural'],
      1.0,
      'Rate damage',
      {
        imageBytes: sampleImage,
        audioSamples: undefined,
      },
    );
    expect(scoreResult.expectedScore).toBe(3.8);

    const schema = {
      context: 'E-commerce return desk',
      questions: [
        {
          id: 'is_damaged',
          type: 'binary' as const,
          prompt: 'Is the item visibly damaged?',
        },
        {
          id: 'resolution',
          type: 'categorical' as const,
          prompt: 'Select refund action.',
          options: [
            {label: 'instant_refund', description: 'Approve instant refund'},
            {label: 'human_audit', description: 'Send to manual review'},
            {label: 'reject_claim', description: 'Reject return'},
          ],
        },
        {
          id: 'severity',
          type: 'ordinal' as const,
          prompt: 'Rate damage severity.',
          options: [
            {label: '1', description: 'Intact'},
            {label: '2', description: 'Scuff'},
            {label: '3', description: 'Moderate'},
            {label: '4', description: 'Broken'},
            {label: '5', description: 'Destroyed'},
          ],
        },
      ],
    };
    const schemaResult = await decisionMaker.evaluate(
      {
        text: 'Return claim: shattered ceramic mug',
        imageBytes: sampleImage,
      },
      schema,
    );
    expect(mockModule['evaluateContextSchema']).toHaveBeenCalled();
    expect(schemaResult['is_damaged'].label).toBe('true');
    expect(schemaResult['resolution'].label).toBe('instant_refund');
    expect(schemaResult['severity'].expectedScore).toBeCloseTo(3.89, 2);

    decisionMaker.setSchema(schema);
    const classifyResult = await decisionMaker.classify({
      text: 'Shattered mug photo',
      imageBytes: sampleImage,
    });
    expect(classifyResult['is_damaged'].label).toBe('true');
  });

  it('throws error when evaluating after close', () => {
    decisionMaker.close();
    expect(() =>
      decisionMaker.evaluateBoolean('test', {condition: 'is test'}),
    ).toThrowError('Decision is already closed.');
  });

  function createFakeLogger(): jasmine.SpyObj<TaskLogger> {
    return jasmine.createSpyObj<TaskLogger>('TaskLogger', [
      'logSessionStart',
      'logSessionEnd',
      'recordCpuInputArrival',
      'recordGpuInputArrival',
      'recordInvocationEnd',
      'close',
    ]);
  }

  it('logs session start once the native maker is created', async () => {
    const fakeLogger = createFakeLogger();
    const maker = new DecisionMakerFake();
    (maker as unknown as DecisionMakerInternals).logger = fakeLogger;
    expect(fakeLogger.logSessionStart).not.toHaveBeenCalled();

    await maker.init();
    expect(fakeLogger.logSessionStart).toHaveBeenCalledTimes(1);
    maker.close();
  });

  it('logs invocations and session end/close on evaluate and close calls', async () => {
    const fakeLogger = createFakeLogger();
    (decisionMaker as unknown as DecisionMakerInternals).logger = fakeLogger;

    await decisionMaker.evaluateBoolean('Play jazz music', {
      condition: 'The user wants to play media',
    });
    expect(fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(0);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(0);

    await decisionMaker.evaluateChoice('Turn off living room', {
      criteria: {'cloud': 'Complex cloud query', 'device': 'On-device task'},
    });
    expect(fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(1);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(1);

    await decisionMaker.evaluateScore('Great service and food', {
      rubric: ['Poor', 'Fair', 'Good', 'Excellent'],
    });
    expect(fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(2);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(2);

    await decisionMaker.evaluateChoiceBatch(
      ['Set a timer', 'Explain quantum mechanics'],
      {criteria: {'cloud': 'Complex cloud query', 'device': 'On-device task'}},
    );
    expect(fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(3);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(3);

    await decisionMaker.evaluate('Production outage', {
      questions: [
        {id: 'is_urgent', type: 'binary', prompt: 'Is it urgent?'},
        {
          id: 'route',
          type: 'categorical',
          prompt: 'Which queue?',
          options: [{label: 'cloud'}, {label: 'device'}],
        },
      ],
    });
    expect(fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(4);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(4);
    expect(fakeLogger.recordCpuInputArrival).not.toHaveBeenCalled();

    decisionMaker.close();
    expect(fakeLogger.logSessionEnd).toHaveBeenCalledTimes(1);
    expect(fakeLogger.close).toHaveBeenCalledTimes(1);
  });

  it('records invocation end when the native evaluation fails', async () => {
    const fakeLogger = createFakeLogger();
    (decisionMaker as unknown as DecisionMakerInternals).logger = fakeLogger;
    mockModule['evaluateBoolean'].and.callFake(() =>
      Promise.reject(new Error('native failure')),
    );

    await expectAsync(
      decisionMaker.evaluateBoolean('text', {condition: 'is test'}),
    ).toBeRejectedWithError('native failure');
    expect(fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(0);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(0);
  });

  it('records CPU input arrival for the CPU-only encoder backend', async () => {
    const cpuMaker = await DecisionMakerFake.create(/* backendType= */ 1);
    const fakeLogger = createFakeLogger();
    (cpuMaker as unknown as DecisionMakerInternals).logger = fakeLogger;

    await cpuMaker.evaluateBoolean('text', {condition: 'is test'});
    expect(fakeLogger.recordCpuInputArrival).toHaveBeenCalledWith(0);
    expect(fakeLogger.recordGpuInputArrival).not.toHaveBeenCalled();
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(0);
    cpuMaker.close();
  });

  it('uses CPU_DELEGATE_BACKEND_TYPE (-1) when delegate is CPU and defaults maxNumTokens to 0', async () => {
    const cpuMaker = new DecisionMakerFake();
    const fakeLogger = createFakeLogger();
    (cpuMaker as unknown as DecisionMakerInternals).logger = fakeLogger;
    (cpuMaker.mockModule as unknown as Record<string, unknown>)[
      'preinitializedWebGPUDevice'
    ] = {};
    const globalScope = (typeof self !== 'undefined'
      ? self
      : globalThis) as unknown as Record<string, unknown>;
    const prevGlobalModule = globalScope['Module'];
    const fakeGlobalModule: Record<string, unknown> = {
      'preinitializedWebGPUDevice': {},
    };
    globalScope['Module'] = fakeGlobalModule;

    try {
      await cpuMaker.setOptions({
        baseOptions: {delegate: 'CPU'},
        backendType: 3,
      });
      expect(cpuMaker.mockModule['createDecision']).toHaveBeenCalledWith(
        '/model.litertlm',
        -1,
        0,
      );
      expect(
        (cpuMaker.mockModule as unknown as Record<string, unknown>)[
          'preinitializedWebGPUDevice'
        ],
      ).toBeUndefined();
      expect(fakeGlobalModule['preinitializedWebGPUDevice']).toBeUndefined();

      await cpuMaker.evaluateBoolean('text', {condition: 'is test'});
      expect(fakeLogger.recordCpuInputArrival).toHaveBeenCalledWith(0);
      expect(fakeLogger.recordGpuInputArrival).not.toHaveBeenCalled();
    } finally {
      globalScope['Module'] = prevGlobalModule;
      cpuMaker.close();
    }
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

  it('aligns .task ZIP first entry payload to a 64-byte boundary in Wasm memory', async () => {
    const fakeHeap = new Uint8Array(1024);
    let nextPtr = 68;
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
      .and.returnValue(Promise.resolve(88888));
    const deleteSdmSpy = jasmine.createSpy('deleteDecision');
    const streamingModule: Record<string, unknown> = {
      'HEAPU8': fakeHeap,
      '_malloc': mallocSpy,
      '_free': freeSpy,
      'createDecisionFromBuffers': createFromBuffersSpy,
      'deleteDecision': deleteSdmSpy,
      '_setAutoRenderToScreen': jasmine.createSpy('_setAutoRenderToScreen'),
      '_closeGraph': jasmine.createSpy('_closeGraph'),
    };
    const streamingMaker = new DecisionMaker(
      streamingModule as unknown as WasmModule,
      null,
    );
    // Construct a synthetic 128-byte ZIP local file header with filename
    // length 12 ("model.tflite", payload offset = 30 + 12 = 42).
    const zipBytes = new Uint8Array(128);
    zipBytes[0] = 0x50;
    zipBytes[1] = 0x4b;
    zipBytes[2] = 0x03;
    zipBytes[3] = 0x04;
    zipBytes[26] = 12;
    zipBytes[27] = 0;
    zipBytes[28] = 0;
    zipBytes[29] = 0;
    await streamingMaker.setOptions({
      baseOptions: {
        modelAssetBuffer: zipBytes,
      },
      backendType: 2,
      maxNumTokens: 4096,
    });
    const passedPtr = createFromBuffersSpy.calls.mostRecent().args[0] as number;
    expect((passedPtr + 42) % 64).toBe(0);
    streamingMaker.close();
    expect(freeSpy).toHaveBeenCalledWith(68);
  });

  it('skips software-emulated WebGPU adapters so WASM SIMD runs instead of SwiftShader', async () => {
    const requestDeviceSpy = jasmine.createSpy('requestDevice');
    const fakeAdapter = {
      isFallbackAdapter: false,
      info: {
        vendor: 'google',
        architecture: 'swiftshader',
        description: 'Google SwiftShader',
      },
      features: new Set<string>(),
      limits: {},
      requestDevice: requestDeviceSpy,
    };
    const navObj = (typeof navigator !== 'undefined'
      ? navigator
      : undefined) as unknown as Record<string, unknown> | undefined;
    if (!navObj) {
      return;
    }
    const prevGpu = navObj['gpu'];
    Object.defineProperty(navObj, 'gpu', {
      value: {
        requestAdapter: jasmine
          .createSpy('requestAdapter')
          .and.returnValue(Promise.resolve(fakeAdapter)),
      },
      configurable: true,
      writable: true,
    });

    const maker = new DecisionMakerFake();
    try {
      await maker.setOptions({
        baseOptions: {delegate: 'GPU'},
        backendType: 2,
      });
      expect(requestDeviceSpy).not.toHaveBeenCalled();
      expect(
        (maker.mockModule as unknown as Record<string, unknown>)[
          'preinitializedWebGPUDevice'
        ],
      ).toBeUndefined();
    } finally {
      Object.defineProperty(navObj, 'gpu', {
        value: prevGpu,
        configurable: true,
        writable: true,
      });
      maker.close();
    }
  });

  it('routes multimodal batch questions through context-aware single-item evaluation', async () => {
    const imgA = new Uint8Array([9, 8]);
    const imgB = new Uint8Array([7, 6]);
    const mockModule = decisionMaker.mockModule as unknown as Record<
      string,
      jasmine.Spy
    >;
    mockModule['evaluateChoiceBatch'].calls.reset();
    mockModule['evaluateContextChoice'].calls.reset();

    const choiceResults = await decisionMaker.evaluateChoiceBatch(
      ['first item', 'second item'],
      {
        instructions: 'Match prototype',
        options: [
          {label: 'a', description: 'Option A', imageBytes: imgA},
          {label: 'b', description: 'Option B', imageBytes: imgB},
        ],
      },
      'Shared header',
    );
    expect(choiceResults.length).toBe(2);
    expect(mockModule['evaluateChoiceBatch']).not.toHaveBeenCalled();
    expect(mockModule['evaluateContextChoice']).toHaveBeenCalledTimes(2);
    expect(mockModule['evaluateContextChoice'].calls.argsFor(0)[1]).toBe(
      'Shared header\nfirst item',
    );

    mockModule['evaluateBooleanBatch'].calls.reset();
    mockModule['evaluateContextBoolean'].calls.reset();
    const boolResults = await decisionMaker.evaluateBooleanBatch(
      ['alpha', 'beta'],
      {
        condition: 'Matches attached image',
        imageBytes: imgA,
      },
    );
    expect(boolResults.length).toBe(2);
    expect(mockModule['evaluateBooleanBatch']).not.toHaveBeenCalled();
    expect(mockModule['evaluateContextBoolean']).toHaveBeenCalledTimes(2);

    mockModule['evaluateScoreBatch'].calls.reset();
    mockModule['evaluateContextScore'].calls.reset();
    const scoreResults = await decisionMaker.evaluateScoreBatch(
      ['alpha', 'beta'],
      {
        rubric: ['Low', 'High'],
        imageBytes: imgB,
      },
    );
    expect(scoreResults.length).toBe(2);
    expect(mockModule['evaluateScoreBatch']).not.toHaveBeenCalled();
    expect(mockModule['evaluateContextScore']).toHaveBeenCalledTimes(2);
  });

  it('passes choice examples to Wasm', async () => {
    const examples = {
      'person_name': ['Kavita Ramachandran', 'Marek Kowalczyk'],
      'acronym': ['GEMM'],
    };
    await decisionMaker.evaluateChoice('VAD', {
      examples,
      inputPrompt: 'NONE',
      balanceClassWeights: false,
    });
    expect(mockModule['evaluateChoice']).toHaveBeenCalledWith(
      54321,
      'VAD',
      {'person_name': '', 'acronym': ''},
      0,
      '',
      false,
      {examples, inputPrompt: 'NONE', balanceClassWeights: false},
    );

    await decisionMaker.evaluateChoice('VAD', {
      options: [
        {label: 'a', description: 'Option A', examples: ['x1', 'x2']},
        {label: 'b', examples: ['y1']},
      ],
    });
    expect(mockModule['evaluateChoice']).toHaveBeenCalledWith(
      54321,
      'VAD',
      {'a': 'Option A', 'b': ''},
      0,
      '',
      false,
      {
        examples: {'a': ['x1', 'x2'], 'b': ['y1']},
        inputPrompt: 'CLASSIFICATION',
        balanceClassWeights: true,
      },
    );

    await decisionMaker.evaluateChoiceBatch(['GEMM', 'Ana'], {examples});
    expect(mockModule['evaluateChoiceBatch']).toHaveBeenCalledWith(
      54321,
      ['GEMM', 'Ana'],
      {'person_name': '', 'acronym': ''},
      0,
      '',
      '',
      false,
      {examples, inputPrompt: 'CLASSIFICATION', balanceClassWeights: true},
    );

    mockModule['prewarmChoice'] = jasmine
      .createSpy('prewarmChoice')
      .and.resolveTo(true);
    await decisionMaker.prewarmChoice({examples, inputPrompt: 'NONE'});
    expect(mockModule['prewarmChoice']).toHaveBeenCalledWith(
      54321,
      {'person_name': '', 'acronym': ''},
      [],
      0,
      '',
      false,
      {examples, inputPrompt: 'NONE', balanceClassWeights: true},
    );
  });

  it('rejects choice questions that mix choices with and without examples', async () => {
    expect(() =>
      decisionMaker.evaluateChoice('x', {
        criteria: {'a': 'A', 'b': 'B'},
        examples: {'a': ['x']},
      }),
    ).toThrowError(/same choices/);
    expect(() =>
      decisionMaker.evaluateChoice('x', {examples: {'a': ['x'], 'b': []}}),
    ).toThrowError(/Choice 'b' needs at least one example/);
    expect(() =>
      decisionMaker.evaluateChoice('x', {examples: {'a': ['x']}}),
    ).toThrowError(/at least 2 choices/);
    expect(() =>
      decisionMaker.evaluateChoice('x', {
        options: [{label: 'a', examples: ['x']}, {label: 'b'}],
      }),
    ).toThrowError(/Choice 'b' needs at least one example/);
    await expectAsync(
      decisionMaker.evaluateChoiceBatch(['x'], {examples: {'a': ['x']}}),
    ).toBeRejectedWithError(/at least 2 choices/);
    expect(() =>
      decisionMaker.evaluateChoice('x', {examples: {'a': [], 'b': []}}),
    ).toThrowError(/at least 1 choice/);
    await expectAsync(
      decisionMaker.evaluateChoiceBatch(['x'], {examples: {'a': [], 'b': []}}),
    ).toBeRejectedWithError(/at least 1 choice/);
    expect(mockModule['evaluateChoice']).not.toHaveBeenCalled();
    expect(mockModule['evaluateChoiceBatch']).not.toHaveBeenCalled();
  });

  it('passes boolean examples to Wasm and checks both lists are given', async () => {
    const question = {
      positiveExamples: ['Zurich', 'Kubernetes'],
      negativeExamples: ['table', 'running'],
      inputPrompt: 'NONE' as const,
    };
    await decisionMaker.evaluateBoolean('Geneva', question);
    expect(mockModule['evaluateBoolean']).toHaveBeenCalledWith(
      54321,
      'Geneva',
      '',
      0.5,
      0,
      false,
      question,
    );

    // With examples, custom options don't turn the question into a choice.
    await decisionMaker.evaluateBoolean('Geneva', {
      ...question,
      condition: 'The input is a proper noun.',
      options: [
        {label: 'false', description: 'Common word'},
        {label: 'true', description: 'Proper noun'},
      ],
    });
    expect(mockModule['evaluateChoice']).not.toHaveBeenCalled();
    expect(mockModule['evaluateBoolean']).toHaveBeenCalledTimes(2);

    await decisionMaker.evaluateBooleanBatch(['Bern'], question);
    expect(mockModule['evaluateBooleanBatch']).toHaveBeenCalledWith(
      54321,
      ['Bern'],
      '',
      0.5,
      0,
      '',
      false,
      question,
    );

    expect(() =>
      decisionMaker.evaluateBoolean('Geneva', {positiveExamples: ['Zurich']}),
    ).toThrowError(/both positiveExamples and negativeExamples/);

    // With a context and no condition, the condition is the context alone, as
    // in the schema path that prewarmBoolean uses.
    await decisionMaker.evaluateBoolean('Geneva', {
      ...question,
      context: 'Word types.',
    });
    expect(mockModule['evaluateBoolean']).toHaveBeenCalledWith(
      54321,
      'Geneva',
      'Word types.',
      0.5,
      0,
      false,
      question,
    );
    await decisionMaker.evaluateBooleanBatch(['Bern'], {
      ...question,
      context: 'Word types.',
    });
    expect(mockModule['evaluateBooleanBatch']).toHaveBeenCalledWith(
      54321,
      ['Bern'],
      'Word types.',
      0.5,
      0,
      '',
      false,
      question,
    );
  });

  it('ignores inherited properties as choice keys', async () => {
    const criteria = {'constructor': 'A', 'toString': 'B'};
    await decisionMaker.evaluateChoice('x', {criteria});
    expect(mockModule['evaluateChoice']).toHaveBeenCalledWith(
      54321,
      'x',
      criteria,
      0,
      '',
      false,
      {examples: {}, inputPrompt: 'CLASSIFICATION', balanceClassWeights: true},
    );
    // `toString` is inherited by `examples`, not one of its own keys.
    expect(() =>
      decisionMaker.evaluateChoice('x', {
        criteria,
        examples: {'constructor': ['a'], 'other': ['b']},
      }),
    ).toThrowError(/same choices/);
  });

  it('checks and forwards examples in schemas', async () => {
    const questions = [
      {
        id: 'word_type',
        type: 'choice' as const,
        prompt: '',
        options: [
          {label: 'name', examples: ['Marek']},
          {label: 'acronym', examples: ['GEMM']},
        ],
        inputPrompt: 'NONE' as const,
      },
      {
        id: 'is_term',
        type: 'boolean' as const,
        prompt: '',
        positiveExamples: ['Zurich'],
        negativeExamples: ['table'],
      },
    ];
    await decisionMaker.evaluate('VAD', {questions});
    expect(mockModule['evaluateSchema']).toHaveBeenCalledWith(
      54321,
      'VAD',
      '',
      questions,
    );

    await expectAsync(
      decisionMaker.evaluate('VAD', {
        questions: [
          {
            id: 'mixed',
            type: 'choice',
            prompt: '',
            options: [{label: 'name', examples: ['Marek']}, {label: 'other'}],
          },
        ],
      }),
    ).toBeRejectedWithError(/Choice 'other' needs at least one example/);
    await expectAsync(
      decisionMaker.prewarmBoolean({positiveExamples: ['Zurich']}),
    ).toBeRejectedWithError(/both positiveExamples and negativeExamples/);
  });
});
