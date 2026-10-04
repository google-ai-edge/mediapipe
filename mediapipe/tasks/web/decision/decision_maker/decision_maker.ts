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

import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {
  CachedGraphRunner,
  TaskRunner,
} from '../../../../tasks/web/core/task_runner';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {WasmModule} from '../../../../web/graph_runner/graph_runner';
// Placeholder for internal dependency on trusted resource url
import type {
  BooleanQuestion,
  ChoiceQuestion,
  ClassifierEvaluateOptions,
  ClassifierExpectedInput,
  ClassifierOption,
  ClassifierQuestion,
  ClassifierQuestionType,
  ClassifierSchema,
  DecisionMakerOptions,
  ScoreQuestion,
} from './decision_maker_options';
import type {
  BooleanResult,
  ChoiceResult,
  ClassifierDecision,
  ClassifierOptionProbability,
  ClassifierResult,
  ScoreResult,
} from './decision_maker_result';

export type {
  BooleanQuestion,
  BooleanResult,
  ChoiceQuestion,
  ChoiceResult,
  ClassifierDecision,
  ClassifierEvaluateOptions,
  ClassifierExpectedInput,
  ClassifierOption,
  ClassifierOptionProbability,
  ClassifierQuestion,
  ClassifierQuestionType,
  ClassifierResult,
  ClassifierSchema,
  DecisionMakerOptions,
  ScoreQuestion,
  ScoreResult,
};

/**
 * `DecisionMakerOptions.backendType` value for the encoder backend. This is
 * the only backend that runs on CPU; all other backends run on WebGPU (see the
 * delegate selection in `decision_maker_api.cc`).
 */
const ENCODER_BACKEND_TYPE = 1;

/**
 * `backendType` value passed to `createDecision` / `createDecisionFromBuffers`
 * in `decision_maker_api.cc` to select `MP_DELEGATE_CPU` (`backend_type < 0`).
 */
const CPU_DELEGATE_BACKEND_TYPE = -1;

/**
 * The Wasm module for Decision with custom C++ bindings.
 */
declare interface DecisionMakerWasmModule {
  createDecision(
    modelPath: string,
    backendType: number,
    maxNumTokens: number,
  ): Promise<number>;

  createDecisionFromBuffers?(
    modelPtr: number,
    modelSize: number,
    plePtr: number,
    pleSize: number,
    backendType: number,
    maxNumTokens: number,
  ): Promise<number>;

  deleteDecision(makerPtr: number): void;

  evaluateBoolean(
    makerPtr: number,
    text: string,
    condition: string,
    threshold: number,
    temperature: number,
  ): Promise<BooleanResult>;

  evaluateChoice(
    makerPtr: number,
    text: string,
    criteria: Record<string, string>,
    temperature: number,
    instructions: string,
  ): Promise<ChoiceResult>;

  evaluateScore(
    makerPtr: number,
    text: string,
    rubric: string[],
    temperature: number,
    instructions: string,
  ): Promise<ScoreResult>;

  evaluateBooleanBatch?(
    makerPtr: number,
    texts: string[],
    condition: string,
    threshold: number,
    temperature: number,
    sharedPrefix: string,
  ): Promise<BooleanResult[]>;

  evaluateChoiceBatch?(
    makerPtr: number,
    texts: string[],
    criteria: Record<string, string>,
    temperature: number,
    instructions: string,
    sharedPrefix: string,
  ): Promise<ChoiceResult[]>;

  evaluateScoreBatch?(
    makerPtr: number,
    texts: string[],
    rubric: string[],
    temperature: number,
    instructions: string,
    sharedPrefix: string,
  ): Promise<ScoreResult[]>;

  prewarmSchema?(
    makerPtr: number,
    schemaContext: string,
    questions: ClassifierQuestion[],
  ): Promise<boolean>;

  evaluateSchema?(
    makerPtr: number,
    text: string,
    schemaContext: string,
    questions: ClassifierQuestion[],
  ): Promise<ClassifierResult>;
}

function estimateTokens(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return Math.max(1, Math.ceil(trimmed.length / 4));
}

function resolveChoiceCriteria(
  question: ChoiceQuestion,
): Record<string, string> {
  const criteria: Record<string, string> = {};
  if (question.criteria) {
    for (const [k, v] of Object.entries(question.criteria)) {
      criteria[k] = v;
    }
  }
  if (question.options) {
    for (const opt of question.options) {
      criteria[opt.label] = opt.description ?? '';
    }
  }
  return criteria;
}

function resolveScoreRubric(question: ScoreQuestion): {
  rubric: string[];
  labels: string[];
  levels: number[];
  hasCustomLevels: boolean;
} {
  if (question.options && question.options.length > 0) {
    const rubric = question.options.map((o) =>
      o.description ? `${o.label}: ${o.description}` : o.label,
    );
    const labels = question.options.map((o) => o.label);
    const parsed = labels.map((l) => Number(l));
    const allNumeric =
      parsed.every((n) => Number.isFinite(n)) &&
      labels.every((l) => l.trim() !== '');
    const levels = allNumeric ? parsed : labels.map((_, i) => i);
    return {rubric, labels, levels, hasCustomLevels: allNumeric};
  }
  const rubric = question.rubric ?? [];
  const labels = rubric.map((_, i) => String(i));
  const levels = rubric.map((_, i) => i);
  return {rubric, labels, levels, hasCustomLevels: false};
}

function enrichOrdinalResult(
  raw: ScoreResult,
  labels: string[],
  levels: number[],
  hasCustomLevels: boolean,
): ScoreResult {
  const probs = raw.probabilities ?? [];
  let bestIdx = 0;
  let maxProb = -1;
  let expectedScore = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > maxProb) {
      maxProb = p;
      bestIdx = i;
    }
    const levelVal = i < levels.length ? levels[i] : i;
    expectedScore += levelVal * p;
  }
  return {
    ...raw,
    expectedScore:
      hasCustomLevels && probs.length > 0 ? expectedScore : raw.expectedScore,
    selectedKey:
      bestIdx < labels.length
        ? labels[bestIdx]
        : (raw.selectedKey ?? String(bestIdx)),
  };
}

const recycledWasmModules: WasmModule[] = [];

/**
 * MediaPipe Task that performs semantic decision making.
 */
export class DecisionMaker extends TaskRunner {
  protected override baseOptions: BaseOptionsProto = new BaseOptionsProto();
  private backendType = 0;
  private maxNumTokens = 0;
  private makerPtr = 0;
  private modelWasmPtr = 0;
  private modelWasmRawPtr = 0;
  private pleWasmPtr = 0;
  private pleWasmRawPtr = 0;
  private isClosed = false;
  private sessionSchema?: ClassifierSchema;
  private lastContextUsage = 0;
  private loggerTimestamp = 0;

  /**
   * Initializes the Wasm runtime and creates a new Decision based
   * on options.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param options The options for the DecisionMaker. Note that either a path
   *     to the model or the model itself needs to be provided (via
   *     `baseOptions`).
   */
  static async createFromOptions(
    wasmFileset: WasmFileset,
    options: DecisionMakerOptions,
  ): Promise<DecisionMaker> {
    const recycledWasm = recycledWasmModules.pop();
    if (recycledWasm) {
      const instance = new DecisionMaker(recycledWasm, null);
      instance.enableLogging(options);
      try {
        await instance.setOptions(options);
        return instance;
      } catch (e) {
        instance.close();
        throw e;
      }
    }
    return TaskRunner.createInstance(
      DecisionMaker,
      /* canvas= */ null,
      wasmFileset,
      options,
    );
  }

  /**
   * Initializes the Wasm runtime and creates a new Decision from
   * model asset path.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetPath The path to the model asset.
   * @param backendType The backend type (0=AUTO, 1=ENCODER, 2=DIRECT_LOGIT,
   *     3=EMBEDDING).
   * @param maxNumTokens The maximum number of tokens in prompt context.
   */
  static createFromModelPath(
    wasmFileset: WasmFileset,
    modelAssetPath: string,
    backendType = 0,
    maxNumTokens = 0,
  ): Promise<DecisionMaker> {
    return DecisionMaker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath,
      },
      backendType,
      maxNumTokens,
    });
  }

  /**
   * Initializes the Wasm runtime and creates a new Decision from
   * model buffer.
   *
   * @export
   * @param wasmFileset A configuration object that provides the location of
   *     the Wasm binary and its loader.
   * @param modelAssetBuffer An array or a stream containing a binary
   *     representation of the model.
   * @param backendType The backend type (0=AUTO, 1=ENCODER, 2=DIRECT_LOGIT,
   *     3=EMBEDDING).
   * @param maxNumTokens The maximum number of tokens in prompt context.
   */
  static createFromModelBuffer(
    wasmFileset: WasmFileset,
    modelAssetBuffer: Uint8Array | ReadableStreamDefaultReader,
    backendType = 0,
    maxNumTokens = 0,
  ): Promise<DecisionMaker> {
    return DecisionMaker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetBuffer,
      },
      backendType,
      maxNumTokens,
    });
  }

  /** @hideconstructor */
  constructor(
    wasmModule: WasmModule,
    glCanvas?: HTMLCanvasElement | OffscreenCanvas | null,
  ) {
    super(new CachedGraphRunner(wasmModule, glCanvas));
  }

  protected override getTaskName(): string {
    return 'DecisionMaker';
  }

  /**
   * Runs a native evaluation and records it with the usage logger: input
   * arrival is recorded before `run()` is invoked and invocation end once the
   * returned promise settles.
   */
  private logInvocation<T>(run: () => Promise<T>): Promise<T> {
    const timestamp = this.loggerTimestamp++;
    if (this.logger) {
      if (
        this.backendType === ENCODER_BACKEND_TYPE ||
        this.backendType === CPU_DELEGATE_BACKEND_TYPE
      ) {
        this.logger.recordCpuInputArrival(timestamp);
      } else {
        this.logger.recordGpuInputArrival(timestamp);
      }
    }
    return run().then(
      (result) => {
        this.logger?.recordInvocationEnd(timestamp);
        return result;
      },
      (error: unknown) => {
        this.logger?.recordInvocationEnd(timestamp);
        throw error;
      },
    );
  }

  /**
   * Maximum token window supported by the active model/session.
   *
   * @export
   */
  get contextWindow(): number {
    return this.maxNumTokens;
  }

  /**
   * Estimated token usage of the active schema and most recent evaluation.
   *
   * @export
   */
  get contextUsage(): number {
    return this.lastContextUsage;
  }

  /**
   * Updates the session-bound ClassifierSchema used by `classify()`.
   *
   * @export
   * @param schema The schema to bind to this session.
   */
  setSchema(schema: ClassifierSchema): void {
    this.sessionSchema = schema;
    this.lastContextUsage = this.computeSchemaTokens(schema);
  }

  /**
   * Prewarms the underlying engine KV/prefix cache for the given schema.
   *
   * @export
   * @param schema The schema whose shared prefix should be prewarmed.
   */
  async prewarm(schema: ClassifierSchema): Promise<void> {
    if (this.makerPtr === 0) return;
    const questions = schema.questions ?? [];
    if (questions.length === 0) return;
    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    if (typeof wasmModule.prewarmSchema === 'function') {
      await wasmModule.prewarmSchema(
        this.makerPtr,
        schema.context ?? '',
        questions,
      );
    } else if (typeof wasmModule.evaluateSchema === 'function') {
      await wasmModule.evaluateSchema(
        this.makerPtr,
        '__PREWARM__',
        schema.context ?? '',
        questions,
      );
    }
  }

  private computeSchemaTokens(schema?: ClassifierSchema): number {
    if (!schema) return 0;
    let total = schema.context ? estimateTokens(schema.context) : 0;
    for (const q of schema.questions ?? []) {
      total += estimateTokens(q.prompt) + 4;
      for (const opt of q.options ?? []) {
        total +=
          estimateTokens(
            opt.description ? `${opt.label}: ${opt.description}` : opt.label,
          ) + 2;
      }
    }
    return total;
  }

  /**
   * Estimates the total token usage for evaluating `input` (and optional
   * per-call `context`) against the session schema.
   *
   * @export
   * @param input The input text to be evaluated.
   * @param options Optional per-call evaluation options.
   */
  async measureContextUsage(
    input: string,
    options?: ClassifierEvaluateOptions,
  ): Promise<number> {
    const stateText = options?.context ? `${options.context}\n${input}` : input;
    return (
      this.computeSchemaTokens(this.sessionSchema) + estimateTokens(stateText)
    );
  }

  private allocateAlignedWasmBuffer(size: number): {
    ptr: number;
    rawPtr: number;
  } {
    const rawWasm = this.graphRunner.wasmModule;
    const allocSize = size >= 64 ? size + 16 : size;
    const rawPtr = rawWasm._malloc(allocSize) >>> 0;
    if (!rawPtr) {
      throw new Error(`Failed to allocate ${size} bytes on Wasm heap.`);
    }
    const ptr = size >= 64 ? ((rawPtr + 15) & ~15) >>> 0 : rawPtr;
    return {ptr, rawPtr};
  }

  /**
   * Streams an asset (from URL, ReadableStreamDefaultReader, or Uint8Array)
   * directly into 16-byte aligned Wasm linear memory (`_malloc(size + 16)`)
   * using the `WasmFileReference` / `StreamingReader` pattern so no large JS
   * ArrayBuffer or MEMFS copy is retained.
   */
  private async streamAssetToWasmHeap(
    assetPath?: string | string,
    assetBuffer?: Uint8Array | ReadableStreamDefaultReader,
    knownSize?: number,
  ): Promise<{ptr: number; rawPtr: number; size: number}> {
    const rawWasm = this.graphRunner.wasmModule;
    if (assetBuffer instanceof Uint8Array) {
      const size = assetBuffer.byteLength;
      if (size === 0) return {ptr: 0, rawPtr: 0, size: 0};
      const {ptr, rawPtr} = this.allocateAlignedWasmBuffer(size);
      rawWasm.HEAPU8.set(assetBuffer, ptr);
      return {ptr, rawPtr, size};
    }

    let reader: ReadableStreamDefaultReader | undefined = assetBuffer;
    let expectedSize = knownSize && knownSize > 0 ? knownSize : 0;

    if (!reader && assetPath) {
      const urlStr = assetPath.toString();
      const response = await fetch(urlStr);
      if (!response.ok || !response.body) {
        throw new Error(
          `Failed to fetch asset: ${urlStr} (${response.status})`,
        );
      }
      const headerLen = Number(response.headers.get('content-length') || 0);
      if (headerLen > 0) {
        expectedSize = headerLen;
      }
      reader = response.body.getReader();
    }

    if (!reader) {
      return {ptr: 0, rawPtr: 0, size: 0};
    }

    if (expectedSize > 0) {
      const {ptr, rawPtr} = this.allocateAlignedWasmBuffer(expectedSize);
      let offset = 0;
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        if (value && value.byteLength > 0) {
          const chunkLen = Number(value.byteLength);
          if (offset + chunkLen > expectedSize) {
            rawWasm._free(rawPtr);
            throw new Error(
              `Stream exceeded expected content-length (${expectedSize} bytes).`,
            );
          }
          rawWasm.HEAPU8.set(value, (ptr + offset) >>> 0);
          offset += chunkLen;
        }
      }
      return {ptr, rawPtr, size: offset};
    }

    // Fallback when content-length / knownSize is unavailable: accumulate
    // small stream chunks and release each chunk immediately as it is copied
    // to Wasm linear memory (StreamingReader DiscardableDataChunk pattern).
    const chunks: Array<Uint8Array | undefined> = [];
    let totalSize = 0;
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      if (value && value.byteLength > 0) {
        const chunkLen = Number(value.byteLength);
        chunks.push(value);
        totalSize += chunkLen;
      }
    }
    if (totalSize === 0) {
      return {ptr: 0, rawPtr: 0, size: 0};
    }
    const {ptr, rawPtr} = this.allocateAlignedWasmBuffer(totalSize);
    let offset = 0;
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      if (chunk) {
        rawWasm.HEAPU8.set(chunk, (ptr + offset) >>> 0);
        offset += chunk.byteLength;
        chunks[i] = undefined;
      }
    }
    return {ptr, rawPtr, size: totalSize};
  }

  /**
   * Sets the options for the DecisionMaker and initializes the underlying
   * engine. Options cannot be updated once the engine has been created.
   *
   * @export
   * @param options The options for the DecisionMaker.
   */
  override async setOptions(options: DecisionMakerOptions): Promise<void> {
    if (this.makerPtr !== 0) {
      throw new Error(
        'Decision does not support updating options after ' + 'initialization.',
      );
    }
    if (options.baseOptions?.delegate === 'CPU') {
      this.backendType = CPU_DELEGATE_BACKEND_TYPE;
    } else if (options.backendType !== undefined) {
      this.backendType = options.backendType;
    }
    if (options.maxNumTokens !== undefined) {
      this.maxNumTokens = options.maxNumTokens;
    }
    if (options.schema) {
      this.setSchema(options.schema);
    } else if (options.questions || options.context || options.expectedInputs) {
      this.setSchema({
        context: options.context,
        expectedInputs: options.expectedInputs,
        questions: options.questions,
      });
    }

    if (this.backendType === CPU_DELEGATE_BACKEND_TYPE) {
      // Use quoted property access: Emscripten reads this field by name
      // from the Module object, so it must not be renamed by the compiler.
      const wasmModule = this.graphRunner.wasmModule as unknown as Record<
        string,
        unknown
      >;
      wasmModule['preinitializedWebGPUDevice'] = undefined;
      const globalScope = (typeof self !== 'undefined'
        ? self
        : globalThis) as unknown as Record<string, unknown>;
      if (typeof globalScope['Module'] === 'object' && globalScope['Module']) {
        (globalScope['Module'] as Record<string, unknown>)[
          'preinitializedWebGPUDevice'
        ] = undefined;
      }
    } else if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        // Use quoted property access: Emscripten reads this field by name
        // from the Module object, so it must not be renamed by the compiler.
        const wasmModule = this.graphRunner.wasmModule as unknown as Record<
          string,
          unknown
        >;
        if (!wasmModule['preinitializedWebGPUDevice']) {
          const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
          });
          const adapterInfo = (
            adapter as unknown as {
              info?: {
                vendor?: string;
                architecture?: string;
                description?: string;
              };
            }
          )?.info;
          const isFallback = Boolean(
            (adapter as unknown as {isFallbackAdapter?: boolean})
              ?.isFallbackAdapter,
          );
          const isSoftwareAdapter =
            isFallback ||
            /swiftshader|llvmpipe|software|lavapipe/i.test(
              `${adapterInfo?.vendor ?? ''} ${adapterInfo?.architecture ?? ''} ${adapterInfo?.description ?? ''}`,
            );
          if (adapter && !isSoftwareAdapter) {
            const requiredFeatures: GPUFeatureName[] = [];
            for (const feat of [
              'shader-f16',
              'subgroups',
              'subgroups-f16',
            ] as GPUFeatureName[]) {
              if (adapter.features.has(feat)) {
                requiredFeatures.push(feat);
              }
            }
            const requiredLimits: Record<string, number> = {};
            for (const limitKey of [
              'maxBufferSize',
              'maxStorageBufferBindingSize',
              'maxStorageBuffersPerShaderStage',
              'maxComputeWorkgroupStorageSize',
            ]) {
              const val = (
                adapter.limits as unknown as Record<string, number | undefined>
              )[limitKey];
              if (typeof val === 'number') {
                requiredLimits[limitKey] = val;
              }
            }
            const device = await adapter.requestDevice({
              requiredFeatures,
              requiredLimits,
            });
            const deviceWithInfo = device as unknown as {adapterInfo?: unknown};
            if (!deviceWithInfo.adapterInfo && adapterInfo) {
              try {
                Object.defineProperty(device, 'adapterInfo', {
                  value: adapterInfo,
                  writable: true,
                  configurable: true,
                });
              } catch {
                try {
                  deviceWithInfo.adapterInfo = adapterInfo;
                } catch {}
              }
            }
            wasmModule['preinitializedWebGPUDevice'] = device;
          }
        }
        if (wasmModule['preinitializedWebGPUDevice']) {
          const globalScope = self as unknown as Record<string, unknown>;
          if (
            typeof globalScope['Module'] === 'object' &&
            globalScope['Module']
          ) {
            (globalScope['Module'] as Record<string, unknown>)[
              'preinitializedWebGPUDevice'
            ] = wasmModule['preinitializedWebGPUDevice'];
          }
        }
      } catch (e) {
        console.warn('WebGPU device initialization failed:', e);
      }
    }

    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    const rawWasm = this.graphRunner.wasmModule;

    const loadTfliteModel =
      !!options.baseOptions &&
      (!!options.baseOptions.modelAssetPath ||
        !!options.baseOptions.modelAssetBuffer);

    if (
      loadTfliteModel &&
      typeof wasmModule.createDecisionFromBuffers === 'function' &&
      typeof rawWasm._malloc === 'function'
    ) {
      const modelAlloc = await this.streamAssetToWasmHeap(
        options.baseOptions?.modelAssetPath,
        options.baseOptions?.modelAssetBuffer,
        options.modelAssetSize,
      );
      this.modelWasmPtr = modelAlloc.ptr;
      this.modelWasmRawPtr = modelAlloc.rawPtr;

      let pleSize = 0;
      if (options.companionPleAssetPath || options.companionPleAssetBuffer) {
        try {
          const pleAlloc = await this.streamAssetToWasmHeap(
            options.companionPleAssetPath,
            options.companionPleAssetBuffer,
            options.companionPleAssetSize,
          );
          this.pleWasmPtr = pleAlloc.ptr;
          this.pleWasmRawPtr = pleAlloc.rawPtr;
          pleSize = pleAlloc.size;
        } catch (e) {
          console.warn('Optional companion per-layer embedder not loaded:', e);
        }
      }

      this.makerPtr = await wasmModule.createDecisionFromBuffers(
        this.modelWasmPtr,
        modelAlloc.size,
        this.pleWasmPtr,
        pleSize,
        this.backendType,
        this.maxNumTokens,
      );
      this.logger?.logSessionStart();
      return;
    }

    await this.applyOptions(
      options,
      /* loadTfliteModel= */ loadTfliteModel,
      /* isLiteRtLmModel= */ true,
    );

    const modelPath = '/model.litertlm';

    this.makerPtr = await wasmModule.createDecision(
      modelPath,
      this.backendType,
      this.maxNumTokens,
    );
    this.logger?.logSessionStart();
  }

  protected override refreshGraph(): void {
    // Initialization is performed asynchronously in setOptions().
  }

  /**
   * Evaluates a Boolean Predicate condition against input text.
   *
   * @export
   * @param text The input text to be evaluated.
   * @param question The boolean question to evaluate.
   * @return The boolean evaluation result.
   */
  evaluateBoolean(
    text: string,
    question: BooleanQuestion,
  ): Promise<BooleanResult> {
    if (this.makerPtr === 0) {
      throw new Error('Decision is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    const condition = question.context
      ? `${question.context}\n${question.condition}`
      : question.condition;
    return this.logInvocation(() =>
      wasmModule.evaluateBoolean(
        this.makerPtr,
        text,
        condition,
        question.threshold ?? 0.5,
        question.temperature ?? 0,
      ),
    );
  }

  /**
   * Evaluates a Categorical Choice against input text.
   *
   * @export
   * @param text The input text to be evaluated.
   * @param question The choice question to evaluate.
   * @return The choice evaluation result.
   */
  evaluateChoice(
    text: string,
    question: ChoiceQuestion,
  ): Promise<ChoiceResult> {
    if (this.makerPtr === 0) {
      throw new Error('Decision is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    const criteria = resolveChoiceCriteria(question);
    const baseInstructions = question.instructions ?? '';
    const instructions = question.context
      ? baseInstructions
        ? `${question.context}\n${baseInstructions}`
        : question.context
      : baseInstructions;
    return this.logInvocation(() =>
      wasmModule.evaluateChoice(
        this.makerPtr,
        text,
        criteria,
        question.temperature ?? 0,
        instructions,
      ),
    );
  }

  /**
   * Evaluates an Ordinal Score rubric against input text.
   *
   * @export
   * @param text The input text to be evaluated.
   * @param question The score question to evaluate.
   * @return The score evaluation result.
   */
  async evaluateScore(
    text: string,
    question: ScoreQuestion,
  ): Promise<ScoreResult> {
    if (this.makerPtr === 0) {
      throw new Error('Decision is already closed.');
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    const {rubric, labels, levels, hasCustomLevels} =
      resolveScoreRubric(question);
    const baseInstructions = question.instructions ?? '';
    const instructions = question.context
      ? baseInstructions
        ? `${question.context}\n${baseInstructions}`
        : question.context
      : baseInstructions;
    const raw = await this.logInvocation(() =>
      wasmModule.evaluateScore(
        this.makerPtr,
        text,
        rubric,
        question.temperature ?? 0,
        instructions,
      ),
    );
    return enrichOrdinalResult(raw, labels, levels, hasCustomLevels);
  }

  /**
   * Evaluates a multi-question `ClassifierSchema` against `input` and returns
   * a unified `ClassifierResult` record keyed by question `id` (compatible
   * with `window.Classifier`).
   *
   * @export
   * @param input The input text to be evaluated.
   * @param schema The schema containing the questions to evaluate.
   * @param options Optional per-call evaluation options.
   * @return The results keyed by question `id`.
   */
  async evaluate(
    input: string,
    schema: ClassifierSchema,
    options?: ClassifierEvaluateOptions,
  ): Promise<ClassifierResult> {
    if (this.makerPtr === 0) {
      throw new Error('Decision is already closed.');
    }
    if (options?.signal?.aborted) {
      throw new DOMException('The operation was aborted.', 'AbortError');
    }
    const stateText = options?.context ? `${options.context}\n${input}` : input;
    const schemaContext = schema.context;
    this.lastContextUsage =
      this.computeSchemaTokens(schema) + estimateTokens(stateText);

    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    if (
      typeof wasmModule.evaluateSchema === 'function' &&
      (schema.questions ?? []).length > 0
    ) {
      return this.logInvocation(() =>
        wasmModule.evaluateSchema!(
          this.makerPtr,
          stateText,
          schemaContext ?? '',
          schema.questions ?? [],
        ),
      );
    }

    const output: ClassifierResult = {};
    for (const q of schema.questions ?? []) {
      if (q.type === 'binary' || q.type === 'boolean') {
        const res = await this.evaluateBoolean(stateText, {
          condition: q.prompt,
          context: schemaContext,
          threshold: q.threshold,
          temperature: q.temperature,
          normalizePrior: q.normalizePrior,
          options: q.options,
        });
        const pTrue = res.probabilityTrue;
        const threshold = q.threshold ?? 0.5;
        const falseLabel =
          q.options && q.options.length >= 2 ? q.options[0].label : 'false';
        const trueLabel =
          q.options && q.options.length >= 2 ? q.options[1].label : 'true';
        const label = pTrue >= threshold ? trueLabel : falseLabel;
        output[q.id] = {
          id: q.id,
          label,
          confidence: res.confidence,
          probability: pTrue,
          probabilities: [
            {label: falseLabel, probability: Math.max(0, 1 - pTrue)},
            {label: trueLabel, probability: pTrue},
          ],
        };
      } else if (q.type === 'categorical' || q.type === 'choice') {
        const res = await this.evaluateChoice(stateText, {
          options: q.options,
          instructions: q.prompt,
          context: schemaContext,
          temperature: q.temperature,
          normalizePrior: q.normalizePrior,
        });
        const orderedLabels = (q.options ?? []).map((o) => o.label);
        const probabilities: ClassifierOptionProbability[] =
          orderedLabels.length > 0
            ? orderedLabels.map((l) => ({
                label: l,
                probability: res.probabilities[l] ?? 0,
              }))
            : Object.entries(res.probabilities).map(([l, p]) => ({
                label: l,
                probability: p,
              }));
        output[q.id] = {
          id: q.id,
          label: res.selectedKey,
          confidence: res.confidence,
          probabilities,
          predictionSet: res.predictionSet,
        };
      } else if (q.type === 'ordinal' || q.type === 'score') {
        const res = await this.evaluateScore(stateText, {
          options: q.options,
          instructions: q.prompt,
          context: schemaContext,
          temperature: q.temperature,
        });
        const labels = (q.options ?? []).map((o) => o.label);
        const probabilities: ClassifierOptionProbability[] = (
          res.probabilities ?? []
        ).map((p, i) => ({
          label: i < labels.length ? labels[i] : String(i),
          probability: p,
        }));
        output[q.id] = {
          id: q.id,
          label: res.selectedKey ?? labels[0] ?? '0',
          confidence: res.confidence,
          expectedScore: res.expectedScore,
          probabilities,
        };
      }
    }
    return output;
  }

  /**
   * Evaluates a multi-question `ClassifierSchema` across a batch of `inputs`.
   *
   * @export
   * @param inputs The input texts to be evaluated.
   * @param schema The schema containing the questions to evaluate.
   * @param options Optional per-call evaluation options.
   * @return One result per input, in the same order as `inputs`.
   */
  async evaluateBatch(
    inputs: string[],
    schema: ClassifierSchema,
    options?: ClassifierEvaluateOptions,
  ): Promise<ClassifierResult[]> {
    const results: ClassifierResult[] = [];
    for (const input of inputs) {
      results.push(await this.evaluate(input, schema, options));
    }
    return results;
  }

  /**
   * Evaluates `input` against the session-bound `ClassifierSchema` configured
   * via `DecisionMakerOptions` or `setSchema()`.
   *
   * @export
   * @param input The input text to be evaluated.
   * @param options Optional per-call evaluation options.
   * @return The results keyed by question `id`.
   */
  async classify(
    input: string,
    options?: ClassifierEvaluateOptions,
  ): Promise<ClassifierResult> {
    if (!this.sessionSchema) {
      throw new Error(
        'No session schema configured. Pass `schema` or `questions` in ' +
          'DecisionMakerOptions or call setSchema() before classify().',
      );
    }
    return this.evaluate(input, this.sessionSchema, options);
  }

  /**
   * Evaluates a Boolean Predicate condition against a batch of input texts
   * (with an optional sharedPrefix for candidate batching).
   *
   * @export
   * @param texts The input texts to be evaluated.
   * @param question The boolean question to evaluate.
   * @param sharedPrefix An optional prefix shared by all inputs.
   * @return One result per input text.
   */
  async evaluateBooleanBatch(
    texts: string[],
    question: BooleanQuestion,
    sharedPrefix = '',
  ): Promise<BooleanResult[]> {
    if (this.makerPtr === 0) {
      throw new Error('Decision is already closed.');
    }
    if (texts.length === 0) {
      return [];
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    const condition = question.context
      ? `${question.context}\n${question.condition}`
      : question.condition;
    if (typeof wasmModule.evaluateBooleanBatch === 'function') {
      return this.logInvocation(() =>
        wasmModule.evaluateBooleanBatch!(
          this.makerPtr,
          texts,
          condition,
          question.threshold ?? 0.5,
          question.temperature ?? 0,
          sharedPrefix,
        ),
      );
    }
    const results: BooleanResult[] = [];
    for (const t of texts) {
      const merged = sharedPrefix ? `${sharedPrefix}\n${t}` : t;
      results.push(await this.evaluateBoolean(merged, question));
    }
    return results;
  }

  /**
   * Evaluates a Categorical Choice against a batch of input texts
   * (with an optional sharedPrefix for candidate batching).
   *
   * @export
   * @param texts The input texts to be evaluated.
   * @param question The choice question to evaluate.
   * @param sharedPrefix An optional prefix shared by all inputs.
   * @return One result per input text.
   */
  async evaluateChoiceBatch(
    texts: string[],
    question: ChoiceQuestion,
    sharedPrefix = '',
  ): Promise<ChoiceResult[]> {
    if (this.makerPtr === 0) {
      throw new Error('Decision is already closed.');
    }
    if (texts.length === 0) {
      return [];
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    const criteria = resolveChoiceCriteria(question);
    const baseInstructions = question.instructions ?? '';
    const instructions = question.context
      ? baseInstructions
        ? `${question.context}\n${baseInstructions}`
        : question.context
      : baseInstructions;
    if (typeof wasmModule.evaluateChoiceBatch === 'function') {
      return this.logInvocation(() =>
        wasmModule.evaluateChoiceBatch!(
          this.makerPtr,
          texts,
          criteria,
          question.temperature ?? 0,
          instructions,
          sharedPrefix,
        ),
      );
    }
    const results: ChoiceResult[] = [];
    for (const t of texts) {
      const merged = sharedPrefix ? `${sharedPrefix}\n${t}` : t;
      results.push(await this.evaluateChoice(merged, question));
    }
    return results;
  }

  /**
   * Evaluates an Ordinal Score rubric against a batch of input texts
   * (with an optional sharedPrefix for candidate batching).
   *
   * @export
   * @param texts The input texts to be evaluated.
   * @param question The score question to evaluate.
   * @param sharedPrefix An optional prefix shared by all inputs.
   * @return One result per input text.
   */
  async evaluateScoreBatch(
    texts: string[],
    question: ScoreQuestion,
    sharedPrefix = '',
  ): Promise<ScoreResult[]> {
    if (this.makerPtr === 0) {
      throw new Error('Decision is already closed.');
    }
    if (texts.length === 0) {
      return [];
    }
    const wasmModule = this.graphRunner
      .wasmModule as unknown as DecisionMakerWasmModule;
    const {rubric, labels, levels, hasCustomLevels} =
      resolveScoreRubric(question);
    const baseInstructions = question.instructions ?? '';
    const instructions = question.context
      ? baseInstructions
        ? `${question.context}\n${baseInstructions}`
        : question.context
      : baseInstructions;
    if (typeof wasmModule.evaluateScoreBatch === 'function') {
      const rawBatch = await this.logInvocation(() =>
        wasmModule.evaluateScoreBatch!(
          this.makerPtr,
          texts,
          rubric,
          question.temperature ?? 0,
          instructions,
          sharedPrefix,
        ),
      );
      return rawBatch.map((r) =>
        enrichOrdinalResult(r, labels, levels, hasCustomLevels),
      );
    }
    const results: ScoreResult[] = [];
    for (const t of texts) {
      const merged = sharedPrefix ? `${sharedPrefix}\n${t}` : t;
      results.push(await this.evaluateScore(merged, question));
    }
    return results;
  }

  /**
   * Closes the DecisionMaker and releases the native engine and model memory.
   *
   * @export
   */
  override close(): void {
    if (this.isClosed) return;
    this.isClosed = true;
    const rawWasm = this.graphRunner.wasmModule;
    const wasmModule = rawWasm as unknown as DecisionMakerWasmModule;
    if (this.makerPtr !== 0) {
      try {
        wasmModule.deleteDecision(this.makerPtr);
      } catch {}
      this.makerPtr = 0;
    }
    const modelFreePtr = this.modelWasmRawPtr || this.modelWasmPtr;
    if (modelFreePtr !== 0) {
      try {
        rawWasm._free(modelFreePtr);
      } catch {}
      this.modelWasmPtr = 0;
      this.modelWasmRawPtr = 0;
    }
    const pleFreePtr = this.pleWasmRawPtr || this.pleWasmPtr;
    if (pleFreePtr !== 0) {
      try {
        rawWasm._free(pleFreePtr);
      } catch {}
      this.pleWasmPtr = 0;
      this.pleWasmRawPtr = 0;
    }
    super.close();
    if (rawWasm && !recycledWasmModules.includes(rawWasm)) {
      recycledWasmModules.push(rawWasm);
    }
  }
}


