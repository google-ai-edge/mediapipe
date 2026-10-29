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
  TextToSpeech,
  TextToSpeechAudioChunk,
  TextToSpeechOptions,
  TextToSpeechResult,
  VoiceInfo,
} from './text_to_speech';

interface TextToSpeechInternals {
  logger?: TaskLogger;
}

class TextToSpeechFake extends TextToSpeech {
  mockModule: Record<string, jasmine.Spy>;

  static async create(
    options: TextToSpeechOptions = {},
  ): Promise<TextToSpeechFake> {
    const fake = new TextToSpeechFake();
    await fake.init(options);
    return fake;
  }

  constructor() {
    const mockVoices: VoiceInfo[] = [
      {
        voiceId: 'af_heart',
        displayName: 'Heart (en-US)',
        locale: 'en-US',
        sampleRateHz: 24000,
        isDownloaded: true,
      },
      {
        voiceId: 'ef_dora',
        displayName: 'Dora (es-ES)',
        locale: 'es-ES',
        sampleRateHz: 24000,
        isDownloaded: true,
      },
    ];

    const mockAudioResult: TextToSpeechResult = {
      samples: new Float32Array([0.1, 0.2, -0.1, -0.2]),
      sampleRateHz: 24000,
    };

    const chunks: Array<TextToSpeechAudioChunk | null> = [
      {
        samples: new Float32Array([0.1, 0.2]),
        sampleRateHz: 24000,
        chunkIndex: 0,
        isFinal: false,
      },
      {
        samples: new Float32Array([-0.1, -0.2]),
        sampleRateHz: 24000,
        chunkIndex: 1,
        isFinal: true,
      },
      null,
    ];
    let chunkCursor = 0;

    const module = {
      'FS_createDataFile': jasmine.createSpy('FS_createDataFile'),
      'FS_unlink': jasmine.createSpy('FS_unlink'),
      'createTextToSpeech': jasmine
        .createSpy('createTextToSpeech')
        .and.returnValue(Promise.resolve(54321)),
      'deleteTextToSpeech': jasmine.createSpy('deleteTextToSpeech'),
      'listTextToSpeechVoices': jasmine
        .createSpy('listTextToSpeechVoices')
        .and.returnValue(Promise.resolve(mockVoices)),
      'synthesizeTextToSpeech': jasmine
        .createSpy('synthesizeTextToSpeech')
        .and.returnValue(Promise.resolve(mockAudioResult)),
      'startTextToSpeechStream': jasmine
        .createSpy('startTextToSpeechStream')
        .and.callFake(() => {
          chunkCursor = 0;
          return Promise.resolve(98765);
        }),
      'readNextTextToSpeechChunk': jasmine
        .createSpy('readNextTextToSpeechChunk')
        .and.callFake(() => {
          const next =
            chunkCursor < chunks.length ? chunks[chunkCursor++] : null;
          return Promise.resolve(next);
        }),
      'pushTextToSpeechStream': jasmine
        .createSpy('pushTextToSpeechStream')
        .and.returnValue(Promise.resolve(true)),
      'finishTextToSpeechStreamInput': jasmine
        .createSpy('finishTextToSpeechStreamInput')
        .and.returnValue(Promise.resolve(true)),
      'cancelTextToSpeechStream': jasmine.createSpy('cancelTextToSpeechStream'),
      'deleteTextToSpeechStream': jasmine.createSpy('deleteTextToSpeechStream'),
      'cancelTextToSpeech': jasmine.createSpy('cancelTextToSpeech'),
      '_setAutoRenderToScreen': jasmine.createSpy('_setAutoRenderToScreen'),
      '_closeGraph': jasmine.createSpy('_closeGraph'),
    };
    super(module as unknown as WasmModule, null);
    this.mockModule = module as Record<string, jasmine.Spy>;
  }

  async init(options: TextToSpeechOptions = {}): Promise<void> {
    await this.setOptions(options);
  }
}

describe('TextToSpeech', () => {
  let tts: TextToSpeechFake;
  let mockModule: Record<string, jasmine.Spy>;

  beforeEach(async () => {
    tts = await TextToSpeechFake.create();
    mockModule = tts.mockModule;
  });

  afterEach(() => {
    if (tts) {
      tts.close();
    }
  });

  it('initializes with default options (CPU delegate by default)', () => {
    expect(tts).toBeDefined();
    expect(mockModule['createTextToSpeech']).toHaveBeenCalledWith(
      '/model.litertlm',
      -1,
      '',
      '',
      1.0,
      0,
      '',
    );
  });

  it('initializes with custom options and GPU delegate', async () => {
    const customTts = await TextToSpeechFake.create({
      baseOptions: {delegate: 'GPU'},
      defaultVoiceId: 'af_heart',
      defaultLocale: 'en-US',
      speakingRate: 1.25,
      targetBucket: 128,
      cacheDirectory: '/tts_cache',
    });
    expect(customTts.mockModule['createTextToSpeech']).toHaveBeenCalledWith(
      '/model.litertlm',
      0,
      'af_heart',
      'en-US',
      1.25,
      128,
      '/tts_cache',
    );
    customTts.close();
  });

  it('lists voices and synthesizes unary speech', async () => {
    const voices = await tts.listVoices();
    expect(mockModule['listTextToSpeechVoices']).toHaveBeenCalledWith(54321);
    expect(voices.length).toBe(2);
    expect(voices[0].voiceId).toBe('af_heart');

    const audio = await tts.synthesize('Hola mundo', {
      voiceId: 'ef_dora',
      locale: 'es-ES',
    });
    expect(mockModule['synthesizeTextToSpeech']).toHaveBeenCalledWith(
      54321,
      'Hola mundo',
      'ef_dora',
      'es-ES',
    );
    expect(audio.sampleRateHz).toBe(24000);
    expect(audio.samples.length).toBe(4);
  });

  it('supports pull-based and async-iterable streaming synthesis with incremental pushText', async () => {
    const stream = await tts.synthesizeStream(
      'First sentence.',
      {voiceId: 'af_heart'},
      /* finishInput= */ false,
    );
    expect(mockModule['startTextToSpeechStream']).toHaveBeenCalledWith(
      54321,
      'First sentence.',
      'af_heart',
      '',
      false,
    );

    await stream.pushText(' Second sentence.');
    expect(mockModule['pushTextToSpeechStream']).toHaveBeenCalledWith(
      98765,
      ' Second sentence.',
    );

    await stream.finishInput();
    expect(mockModule['finishTextToSpeechStreamInput']).toHaveBeenCalledWith(
      98765,
    );

    const collected: TextToSpeechAudioChunk[] = [];
    for await (const chunk of stream) {
      collected.push(chunk);
    }
    expect(collected.length).toBe(2);
    expect(collected[0].chunkIndex).toBe(0);
    expect(collected[0].isFinal).toBeFalse();
    expect(collected[1].chunkIndex).toBe(1);
    expect(collected[1].isFinal).toBeTrue();
    expect(mockModule['deleteTextToSpeechStream']).toHaveBeenCalledWith(98765);
  });

  it('cancels active synthesis and stream handles', async () => {
    const stream = await tts.synthesizeStream('Hello world.');
    stream.cancel();
    expect(mockModule['cancelTextToSpeechStream']).toHaveBeenCalledWith(98765);
    stream.close();

    tts.cancel();
    expect(mockModule['cancelTextToSpeech']).toHaveBeenCalledWith(54321);
  });

  it('deletes native pointer on close and throws when called after close', async () => {
    tts.close();
    expect(mockModule['deleteTextToSpeech']).toHaveBeenCalledWith(54321);
    expect(() => tts.synthesize('test')).toThrowError(
      'TextToSpeech is already closed.',
    );
    expect(() => tts.listVoices()).toThrowError(
      'TextToSpeech is already closed.',
    );
    await expectAsync(tts.synthesizeStream('test')).toBeRejectedWithError(
      'TextToSpeech is already closed.',
    );
  });

  it('streams model buffer into Wasm heap with zero-copy pointers and frees on close', async () => {
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
    const createFromBufferSpy = jasmine
      .createSpy('createTextToSpeechFromBuffer')
      .and.returnValue(Promise.resolve(88888));
    const deleteSpy = jasmine.createSpy('deleteTextToSpeech');
    const streamingModule: Record<string, unknown> = {
      'HEAPU8': fakeHeap,
      '_malloc': mallocSpy,
      '_free': freeSpy,
      'createTextToSpeechFromBuffer': createFromBufferSpy,
      'deleteTextToSpeech': deleteSpy,
      '_setAutoRenderToScreen': jasmine.createSpy('_setAutoRenderToScreen'),
      '_closeGraph': jasmine.createSpy('_closeGraph'),
    };
    const streamingTts = new TextToSpeech(
      streamingModule as unknown as WasmModule,
      null,
    );
    const modelBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    await streamingTts.setOptions({
      baseOptions: {
        modelAssetBuffer: modelBytes,
      },
      defaultVoiceId: 'af_heart',
    });
    expect(mallocSpy).toHaveBeenCalledWith(8);
    expect(createFromBufferSpy).toHaveBeenCalledWith(
      64,
      8,
      -1,
      'af_heart',
      '',
      1.0,
      0,
      '',
    );
    expect(Array.from(fakeHeap.slice(64, 72))).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8,
    ]);

    streamingTts.close();
    expect(deleteSpy).toHaveBeenCalledWith(88888);
    expect(freeSpy).toHaveBeenCalledWith(64);
  });

  it('frees model buffer if creation from buffer fails', async () => {
    const fakeHeap = new Uint8Array(1024);
    const mallocSpy = jasmine.createSpy('_malloc').and.returnValue(128);
    const freeSpy = jasmine.createSpy('_free');
    const failingCreateFromBufferSpy = jasmine
      .createSpy('createTextToSpeechFromBuffer')
      .and.rejectWith(new Error('Wasm init error'));
    const streamingModule: Record<string, unknown> = {
      'HEAPU8': fakeHeap,
      '_malloc': mallocSpy,
      '_free': freeSpy,
      'createTextToSpeechFromBuffer': failingCreateFromBufferSpy,
      '_setAutoRenderToScreen': jasmine.createSpy('_setAutoRenderToScreen'),
      '_closeGraph': jasmine.createSpy('_closeGraph'),
    };
    const streamingTts = new TextToSpeech(
      streamingModule as unknown as WasmModule,
      null,
    );
    await expectAsync(
      streamingTts.setOptions({
        baseOptions: {
          modelAssetBuffer: new Uint8Array([1, 2, 3, 4]),
        },
      }),
    ).toBeRejectedWithError('Wasm init error');
    expect(freeSpy).toHaveBeenCalledWith(128);
    streamingTts.close();
  });

  it('logs invocations and session lifecycle with TaskLogger', async () => {
    const fakeLogger = jasmine.createSpyObj<TaskLogger>('TaskLogger', [
      'logSessionStart',
      'logSessionEnd',
      'recordCpuInputArrival',
      'recordGpuInputArrival',
      'recordInvocationEnd',
      'close',
    ]);
    (tts as unknown as TextToSpeechInternals).logger = fakeLogger;

    await tts.synthesize('Hello world');
    expect(fakeLogger.recordCpuInputArrival).toHaveBeenCalledWith(0);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(0);

    tts.close();
    expect(fakeLogger.logSessionEnd).toHaveBeenCalledTimes(1);
    expect(fakeLogger.close).toHaveBeenCalledTimes(1);
  });
});
