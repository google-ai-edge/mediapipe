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
    );
  });

  it('initializes with custom options and GPU delegate', async () => {
    const customTts = await TextToSpeechFake.create({
      baseOptions: {delegate: 'GPU'},
      defaultVoiceId: 'af_heart',
      defaultLocale: 'en-US',
      speakingRate: 1.25,
      targetBucket: 128,
    });
    expect(customTts.mockModule['createTextToSpeech']).toHaveBeenCalledWith(
      '/model.litertlm',
      0,
      'af_heart',
      'en-US',
      1.25,
      128,
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

  it('waits for more input when an incremental stream needs it', async () => {
    let pushed = 0;
    let finished = false;
    const chunk: TextToSpeechAudioChunk = {
      samples: new Float32Array([0.1, 0.2]),
      sampleRateHz: 24000,
      chunkIndex: 0,
      isFinal: false,
    };
    let delivered = 0;
    // The native stream reports `undefined` (needs input) until text arrives,
    // then returns one chunk per pushed sentence, and `null` once input is
    // finished and all chunks are delivered.
    mockModule['readNextTextToSpeechChunk'].and.callFake(() => {
      if (delivered < pushed) {
        ++delivered;
        return Promise.resolve(chunk);
      }
      return Promise.resolve(finished ? null : undefined);
    });
    mockModule['pushTextToSpeechStream'].and.callFake(() => {
      ++pushed;
      return Promise.resolve(true);
    });
    mockModule['finishTextToSpeechStreamInput'].and.callFake(() => {
      finished = true;
      return Promise.resolve(true);
    });

    const stream = await tts.synthesizeStream(
      '',
      {voiceId: 'af_heart'},
      /* finishInput= */ false,
    );
    const first: {done: boolean; chunk?: TextToSpeechAudioChunk | null} = {
      done: false,
    };
    const firstRead = stream.readNext().then((c) => {
      first.done = true;
      first.chunk = c;
    });
    // Let the read reach the native stream and start waiting for input.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(first.done).toBeFalse();

    await stream.pushText('Tokens arrive one by one.');
    await firstRead;
    expect(first.chunk).toEqual(chunk);

    const rest: TextToSpeechAudioChunk[] = [];
    const reading = (async () => {
      for await (const c of stream) {
        rest.push(c);
      }
    })();
    await stream.pushText(' Then we finish.');
    await stream.finishInput();
    await reading;
    expect(rest.length).toBe(1);
    expect(mockModule['deleteTextToSpeechStream']).toHaveBeenCalledWith(98765);
  });

  it('wakes a stream that waits for input when the task is cancelled', async () => {
    mockModule['readNextTextToSpeechChunk'].and.returnValue(
      Promise.resolve(undefined),
    );
    const stream = await tts.synthesizeStream('', {}, /* finishInput= */ false);
    const read = stream.readNext();
    await new Promise((resolve) => setTimeout(resolve, 0));
    // A cancelled native stream throws on the next read.
    mockModule['readNextTextToSpeechChunk'].and.callFake(() =>
      Promise.reject(new Error('Synthesis cancelled.')),
    );
    tts.cancel();
    await expectAsync(read).toBeRejectedWithError('Synthesis cancelled.');
    stream.close();
  });

  it('cancels active synthesis and stream handles', async () => {
    const stream = await tts.synthesizeStream('Hello world.');
    stream.cancel();
    expect(mockModule['cancelTextToSpeechStream']).toHaveBeenCalledWith(98765);
    stream.close();

    tts.cancel();
    expect(mockModule['cancelTextToSpeech']).toHaveBeenCalledWith(54321);
  });

  it('opens an incremental stream via openStream with finishInput=false by default', async () => {
    const stream = await tts.openStream('', {voiceId: 'af_heart'});
    expect(mockModule['startTextToSpeechStream']).toHaveBeenCalledWith(
      54321,
      '',
      'af_heart',
      '',
      false,
    );
    await stream.pushText('Hello ');
    expect(mockModule['pushTextToSpeechStream']).toHaveBeenCalledWith(
      98765,
      'Hello ',
    );
    await stream.finishInput();
    expect(mockModule['finishTextToSpeechStreamInput']).toHaveBeenCalledWith(
      98765,
    );
    stream.close();
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

  it('closes active streams before deleting the engine on close', async () => {
    const deletions: string[] = [];
    mockModule['deleteTextToSpeechStream'].and.callFake(() => {
      deletions.push('stream');
    });
    mockModule['deleteTextToSpeech'].and.callFake(() => {
      deletions.push('engine');
    });
    mockModule['readNextTextToSpeechChunk'].and.returnValue(
      Promise.resolve(undefined),
    );
    const stream = await tts.synthesizeStream('', {}, /* finishInput= */ false);
    const read = stream.readNext();
    // Let the read reach the native stream and start waiting for input.
    await new Promise((resolve) => setTimeout(resolve, 0));

    tts.close();

    expect(mockModule['deleteTextToSpeechStream']).toHaveBeenCalledWith(98765);
    expect(deletions).toEqual(['stream', 'engine']);
    // The pending read wakes up and observes the closed stream.
    await expectAsync(read).toBeResolvedTo(null);
    await expectAsync(stream.pushText('late')).toBeRejectedWithError(
      'TextToSpeechStream is already closed.',
    );
    // Closing the stream again must not touch the recycled Wasm module.
    stream.close();
    expect(mockModule['deleteTextToSpeechStream']).toHaveBeenCalledTimes(1);
  });

  it('writes a model buffer to the Wasm filesystem and loads it by path', async () => {
    const modelBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const bufferTts = await TextToSpeechFake.create({
      baseOptions: {modelAssetBuffer: modelBytes},
      defaultVoiceId: 'af_heart',
    });
    const module = bufferTts.mockModule;
    expect(module['FS_createDataFile']).toHaveBeenCalledWith(
      '/',
      'model.litertlm',
      modelBytes,
      true,
      false,
      true,
    );
    expect(module['createTextToSpeech']).toHaveBeenCalledWith(
      '/model.litertlm',
      -1,
      'af_heart',
      '',
      1.0,
      0,
    );
    bufferTts.close();
  });

  it('reads a model stream into the Wasm filesystem before creation', async () => {
    const modelBytes = new Uint8Array([1, 2, 3, 4, 5, 6]);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(modelBytes.subarray(0, 4));
        controller.enqueue(modelBytes.subarray(4));
        controller.close();
      },
    });
    const streamTts = await TextToSpeechFake.create({
      baseOptions: {modelAssetBuffer: stream.getReader()},
    });
    const module = streamTts.mockModule;
    expect(module['FS_createDataFile']).toHaveBeenCalledWith(
      '/',
      'model.litertlm',
      modelBytes,
      true,
      false,
      true,
    );
    expect(module['FS_createDataFile']).toHaveBeenCalledBefore(
      module['createTextToSpeech'],
    );
    streamTts.close();
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
