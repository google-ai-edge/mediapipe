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
  AudioTranscriber,
  AudioTranscriberOptions,
  AudioTranscriberResult,
} from './audio_transcriber';

interface AudioTranscriberInternals {
  logger?: TaskLogger;
}

class AudioTranscriberFake extends AudioTranscriber {
  mockModule: Record<string, jasmine.Spy>;

  static async create(
    options: AudioTranscriberOptions = {},
  ): Promise<AudioTranscriberFake> {
    const fake = new AudioTranscriberFake();
    await fake.init(options);
    return fake;
  }

  constructor() {
    const mockTranscribeResult: AudioTranscriberResult = {
      segments: [
        {
          transcript: 'Hello world from MediaPipe Web.',
          isFinal: true,
          fromTimestampMs: 0,
          toTimestampMs: 1200,
        },
      ],
      text: 'Hello world from MediaPipe Web.',
    };

    const mockStreamResult: AudioTranscriberResult = {
      segments: [
        {
          transcript: 'Finalized stream segment.',
          isFinal: true,
          fromTimestampMs: 1200,
          toTimestampMs: 2400,
        },
      ],
      text: 'Finalized stream segment.',
    };

    const module = {
      'FS_createDataFile': jasmine.createSpy('FS_createDataFile'),
      'FS_unlink': jasmine.createSpy('FS_unlink'),
      'createAudioTranscriber': jasmine
        .createSpy('createAudioTranscriber')
        .and.returnValue(Promise.resolve(12345)),
      'deleteAudioTranscriber': jasmine.createSpy('deleteAudioTranscriber'),
      'transcribeAudio': jasmine
        .createSpy('transcribeAudio')
        .and.returnValue(Promise.resolve(mockTranscribeResult)),
      'finishAudioTranscriberStream': jasmine
        .createSpy('finishAudioTranscriberStream')
        .and.returnValue(Promise.resolve(mockStreamResult)),
      'prewarmAudioTranscriber': jasmine
        .createSpy('prewarmAudioTranscriber')
        .and.returnValue(Promise.resolve(true)),
      '_setAutoRenderToScreen': jasmine.createSpy('_setAutoRenderToScreen'),
      '_closeGraph': jasmine.createSpy('_closeGraph'),
    };
    super(module as unknown as WasmModule, null);
    this.mockModule = module as Record<string, jasmine.Spy>;
  }

  async init(options: AudioTranscriberOptions = {}): Promise<void> {
    await this.setOptions(options);
  }
}

describe('AudioTranscriber', () => {
  let transcriber: AudioTranscriberFake;
  let mockModule: Record<string, jasmine.Spy>;

  beforeEach(async () => {
    transcriber = await AudioTranscriberFake.create();
    mockModule = transcriber.mockModule;
  });

  afterEach(() => {
    if (transcriber) {
      transcriber.close();
    }
  });

  it('initializes with default options', () => {
    expect(transcriber).toBeDefined();
    expect(mockModule['createAudioTranscriber']).toHaveBeenCalledWith(
      '/model.litertlm',
      0,
      16000,
      false,
      0,
      '',
      '',
    );
  });

  it('initializes with custom options and CPU delegate', async () => {
    const customTranscriber = await AudioTranscriberFake.create({
      baseOptions: {delegate: 'CPU'},
      sampleRateHz: 24000,
      enableInterimResults: true,
      modelArchitecture: 'OMNI',
      prompt: 'Transcribe English audio:',
      cacheDir: '/cache',
    });
    expect(
      customTranscriber.mockModule['createAudioTranscriber'],
    ).toHaveBeenCalledWith(
      '/model.litertlm',
      -1,
      24000,
      true,
      1,
      'Transcribe English audio:',
      '/cache',
    );
    customTranscriber.close();
  });

  it('transcribes unary and streaming audio chunks and finishes stream', async () => {
    const pcm = new Float32Array([0.1, -0.1, 0.2, -0.2]);
    const unaryResult = await transcriber.transcribe(pcm);
    expect(mockModule['transcribeAudio']).toHaveBeenCalledWith(
      12345,
      pcm,
      true,
    );
    expect(unaryResult.text).toBe('Hello world from MediaPipe Web.');
    expect(unaryResult.segments.length).toBe(1);
    expect(unaryResult.segments[0].isFinal).toBeTrue();

    await transcriber.transcribeStream(pcm);
    expect(mockModule['transcribeAudio']).toHaveBeenCalledWith(
      12345,
      pcm,
      false,
    );

    const finalResult = await transcriber.finishStream();
    expect(mockModule['finishAudioTranscriberStream']).toHaveBeenCalledWith(
      12345,
    );
    expect(finalResult.text).toBe('Finalized stream segment.');
  });

  it('prewarms the underlying inference engine', async () => {
    await transcriber.prewarm();
    expect(mockModule['prewarmAudioTranscriber']).toHaveBeenCalledWith(12345);
  });

  it('deletes native pointer on close and throws when called after close', async () => {
    transcriber.close();
    expect(mockModule['deleteAudioTranscriber']).toHaveBeenCalledWith(12345);
    expect(() => transcriber.transcribe(new Float32Array([0.1]))).toThrowError(
      'AudioTranscriber is already closed.',
    );
    expect(() => transcriber.finishStream()).toThrowError(
      'AudioTranscriber is already closed.',
    );
    await expectAsync(transcriber.prewarm()).toBeRejectedWithError(
      'AudioTranscriber is already closed.',
    );
  });

  it('writes a model buffer to the Wasm filesystem and loads it by path', async () => {
    const modelBytes = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const bufferTranscriber = await AudioTranscriberFake.create({
      baseOptions: {modelAssetBuffer: modelBytes},
      sampleRateHz: 16000,
    });
    const module = bufferTranscriber.mockModule;
    expect(module['FS_createDataFile']).toHaveBeenCalledWith(
      '/',
      'model.litertlm',
      modelBytes,
      true,
      false,
      false,
    );
    expect(module['createAudioTranscriber']).toHaveBeenCalledWith(
      '/model.litertlm',
      0,
      16000,
      false,
      0,
      '',
      '',
    );
    bufferTranscriber.close();
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
    const streamTranscriber = await AudioTranscriberFake.create({
      baseOptions: {modelAssetBuffer: stream.getReader()},
    });
    const module = streamTranscriber.mockModule;
    expect(module['FS_createDataFile']).toHaveBeenCalledWith(
      '/',
      'model.litertlm',
      modelBytes,
      true,
      false,
      false,
    );
    expect(module['FS_createDataFile']).toHaveBeenCalledBefore(
      module['createAudioTranscriber'],
    );
    streamTranscriber.close();
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
    (transcriber as unknown as AudioTranscriberInternals).logger = fakeLogger;

    await transcriber.transcribe(new Float32Array([0.1, 0.2]));
    expect(fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(0);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(0);

    await transcriber.finishStream();
    expect(fakeLogger.recordGpuInputArrival).toHaveBeenCalledWith(1);
    expect(fakeLogger.recordInvocationEnd).toHaveBeenCalledWith(1);

    transcriber.close();
    expect(fakeLogger.logSessionEnd).toHaveBeenCalledTimes(1);
    expect(fakeLogger.close).toHaveBeenCalledTimes(1);
  });
});
