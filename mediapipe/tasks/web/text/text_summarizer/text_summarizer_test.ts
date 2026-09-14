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
import {SummarizationMode, TextSummarizer} from './text_summarizer';

class TextSummarizerFake extends TextSummarizer {
  mockModule: Record<string, jasmine.Spy>;

  constructor(mode: SummarizationMode = 'KEYPOINTS', maxNumTokens = 0) {
    const module = {
      'FS_createDataFile': jasmine.createSpy('FS_createDataFile'),
      'FS_unlink': jasmine.createSpy('FS_unlink'),
      'createTextSummarizer': jasmine
        .createSpy('createTextSummarizer')
        .and.returnValue(12345),
      'deleteTextSummarizer': jasmine.createSpy('deleteTextSummarizer'),
      'summarize': jasmine.createSpy('summarize').and.returnValue('summary'),
      '_setAutoRenderToScreen': jasmine.createSpy('_setAutoRenderToScreen'),
      '_closeGraph': jasmine.createSpy('_closeGraph'),
    };
    super(module as unknown as WasmModule, null);
    this.mockModule = module as Record<string, jasmine.Spy>;
    this.setOptions({mode, maxNumTokens});
  }
}

describe('TextSummarizer', () => {
  let textSummarizer: TextSummarizerFake;
  // Use a generic record for the mock module to allow adding spies.
  let mockModule: Record<string, jasmine.Spy>;

  beforeEach(async () => {
    textSummarizer = new TextSummarizerFake();
    mockModule = textSummarizer.mockModule;
  });

  afterEach(() => {
    if (textSummarizer) {
      textSummarizer.close();
    }
  });

  it('initializes with default mode KEYPOINTS', () => {
    expect(textSummarizer).toBeDefined();
    expect(mockModule['createTextSummarizer']).toHaveBeenCalledWith(
      '/model.litertlm',
      0,
      'KEYPOINTS',
    );
  });

  it('initializes with TLDR mode', () => {
    const tldrSummarizer = new TextSummarizerFake('TLDR');
    expect(tldrSummarizer).toBeDefined();
    expect(
      tldrSummarizer.mockModule['createTextSummarizer'],
    ).toHaveBeenCalledWith('/model.litertlm', 0, 'TLDR');
  });

  it('initializes with custom maxNumTokens', () => {
    const customSummarizer = new TextSummarizerFake('KEYPOINTS', 512);
    expect(customSummarizer).toBeDefined();
    expect(
      customSummarizer.mockModule['createTextSummarizer'],
    ).toHaveBeenCalledWith('/model.litertlm', 512, 'KEYPOINTS');
  });

  it('summarizes text', () => {
    const result = textSummarizer.summarize('text');
    expect(mockModule['summarize']).toHaveBeenCalledWith(12345, 'text');
    expect(result).toBe('summary');
  });

  it('deletes summarizer pointer on close', () => {
    textSummarizer.close();
    expect(mockModule['deleteTextSummarizer']).toHaveBeenCalledWith(12345);
  });

  it('throws error when summarizing after close', () => {
    textSummarizer.close();
    expect(() => textSummarizer.summarize('text')).toThrowError(
      'TextSummarizer is already closed.',
    );
  });
});
