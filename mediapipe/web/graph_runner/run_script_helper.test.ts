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

import 'jasmine';

import {runScript} from '../../web/graph_runner/run_script_helper';

describe('runScript()', () => {
  const globalDocument =
      Object.getOwnPropertyDescriptor(globalThis, 'document');

  afterEach(() => {
    if (globalDocument) {
      Object.defineProperty(globalThis, 'document', globalDocument);
    } else {
      delete (globalThis as {document?: Document}).document;
    }
  });

  it('rejects with an Error when a script fails to load', async () => {
    const scriptUrl = 'https://example.com/vision_wasm_internal.js';
    const loadEvent = {type: 'error'} as Event;
    let errorListener: EventListener|undefined;
    const script = {
      addEventListener: (type: string, listener: EventListener) => {
        if (type === 'error') {
          errorListener = listener;
        }
      },
      crossOrigin: '',
      src: '',
    } as HTMLScriptElement;
    const documentMock = {
      body: {
        appendChild: () => {
          errorListener?.(loadEvent);
        },
      },
      createElement: () => script,
    } as unknown as Document;
    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: documentMock,
    });

    try {
      await runScript(scriptUrl);
      fail('Expected runScript() to reject.');
    } catch (error) {
      expect(error).toEqual(jasmine.any(Error));
      expect((error as Error).message).toContain(scriptUrl);
      expect((error as Error&{cause: unknown}).cause).toBe(loadEvent);
    }
  });
});
