/**
 * Copyright 2025 The MediaPipe Authors.
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

import {
  createModelStream,
  ModelLoader,
} from './efficient_model_loader';

describe('EfficientModelLoader', () => {
  let mockFetch: jasmine.Spy;
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    mockFetch = jasmine.createSpy('fetch');
    globalThis.fetch = mockFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe('createModelStream', () => {
    it('should create a stream from a successful fetch', async () => {
      const mockData = new Uint8Array([1, 2, 3, 4, 5]);
      const mockResponse = {
        ok: true,
        status: 200,
        body: new ReadableStream({
          start(controller) {
            controller.enqueue(mockData);
            controller.close();
          },
        }),
      };
      mockFetch.and.returnValue(Promise.resolve(mockResponse));

      const stream = await createModelStream('http://example.com/model.bin');
      const { value } = await stream.read();

      expect(value).toEqual(mockData);
    });

    it('should throw error for failed fetch', async () => {
      const mockResponse = {
        ok: false,
        status: 404,
      };
      mockFetch.and.returnValue(Promise.resolve(mockResponse));

      await expectAsync(
        createModelStream('http://example.com/nonexistent.bin')
      ).toBeRejectedWithError(/Failed to fetch model.*404/);
    });
  });



  describe('ModelLoader', () => {
    let loader: ModelLoader;

    beforeEach(() => {
      loader = new ModelLoader();
    });

    afterEach(() => {
      loader.cancel();
    });

    it('should load a model successfully', async () => {
      const mockData = new Uint8Array([1, 2, 3]);
      const mockResponse = {
        ok: true,
        status: 200,
        body: new ReadableStream({
          start(controller) {
            controller.enqueue(mockData);
            controller.close();
          },
        }),
      };
      mockFetch.and.returnValue(Promise.resolve(mockResponse));

      const stream = await loader.loadModel('http://example.com/model.bin');
      const { value } = await stream.read();

      expect(value).toEqual(mockData);
    });

    it('should track loading state', async () => {
      mockFetch.and.returnValue(new Promise(() => {})); // Never resolves

      expect(loader.isLoading()).toBeFalse();

      const loadPromise = loader.loadModel('http://example.com/model.bin');
      expect(loader.isLoading()).toBeTrue();

      loader.cancel();
      await expectAsync(loadPromise).toBeRejected();
      expect(loader.isLoading()).toBeFalse();
    });

    it('should cancel previous loading when starting new load', async () => {
      mockFetch.and.returnValue(new Promise(() => {})); // Never resolves

      const firstLoad = loader.loadModel('http://example.com/model1.bin');
      expect(loader.isLoading()).toBeTrue();

      loader.cancel();
      expect(loader.isLoading()).toBeFalse();

      await expectAsync(firstLoad).toBeRejected();
    });



    it('should handle loading errors gracefully', async () => {
      mockFetch.and.returnValue(Promise.reject(new Error('Network failure')));

      await expectAsync(
        loader.loadModel('http://example.com/model.bin')
      ).toBeRejected();

      expect(loader.isLoading()).toBeFalse();
    });
  });
});