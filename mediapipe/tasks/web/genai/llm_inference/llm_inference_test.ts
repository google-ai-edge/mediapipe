/**
 * Copyright 2024 The MediaPipe Authors.
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

import {FilesetResolver} from '../../../../tasks/web/core/fileset_resolver';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {ProgressListener} from '../../../../web/graph_runner/graph_runner_llm_inference_lib';
// Placeholder for internal dependency on trusted resource URL builder
import {LlmInference} from './llm_inference';
import {LlmInferenceOptions} from './llm_inference_options';

// These tests take a long time to run.
jasmine.DEFAULT_TIMEOUT_INTERVAL = 100_000;

function countUniqueCharacters(text: string): number {
  // Ideally, we'd count unique tokens, but we don't have access to the
  // tokenizer here.
  const chars = new Set<string>();
  for (const char of text) {
    chars.add(char);
  }
  return chars.size;
}

describe('LlmInference', () => {
  describe('with converted LLM model', () => {
    const modelUrl = `http://localhost:8080/gemma3_int4_ekv1280.task`;
    let modelData: Uint8Array;
    let defaultOptions: LlmInferenceOptions;
    let genaiFileset: WasmFileset;
    let llmInference: LlmInference;

    function load(options = defaultOptions) {
      return LlmInference.createFromOptions(genaiFileset, options);
    }

    beforeAll(async () => {
      modelData = new Uint8Array(
        await (await fetch(modelUrl.toString())).arrayBuffer(),
      );
      genaiFileset = await FilesetResolver.forGenAiTasks();
    });

    beforeEach(async () => {
      defaultOptions = {
        baseOptions: {modelAssetBuffer: modelData},
        numResponses: 1,
        randomSeed: 1,
        topK: 10,
        temperature: 0.8,
      };
    });

    describe('loading', () => {
      // Tests that load the model are expensive, so we put them here.
      // The other tests reuse the same LlmInference instance.
      afterEach(() => {
        llmInference?.close();
        // This is needed to avoid accidentally calling close multiple times
        // on the same llmInference instance.
        (llmInference as unknown) = undefined;
      });

      it('loads a converted model', async () => {
        llmInference = await load();
        expect(llmInference).toBeDefined();
      });

      it('loads a model, deletes it, and then loads it again', async () => {
        llmInference = await load();

        const prompt = 'What is 4 + 5?';
        const response1 = await llmInference.generateResponse(prompt);

        llmInference.close();
        llmInference = await load();

        const response2 = await llmInference.generateResponse(prompt);

        expect(response1).toBeDefined();
        expect(typeof response1).toEqual('string');
        expect(response2).toBeDefined();
        expect(typeof response2).toEqual('string');
      });

      describe('sampler params', () => {
        // Move this to the running tests once we can pass sampler params
        // on each call.

        it('throws an error if maxTokens is too large', async () => {
          await expectAsync(
            load({...defaultOptions, maxTokens: 10_000}),
          ).toBeRejectedWithError(
            /Max number of tokens is larger than the maximum cache size supported/,
          );
        });

        it('throws an error if input is longer than maxTokens', async () => {
          const prompt = 'a'.repeat(1024);
          const llmInference = await load({...defaultOptions, maxTokens: 10});

          await expectAsync(
            llmInference.generateResponse(prompt),
          ).toBeRejectedWithError(/Input is too long/);
        });

        it('maxTokens affects sampling', async () => {
          const prompt = 'Please write an essay about the meaning of life.';
          llmInference = await load({
            ...defaultOptions,
            maxTokens: 30,
          });
          const fewTokensResponse = await llmInference.generateResponse(prompt);
          llmInference.close();

          llmInference = await load({
            ...defaultOptions,
            maxTokens: 1024,
          });
          const manyTokensResponse =
            await llmInference.generateResponse(prompt);

          expect(fewTokensResponse).not.toEqual(manyTokensResponse);
          expect(fewTokensResponse.length).toBeLessThan(
            manyTokensResponse.length,
          );
          // Note: We're measuring the number of characters, not tokens, since
          // we don't have access to the tokenizer here.
          expect(fewTokensResponse.length).toBeLessThan(100);
          expect(manyTokensResponse.length).toBeGreaterThan(100);
        });

        it('topk affects sampling', async () => {
          pending('Sampler is always set to TopP currently.');
          const prompt = 'hello';
          const lowTopKOptions: LlmInferenceOptions = {
            ...defaultOptions,
            topK: 1,
            temperature: 1,
          };
          llmInference = await load(lowTopKOptions);
          const lowTopKResponse = await llmInference.generateResponse(prompt);
          const lowTopKUniqueness =
            countUniqueCharacters(lowTopKResponse) / lowTopKResponse.length;

          llmInference.close();

          const highTopKOptions: LlmInferenceOptions = {
            ...defaultOptions,
            topK: 20,
            temperature: 1,
          };
          llmInference = await load(highTopKOptions);
          const highTopKResponse = await llmInference.generateResponse(prompt);
          const highTopKUniqueness =
            countUniqueCharacters(highTopKResponse) / highTopKResponse.length;

          expect(lowTopKResponse).not.toEqual(highTopKResponse);
          expect(lowTopKUniqueness).toBeLessThan(highTopKUniqueness);
        });

        it('temperature affects sampling', async () => {
          const prompt = 'hello';
          const lowTemperatureOptions: LlmInferenceOptions = {
            ...defaultOptions,
            temperature: 0,
          };
          llmInference = await load(lowTemperatureOptions);

          const lowTemperatureResponse =
            await llmInference.generateResponse(prompt);
          const lowTemperatureUniqueness =
            countUniqueCharacters(lowTemperatureResponse) /
            lowTemperatureResponse.length;

          llmInference.close();

          const highTemperatureOptions: LlmInferenceOptions = {
            ...defaultOptions,
            temperature: 1,
          };
          llmInference = await load(highTemperatureOptions);

          const highTemperatureResponse =
            await llmInference.generateResponse(prompt);
          const highTemperatureUniqueness =
            countUniqueCharacters(highTemperatureResponse) /
            highTemperatureResponse.length;

          expect(lowTemperatureResponse).not.toEqual(highTemperatureResponse);
          expect(lowTemperatureUniqueness).toBeLessThan(
            highTemperatureUniqueness,
          );
        });

        it('throws an error when numResponses is defined and not 1', async () => {
          await expectAsync(
            load({...defaultOptions, numResponses: 0}),
          ).toBeRejectedWithError(/numResponses.*must be at least 1/);

          await expectAsync(
            load({...defaultOptions, numResponses: 2}),
          ).toBeRejectedWithError(/numResponses > 1.*not supported/);
        });
      });

      it('throws an error when attempting to close while processing', async () => {
        llmInference = await load();

        const responsePromise = llmInference.generateResponse('what is 4 + 5?');

        expect(() => {
          llmInference.close();
        }).toThrowError(/currently loading or processing/);
        expect(typeof (await responsePromise)).toBe('string');
      });
    });

    describe('running', () => {
      beforeAll(async () => {
        llmInference = await load();
      });
      afterAll(() => {
        llmInference?.close();
      });

      it('counts the number of tokens', async () => {
        const prompt = 'Hello. This sentence has some number of tokens.';

        const count = llmInference.sizeInTokens(prompt);

        expect(count).toBeGreaterThan(0);
        expect(count).toBeLessThanOrEqual(prompt.length);
        expect(count).toBe(llmInference.sizeInTokens(prompt)); // Always the same.
      });

      it('throws an error if countTokens is called while processing', async () => {
        const responsePromise = llmInference.generateResponse('what is 4 + 5?');

        expect(() => {
          llmInference.sizeInTokens('asdf');
        }).toThrowError(/currently loading or processing/);

        // Wait for the response to finish. Otherwise, subsequent tests will
        // fail.
        await responsePromise;
      });

      it('generates a response', async () => {
        const prompt = 'What is 4 + 5?';

        const response = await llmInference.generateResponse(prompt);

        expect(response).toBeDefined();
        expect(typeof response).toEqual('string');
        // We don't test for the content of the response since this may be
        // using a fake weights model.
      });

      it('generates the same response given a seed', async () => {
        pending(
          'Due to prefix caching, this is not guaranteed to be the same.',
        );
        const prompt = 'What is 4 + 5?';

        const response1 = await llmInference.generateResponse(prompt);
        const response2 = await llmInference.generateResponse(prompt);

        expect(response1).toEqual(response2);
      });

      it('generates a response with a progress listener', async () => {
        const prompt = 'What is 4 + 5?';
        const responses: string[] = [];
        let progressListenerDone = false;
        const progressListener: ProgressListener = (partialResult, done) => {
          responses.push(partialResult);
          if (done) {
            progressListenerDone = true;
          }
        };

        const response = await llmInference.generateResponse(
          prompt,
          progressListener,
        );

        expect(responses.join('')).toEqual(response);
        expect(progressListenerDone).toBeTrue();
      });

      it('throws an error if called while processing', async () => {
        const prompt = 'What is 4 + 5?';

        const responsePromise = llmInference.generateResponse(prompt);

        await expectAsync(
          llmInference.generateResponse('this should fail'),
        ).toBeRejectedWithError(/currently loading or processing/);
        await expectAsync(responsePromise).toBeResolved();
        expect(typeof (await responsePromise)).toBe('string');
      });
    });
  });
});
