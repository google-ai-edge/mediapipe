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
jasmine.DEFAULT_TIMEOUT_INTERVAL = 180_000;

describe('LlmInference', () => {
  const cases = [
    ['handwritten', `http://localhost:8080/gemma_i4.tflite`],
    [
      'converted',
      `http://localhost:8080/gemma3_it_gpu_f32_ekv4096_dynamic_wi4_final_153192382_type_pt.task`,
    ],
  ];
  for (const [caseName, modelUrl] of cases) {
    describe(`with ${caseName} LLM model`, () => {
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

        it('loads a model', async () => {
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
          if (caseName === 'converted') {
            // Converted models have their maxTokens hardcoded in the model
            // file.
            it('throws an error if maxTokens is too large', async () => {
              await expectAsync(
                load({...defaultOptions, maxTokens: 10_000}),
              ).toBeRejectedWithError(
                /Max number of tokens is larger than the maximum cache size supported/,
              );
            });
          }

          it('throws an error if input is longer than maxTokens', async () => {
            const prompt = 'a'.repeat(1024);
            const llmInference = await load({...defaultOptions, maxTokens: 10});

            // Handwritten throws an error immediately while converted rejects
            // the promise.
            try {
              await llmInference.generateResponse(prompt);
              fail('Expected error');
            } catch (e: unknown) {
              expect(e).toBeInstanceOf(Error);
              expect((e as Error).message).toContain('Input is too long');
            }
          });

          it('maxTokens affects sampling', async () => {
            const prompt = 'Please write an essay about the meaning of life.';
            llmInference = await load({
              ...defaultOptions,
              maxTokens: 30,
            });
            const fewTokensResponse =
              await llmInference.generateResponse(prompt);
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
            const prompt = 'Write a poem about a cat.';
            async function run(topK: number, seed: number) {
              const options: LlmInferenceOptions = {
                ...defaultOptions,
                topK,
                randomSeed: seed,
              };
              llmInference = await load(options);
              const response = await llmInference.generateResponse(prompt);
              llmInference.close();
              return response;
            }

            const topK1Response1 = await run(1, defaultOptions.randomSeed!);
            const topK1Response2 = await run(1, defaultOptions.randomSeed! + 1);
            const topK20Response1 = await run(20, defaultOptions.randomSeed!);
            const topK20Response2 = await run(
              20,
              defaultOptions.randomSeed! + 1,
            );

            // This is needed to avoid accidentally calling close multiple times
            // on the same llmInference instance.
            (llmInference as unknown) = undefined;

            // TODONT: Previously, this test contained a statistical
            // check that higher values of topK would result in more
            // unique responses. However, uniqueness was measured by
            // unique characters, not tokens, so this test was flaky.
            // A better test would use controlled decoding to measure the
            // likelihood of a given response.

            expect(topK1Response1)
              .withContext('TopK=1 does not change with seed')
              .toEqual(topK1Response2);
            expect(topK20Response1)
              .withContext('TopK=20 changes with seed')
              .not.toEqual(topK20Response2);
            expect(topK1Response1)
              .withContext('Changing TopK changes the output')
              .not.toEqual(topK20Response1);
          });

          it('temperature affects sampling', async () => {
            const prompt = 'Write a poem about a cat.';
            async function run(temperature: number, seed: number) {
              const options: LlmInferenceOptions = {
                ...defaultOptions,
                temperature,
                randomSeed: seed,
              };
              llmInference = await load(options);
              const response = await llmInference.generateResponse(prompt);
              llmInference.close();
              return response;
            }

            const temp0Response1 = await run(0, defaultOptions.randomSeed!);
            const temp0Response2 = await run(0, defaultOptions.randomSeed! + 1);
            const temp1Response1 = await run(1, defaultOptions.randomSeed!);
            const temp1Response2 = await run(1, defaultOptions.randomSeed! + 1);

            // This is needed to avoid accidentally calling close multiple times
            // on the same llmInference instance.
            (llmInference as unknown) = undefined;

            // TODONT: Previously, this test contained a statistical
            // check that higher values of temperature would result in
            // more unique responses. However, uniqueness was measured by
            // unique characters, not tokens, so this test was flaky.
            // A better test would use controlled decoding to measure the
            // likelihood of a given response.

            expect(temp0Response1)
              .withContext('Temperature=0 does not change with seed')
              .toEqual(temp0Response2);
            expect(temp1Response1)
              .withContext('Temperature=1 changes with seed')
              .not.toEqual(temp1Response2);
            expect(temp0Response1)
              .withContext('Changing Temperature changes the output')
              .not.toEqual(temp1Response1);
          });

          it('temperature does not affect sampling when topk is 1', async () => {
            const prompt = 'Write a poem about a cat.';
            async function run(temperature: number, seed: number) {
              const options: LlmInferenceOptions = {
                ...defaultOptions,
                temperature,
                topK: 1,
                randomSeed: seed,
              };
              llmInference = await load(options);
              const response = await llmInference.generateResponse(prompt);
              llmInference.close();
              return response;
            }
            const lowTemperatureResponse = await run(
              0,
              defaultOptions.randomSeed!,
            );
            const highTemperatureResponse = await run(
              1,
              defaultOptions.randomSeed!,
            );

            // This is needed to avoid accidentally calling close multiple times
            // on the same llmInference instance.
            (llmInference as unknown) = undefined;

            expect(lowTemperatureResponse).toEqual(highTemperatureResponse);
          });

          if (caseName === 'converted') {
            // Converted models only support one response at a time for now.
            // Remove this once we support multiple responses.
            it('throws an error when numResponses is defined and not 1', async () => {
              await expectAsync(
                load({...defaultOptions, numResponses: 0}),
              ).toBeRejectedWithError(/numResponses.*must be at least 1/);

              await expectAsync(
                load({...defaultOptions, numResponses: 2}),
              ).toBeRejectedWithError(/numResponses > 1.*not supported/);
            });
          }
        });

        it('throws an error when attempting to close while processing', async () => {
          if (caseName === 'handwritten') {
            pending('TODO: msoulanille - Fix this test for handwritten model.');
          }
          llmInference = await load();

          const responsePromise =
            llmInference.generateResponse('what is 4 + 5?');

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
          const responsePromise =
            llmInference.generateResponse('what is 4 + 5?');

          expect(() => {
            llmInference.sizeInTokens('asdf');
          }).toThrowError(
            /[currently loading or processing|invocation.*still ongoing]/,
          );

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
          if (caseName === 'handwritten') {
            pending('TODO: msoulanille - Fix this test for handwritten model.');
          }
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
  }
});
