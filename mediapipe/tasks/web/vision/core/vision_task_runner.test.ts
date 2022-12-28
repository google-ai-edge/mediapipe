/**
 * Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {createSpyWasmModule} from '../../../../tasks/web/core/task_runner_test_utils';
import {ImageSource} from '../../../../web/graph_runner/graph_runner';

import {VisionTaskOptions} from './vision_task_options';
import {VisionTaskRunner} from './vision_task_runner';

class VisionTaskRunnerFake extends VisionTaskRunner<void> {
  baseOptions = new BaseOptionsProto();

  constructor() {
    super(createSpyWasmModule(), /* glCanvas= */ null);
  }

  protected override process(): void {}

  protected override refreshGraph(): void {}

  override setOptions(options: VisionTaskOptions): Promise<void> {
    return this.applyOptions(options);
  }

  override processImageData(image: ImageSource): void {
    super.processImageData(image);
  }

  override processVideoData(imageFrame: ImageSource, timestamp: number): void {
    super.processVideoData(imageFrame, timestamp);
  }
}

describe('VisionTaskRunner', () => {
  let visionTaskRunner: VisionTaskRunnerFake;

  beforeEach(async () => {
    visionTaskRunner = new VisionTaskRunnerFake();
    await visionTaskRunner.setOptions(
        {baseOptions: {modelAssetBuffer: new Uint8Array([])}});
  });

  it('can enable image mode', async () => {
    await visionTaskRunner.setOptions({runningMode: 'image'});
    expect(visionTaskRunner.baseOptions.toObject())
        .toEqual(jasmine.objectContaining({useStreamMode: false}));
  });

  it('can enable video mode', async () => {
    await visionTaskRunner.setOptions({runningMode: 'video'});
    expect(visionTaskRunner.baseOptions.toObject())
        .toEqual(jasmine.objectContaining({useStreamMode: true}));
  });

  it('can clear running mode', async () => {
    await visionTaskRunner.setOptions({runningMode: 'video'});

    // Clear running mode
    await visionTaskRunner.setOptions({runningMode: undefined});
    expect(visionTaskRunner.baseOptions.toObject())
        .toEqual(jasmine.objectContaining({useStreamMode: false}));
  });

  it('cannot process images with video mode', async () => {
    await visionTaskRunner.setOptions({runningMode: 'video'});
    expect(() => {
      visionTaskRunner.processImageData({} as HTMLImageElement);
    }).toThrowError(/Task is not initialized with image mode./);
  });

  it('cannot process video with image mode', async () => {
    // Use default for `useStreamMode`
    expect(() => {
      visionTaskRunner.processVideoData({} as HTMLImageElement, 42);
    }).toThrowError(/Task is not initialized with video mode./);

    // Explicitly set to image mode
    await visionTaskRunner.setOptions({runningMode: 'image'});
    expect(() => {
      visionTaskRunner.processVideoData({} as HTMLImageElement, 42);
    }).toThrowError(/Task is not initialized with video mode./);
  });
});
