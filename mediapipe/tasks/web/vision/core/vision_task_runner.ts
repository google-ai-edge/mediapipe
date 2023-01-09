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

import {NormalizedRect} from '../../../../framework/formats/rect_pb';
import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {ImageProcessingOptions} from '../../../../tasks/web/vision/core/image_processing_options';
import {GraphRunner, ImageSource} from '../../../../web/graph_runner/graph_runner';
import {SupportImage} from '../../../../web/graph_runner/graph_runner_image_lib';
import {SupportModelResourcesGraphService} from '../../../../web/graph_runner/register_model_resources_graph_service';

import {VisionTaskOptions} from './vision_task_options';

// tslint:disable-next-line:enforce-name-casing
const GraphRunnerVisionType =
    SupportModelResourcesGraphService(SupportImage(GraphRunner));
/** An implementation of the GraphRunner that supports image operations */
export class VisionGraphRunner extends GraphRunnerVisionType {}

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/** Base class for all MediaPipe Vision Tasks. */
export abstract class VisionTaskRunner extends TaskRunner {
  /**
   * Constructor to initialize a `VisionTaskRunner`.
   *
   * @param graphRunner the graph runner for this task.
   * @param imageStreamName the name of the input image stream.
   * @param normRectStreamName the name of the input normalized rect image
   *     stream used to provide (mandatory) rotation and (optional)
   *     region-of-interest.
   *
   * @hideconstructor protected
   */
  constructor(
      protected override readonly graphRunner: VisionGraphRunner,
      private readonly imageStreamName: string,
      private readonly normRectStreamName: string) {
    super(graphRunner);
  }

  /** Configures the shared options of a vision task. */
  override applyOptions(options: VisionTaskOptions): Promise<void> {
    if ('runningMode' in options) {
      const useStreamMode =
          !!options.runningMode && options.runningMode !== 'IMAGE';
      this.baseOptions.setUseStreamMode(useStreamMode);
    }
    return super.applyOptions(options);
  }

  /** Sends a single image to the graph and awaits results. */
  protected processImageData(
      image: ImageSource,
      imageProcessingOptions: ImageProcessingOptions|undefined): void {
    if (!!this.baseOptions?.getUseStreamMode()) {
      throw new Error(
          'Task is not initialized with image mode. ' +
          '\'runningMode\' must be set to \'IMAGE\'.');
    }

    // Increment the timestamp by 1 millisecond to guarantee that we send
    // monotonically increasing timestamps to the graph.
    const syntheticTimestamp = this.getLatestOutputTimestamp() + 1;
    this.process(image, imageProcessingOptions, syntheticTimestamp);
  }

  /** Sends a single video frame to the graph and awaits results. */
  protected processVideoData(
      imageFrame: ImageSource,
      imageProcessingOptions: ImageProcessingOptions|undefined,
      timestamp: number): void {
    if (!this.baseOptions?.getUseStreamMode()) {
      throw new Error(
          'Task is not initialized with video mode. ' +
          '\'runningMode\' must be set to \'VIDEO\'.');
    }
    this.process(imageFrame, imageProcessingOptions, timestamp);
  }

  private convertToNormalizedRect(imageProcessingOptions?:
                                      ImageProcessingOptions): NormalizedRect {
    const normalizedRect = new NormalizedRect();

    if (imageProcessingOptions?.regionOfInterest) {
      const roi = imageProcessingOptions.regionOfInterest;

      if (roi.left >= roi.right || roi.top >= roi.bottom) {
        throw new Error('Expected RectF with left < right and top < bottom.');
      }
      if (roi.left < 0 || roi.top < 0 || roi.right > 1 || roi.bottom > 1) {
        throw new Error('Expected RectF values to be in [0,1].');
      }

      normalizedRect.setXCenter((roi.left + roi.right) / 2.0);
      normalizedRect.setYCenter((roi.top + roi.bottom) / 2.0);
      normalizedRect.setWidth(roi.right - roi.left);
      normalizedRect.setHeight(roi.bottom - roi.top);
      return normalizedRect;
    } else {
      normalizedRect.setXCenter(0.5);
      normalizedRect.setYCenter(0.5);
      normalizedRect.setWidth(1);
      normalizedRect.setHeight(1);
    }

    if (imageProcessingOptions?.rotationDegrees) {
      if (imageProcessingOptions?.rotationDegrees % 90 !== 0) {
        throw new Error(
            'Expected rotation to be a multiple of 90°.',
        );
      }

      // Convert to radians anti-clockwise.
      normalizedRect.setRotation(
          -Math.PI * imageProcessingOptions.rotationDegrees / 180.0);
    }

    return normalizedRect;
  }

  /** Runs the graph and blocks on the response. */
  private process(
      imageSource: ImageSource,
      imageProcessingOptions: ImageProcessingOptions|undefined,
      timestamp: number): void {
    const normalizedRect = this.convertToNormalizedRect(imageProcessingOptions);
    this.graphRunner.addProtoToStream(
        normalizedRect.serializeBinary(), 'mediapipe.NormalizedRect',
        this.normRectStreamName, timestamp);
    this.graphRunner.addGpuBufferAsImageToStream(
        imageSource, this.imageStreamName, timestamp ?? performance.now());
    this.finishProcessing();
  }
}


