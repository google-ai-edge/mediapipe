// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <Foundation/Foundation.h>

#import "mediapipe/tasks/ios/components/containers/sources/MPPRegionOfInterest.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"

#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"

/**
 * This class helps create various kinds of packets for Mediapipe Vision Tasks.
 */
@interface MPPVisionPacketCreator : NSObject

/**
 * Creates a MediapPipe Packet wrapping an `MPPImage` that can be send to a graph.
 *
 * @param image The image to send to the MediaPipe graph.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 *    error will be saved.
 *
 * @return The MediaPipe packet containing the image. An empty packet is returned if an error
 *    occurred during the conversion.
 */
+ (mediapipe::Packet)createPacketWithMPPImage:(MPPImage *)image error:(NSError **)error;

/**
 * Creates a MediapPipe Packet wrapping an `MPPImage` that can be send to a graph at the specified
 * timestamp.
 *
 * @param image The image to send to the MediaPipe graph.
 * @param timestampInMilliseconds The timestamp (in milliseconds) to assign to the packet.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 *    error will be saved.
 *
 * @return The MediaPipe packet containing the image. An empty packet is returned if an error
 *    occurred during the conversion.
 */
+ (mediapipe::Packet)createPacketWithMPPImage:(MPPImage *)image
                      timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                        error:(NSError **)error;

/**
 * Creates a MediapPipe Packet wrapping a `NormalizedRect` that can be send to a graph.
 *
 * @param image The `NormalizedRect` to send to the MediaPipe graph.
 *
 * @return The MediaPipe packet containing the normalized rect.
 */
+ (mediapipe::Packet)createPacketWithNormalizedRect:(mediapipe::NormalizedRect &)normalizedRect;

/**
 * Creates a MediapPipe Packet wrapping a `NormalizedRect` that can be send to a graph at the
 * specified timestamp.
 *
 * @param image The `NormalizedRect` to send to the MediaPipe graph.
 * @param timestampInMilliseconds The timestamp (in milliseconds) to assign to the packet.
 *
 * @return The MediaPipe packet containing the normalized rect.
 */
+ (mediapipe::Packet)createPacketWithNormalizedRect:(mediapipe::NormalizedRect &)normalizedRect
                            timestampInMilliseconds:(NSInteger)timestampInMilliseconds;

/**
 * Creates a MediapPipe Packet wrapping a `RenderData` constructed from an `MPPRegionOfInterest`.
 *
 * @param regionOfInterest The `MPPRegionOfInterest` to send to the MediaPipe graph.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return The MediaPipe packet containing the `RenderData` constructed from the given
 * `MPPRegionOfInterest`.
 */
+ (std::optional<mediapipe::Packet>)createRenderDataPacketWithRegionOfInterest:
                                        (MPPRegionOfInterest *)regionOfInterest
                                                                         error:(NSError **)error;

@end
