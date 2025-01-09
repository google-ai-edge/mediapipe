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

#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPRegionOfInterest+Helpers.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"

#include <cstdint>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/timestamp.h"

static const NSUInteger kMicrosecondsPerMillisecond = 1000;

namespace {
using ::mediapipe::RenderData;
using ::mediapipe::Image;
using ::mediapipe::ImageFrame;
using ::mediapipe::MakePacket;
using ::mediapipe::NormalizedRect;
using ::mediapipe::Packet;
using ::mediapipe::Timestamp;
}  // namespace

@implementation MPPVisionPacketCreator

+ (Packet)createPacketWithMPPImage:(MPPImage *)image error:(NSError **)error {
  std::unique_ptr<ImageFrame> imageFrame = [image imageFrameWithError:error];

  if (!imageFrame) {
    return Packet();
  }

  return MakePacket<Image>(std::move(imageFrame));
}

+ (Packet)createPacketWithMPPImage:(MPPImage *)image
           timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                             error:(NSError **)error {
  std::unique_ptr<ImageFrame> imageFrame = [image imageFrameWithError:error];

  if (!imageFrame) {
    return Packet();
  }

  return MakePacket<Image>(std::move(imageFrame))
      .At(Timestamp(int64_t(timestampInMilliseconds * kMicrosecondsPerMillisecond)));
}

+ (Packet)createPacketWithNormalizedRect:(NormalizedRect &)normalizedRect {
  return MakePacket<NormalizedRect>(std::move(normalizedRect));
}

+ (Packet)createPacketWithNormalizedRect:(NormalizedRect &)normalizedRect
                 timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  return MakePacket<NormalizedRect>(std::move(normalizedRect))
      .At(Timestamp(int64_t(timestampInMilliseconds * kMicrosecondsPerMillisecond)));
}

+ (std::optional<Packet>)createRenderDataPacketWithRegionOfInterest:
                             (MPPRegionOfInterest *)regionOfInterest
                                                              error:(NSError **)error {
  std::optional<RenderData> renderData = [regionOfInterest getRenderDataWithError:error];

  if (!renderData.has_value()) {
    return std::nullopt;
  }

  return MakePacket<RenderData>(std::move(renderData.value()));
}

@end
