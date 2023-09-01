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

#import "mediapipe/tasks/ios/vision/image_segmenter/utils/sources/MPPImageSegmenterOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"

namespace {
using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using ImageSegmenterGraphOptionsProto =
    ::mediapipe::tasks::vision::image_segmenter::proto::ImageSegmenterGraphOptions;
using SegmenterOptionsProto = ::mediapipe::tasks::vision::image_segmenter::proto::SegmenterOptions;
}  // namespace

@implementation MPPImageSegmenterOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  ImageSegmenterGraphOptionsProto *imageSegmenterGraphOptionsProto =
      optionsProto->MutableExtension(ImageSegmenterGraphOptionsProto::ext);
  imageSegmenterGraphOptionsProto->Clear();

  [self.baseOptions copyToProto:imageSegmenterGraphOptionsProto->mutable_base_options()
              withUseStreamMode:self.runningMode != MPPRunningModeImage];
  imageSegmenterGraphOptionsProto->set_display_names_locale(self.displayNamesLocale.cppString);
}

@end
