// Copyright 2026 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/text/text_summarizer/sources/MPPTextSummarizer.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/text/text_summarizer/text_summarizer.h"

namespace {
using TextSummarizerCpp = ::mediapipe::tasks::text::text_summarizer::TextSummarizer;
using TextSummarizerOptionsCpp = ::mediapipe::tasks::text::text_summarizer::TextSummarizerOptions;
}  // namespace

@interface MPPTextSummarizer () {
  std::unique_ptr<TextSummarizerCpp> _textSummarizer;
}
@end

@implementation MPPTextSummarizer

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPTextSummarizerOptions *options = [[MPPTextSummarizerOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable instancetype)initWithOptions:(MPPTextSummarizerOptions *)options
                                   error:(NSError **)error {
  self = [super init];
  if (self) {
    auto cppOptions = std::make_unique<TextSummarizerOptionsCpp>();
    NSString *appId = [MPPCommonUtils appID];
    NSString *appVersion = [MPPCommonUtils appVersion];
    NSString *iosVersion = [MPPCommonUtils osVersion];
    cppOptions->base_options.host_environment =
        mediapipe::tasks::core::HostEnvironment::HOST_ENVIRONMENT_IOS;
    cppOptions->base_options.host_system = mediapipe::tasks::core::HostSystem::HOST_SYSTEM_IOS;
    cppOptions->base_options.host_version = iosVersion ? iosVersion.UTF8String : "";
    cppOptions->base_options.app_id = appId ? appId.UTF8String : "";
    cppOptions->base_options.app_version = appVersion ? appVersion.UTF8String : "";

    if (options.baseOptions.modelAssetPath) {
      cppOptions->base_options.model_asset_path = options.baseOptions.modelAssetPath.cppString;
    }

    if (options.baseOptions.delegate != MPPDelegateCPU) {
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeInvalidArgumentError
                description:@"INVALID_ARGUMENT: TextSummarizer::The delegate must be CPU."];
      return nil;
    }
    cppOptions->base_options.delegate = ::mediapipe::tasks::core::BaseOptions::Delegate::CPU;

    switch (options.mode) {
      case MPPTextSummarizerModeTLDR:
        cppOptions->mode = TextSummarizerOptionsCpp::Mode::TLDR;
        break;
      case MPPTextSummarizerModeKeyPoints:
        cppOptions->mode = TextSummarizerOptionsCpp::Mode::KEYPOINTS;
        break;
      default:
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"INVALID_ARGUMENT: TextSummarizer::The mode must be "
                                          @"either TLDR or KeyPoints."];
        return nil;
    }

    cppOptions->max_num_tokens = (int)options.maxNumTokens;
    if (options.cacheDir) {
      cppOptions->cache_dir = options.cacheDir.cppString;
    }

    auto textSummarizerOr = TextSummarizerCpp::Create(std::move(cppOptions));
    if (![MPPCommonUtils checkCppError:textSummarizerOr.status() toError:error]) {
      return nil;
    }
    _textSummarizer = std::move(*textSummarizerOr);
  }
  return self;
}

- (nullable MPPTextSummarizerResult *)summarizeText:(NSString *)text error:(NSError **)error {
  auto resultOr = _textSummarizer->Summarize(text.cppString);
  if (![MPPCommonUtils checkCppError:resultOr.status() toError:error]) {
    return nil;
  }
  NSString *summary = [NSString stringWithCString:resultOr->summary.c_str()
                                         encoding:NSUTF8StringEncoding];
  return [[MPPTextSummarizerResult alloc] initWithSummary:summary];
}

- (BOOL)summarizeStreamingWithText:(NSString *)text
                 completionHandler:(void (^)(MPPTextSummarizerStreamResult *_Nullable result,
                                             NSError *_Nullable error))completionHandler
                             error:(NSError **)error {
  auto cppCallback =
      [completionHandler](
          absl::StatusOr<mediapipe::tasks::text::text_summarizer::TextSummarizerStreamResult>
              streamResultOr) {
        NSError *callbackError = nil;
        if (![MPPCommonUtils checkCppError:streamResultOr.status() toError:&callbackError]) {
          completionHandler(nil, callbackError);
          return;
        }

        NSString *chunk = [NSString stringWithCString:streamResultOr->chunk.c_str()
                                             encoding:NSUTF8StringEncoding];
        MPPTextSummarizerStreamResult *result =
            [[MPPTextSummarizerStreamResult alloc] initWithChunk:chunk done:streamResultOr->done];
        completionHandler(result, nil);
      };

  absl::Status status = _textSummarizer->SummarizeStreaming(text.cppString, std::move(cppCallback));
  return [MPPCommonUtils checkCppError:status toError:error];
}

- (BOOL)closeWithError:(NSError **)error {
  absl::Status status = _textSummarizer->Close();
  return [MPPCommonUtils checkCppError:status toError:error];
}

@end
