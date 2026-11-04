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

#import "mediapipe/tasks/c/core/base_options.h"
#import "mediapipe/tasks/c/text/text_summarizer/text_summarizer.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

namespace {

struct SummarizerStreamContext {
  void (^completionHandler)(MPPTextSummarizerStreamResult *_Nullable result,
                            NSError *_Nullable error);
};

void SummarizerStreamingCallback(void *userData,
                                 const struct MpTextSummarizerStreamResult *cStreamResult,
                                 const char *errorMsg) {
  @autoreleasepool {
    auto *context = static_cast<SummarizerStreamContext *>(userData);
    if (errorMsg) {
      NSError *error = [NSError
          errorWithDomain:MPPTasksErrorDomain
                     code:MPPTasksErrorCodeInternalError
                 userInfo:@{NSLocalizedDescriptionKey : [NSString stringWithUTF8String:errorMsg]}];
      context->completionHandler(nil, error);
      delete context;
      return;
    }
    if (cStreamResult) {
      NSString *chunk =
          cStreamResult->chunk ? [NSString stringWithUTF8String:cStreamResult->chunk] : @"";
      MPPTextSummarizerStreamResult *result =
          [[MPPTextSummarizerStreamResult alloc] initWithChunk:chunk done:cStreamResult->done];
      bool isDone = cStreamResult->done;
      context->completionHandler(result, nil);
      if (isDone) {
        delete context;
      }
    }
  }
}

}  // namespace

@interface MPPTextSummarizer () {
  MpTextSummarizerPtr _summarizer;
}
@end

@implementation MPPTextSummarizer

- (void)dealloc {
  if (_summarizer) {
    MpTextSummarizerClose(_summarizer, nullptr);
    _summarizer = nullptr;
  }
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPTextSummarizerOptions *options = [[MPPTextSummarizerOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable instancetype)initWithOptions:(MPPTextSummarizerOptions *)options
                                   error:(NSError **)error {
  self = [super init];
  if (self) {
    if (options.baseOptions.delegate != MPPDelegateCPU) {
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeInvalidArgumentError
                description:@"INVALID_ARGUMENT: TextSummarizer::The delegate must be CPU."];
      return nil;
    }

    MpTextSummarizerMode mode;
    switch (options.mode) {
      case MPPTextSummarizerModeTLDR:
        mode = MP_TEXT_SUMMARIZER_MODE_TLDR;
        break;
      case MPPTextSummarizerModeKeyPoints:
        mode = MP_TEXT_SUMMARIZER_MODE_KEYPOINTS;
        break;
      default:
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"INVALID_ARGUMENT: TextSummarizer::The mode must be "
                                          @"either TLDR or KeyPoints."];
        return nil;
    }

    MpTextSummarizerOptions cOptions = {};
    [options.baseOptions copyToCOptions:&cOptions.base_options];
    cOptions.mode = mode;
    cOptions.max_num_tokens = (int)options.maxNumTokens;
    cOptions.cache_dir =
        options.cacheDir ? const_cast<char *>(options.cacheDir.UTF8String) : nullptr;
    cOptions.min_log_severity = 4;

    char *cError = nullptr;
    MpStatus status = MpTextSummarizerCreate(&cOptions, &_summarizer, &cError);
    [MPPBaseOptions freeCOptions:&cOptions.base_options];
    if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
      return nil;
    }
  }
  return self;
}

- (nullable MPPTextSummarizerResult *)summarizeText:(NSString *)text error:(NSError **)error {
  MpTextSummarizerResult cResult = {};
  char *cError = nullptr;
  MpStatus status =
      MpTextSummarizerSummarize(_summarizer, text ? text.UTF8String : "", &cResult, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }
  NSString *summary = cResult.summary ? [NSString stringWithUTF8String:cResult.summary] : @"";
  MpTextSummarizerCloseResult(&cResult);
  return [[MPPTextSummarizerResult alloc] initWithSummary:summary];
}

- (BOOL)summarizeStreamingWithText:(NSString *)text
                 completionHandler:(void (^)(MPPTextSummarizerStreamResult *_Nullable result,
                                             NSError *_Nullable error))completionHandler
                             error:(NSError **)error {
  auto *context = new SummarizerStreamContext{[completionHandler copy]};
  char *cError = nullptr;
  MpStatus status = MpTextSummarizerSummarizeStreaming(
      _summarizer, text ? text.UTF8String : "", SummarizerStreamingCallback, context, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    delete context;
    return NO;
  }
  return YES;
}

- (BOOL)closeWithError:(NSError **)error {
  if (!_summarizer) {
    return YES;
  }
  char *cError = nullptr;
  MpStatus status = MpTextSummarizerClose(_summarizer, &cError);
  _summarizer = nullptr;
  return [MPPCommonUtils checkCError:status errorMessage:cError toError:error];
}

@end
