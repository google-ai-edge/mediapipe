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

#import "mediapipe/tasks/ios/text/text_proofreader/sources/MPPTextProofreader.h"

#import "mediapipe/tasks/c/core/base_options.h"
#import "mediapipe/tasks/c/text/text_proofreader/text_proofreader.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

namespace {

NSString *NSStringFromUTF8(const char *utf8String) {
  if (!utf8String) {
    return @"";
  }
  return [NSString stringWithUTF8String:utf8String] ?: @"";
}

NSArray<MPPCorrection *> *ConvertCorrections(const MpCorrection *corrections, int count) {
  if (!corrections || count <= 0) {
    return @[];
  }
  NSMutableArray<MPPCorrection *> *result = [NSMutableArray arrayWithCapacity:count];
  for (int i = 0; i < count; ++i) {
    MPPCorrectionType type = MPPCorrectionTypeSame;
    switch (corrections[i].type) {
      case MP_CORRECTION_TYPE_SAME:
        type = MPPCorrectionTypeSame;
        break;
      case MP_CORRECTION_TYPE_INSERTION:
        type = MPPCorrectionTypeInsertion;
        break;
      case MP_CORRECTION_TYPE_DELETION:
        type = MPPCorrectionTypeDeletion;
        break;
    }
    NSString *text = NSStringFromUTF8(corrections[i].text);
    [result addObject:[[MPPCorrection alloc] initWithType:type text:text]];
  }
  return result;
}

struct ProofreaderStreamContext {
  void (^completionHandler)(MPPTextProofreaderStreamResult *_Nullable result,
                            NSError *_Nullable error);
  dispatch_queue_t callbackQueue;
};

void ProofreaderStreamingCallback(void *userData,
                                  const struct MpTextProofreaderStreamResult *cStreamResult,
                                  const char *errorMsg) {
  @autoreleasepool {
    auto *context = static_cast<ProofreaderStreamContext *>(userData);
    dispatch_queue_t callbackQueue = context->callbackQueue;
    if (errorMsg) {
      NSError *error = nil;
      [MPPCommonUtils createCustomError:&error
                               withCode:MPPTasksErrorCodeInternalError
                            description:NSStringFromUTF8(errorMsg)];
      dispatch_async(callbackQueue, ^{
        context->completionHandler(nil, error);
        delete context;
      });
      return;
    }
    if (cStreamResult) {
      NSString *chunk = NSStringFromUTF8(cStreamResult->chunk);
      bool isDone = cStreamResult->done;
      NSArray<MPPCorrection *> *corrections =
          isDone
              ? ConvertCorrections(cStreamResult->corrections, cStreamResult->corrections_count)
              : nil;
      MPPTextProofreaderStreamResult *result =
          [[MPPTextProofreaderStreamResult alloc] initWithChunk:chunk
                                                    corrections:corrections
                                                           done:isDone];
      dispatch_async(callbackQueue, ^{
        context->completionHandler(result, nil);
        if (isDone) {
          delete context;
        }
      });
    }
  }
}

}  // namespace

@interface MPPTextProofreader () {
  MpTextProofreaderPtr _proofreader;
  dispatch_queue_t _callbackQueue;
}
@end

@implementation MPPTextProofreader

- (void)dealloc {
  if (_proofreader) {
    MpTextProofreaderClose(_proofreader, nullptr);
    _proofreader = nullptr;
  }
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPTextProofreaderOptions *options = [[MPPTextProofreaderOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable instancetype)initWithOptions:(MPPTextProofreaderOptions *)options
                                   error:(NSError **)error {
  self = [super init];
  if (self) {
    if (options.baseOptions.delegate != MPPDelegateCPU) {
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeInvalidArgumentError
                description:@"INVALID_ARGUMENT: TextProofreader::The delegate must be CPU."];
      return nil;
    }

    MpTextProofreaderOptions cOptions = {};
    [options.baseOptions copyToCOptions:&cOptions.base_options];
    cOptions.max_num_tokens = static_cast<int>(options.maxNumTokens);
    cOptions.cache_dir = options.cacheDir ? options.cacheDir.UTF8String : nullptr;
    cOptions.min_log_severity = 4;

    char *cError = nullptr;
    MpStatus status = MpTextProofreaderCreate(&cOptions, &_proofreader, &cError);
    [MPPBaseOptions freeCOptions:&cOptions.base_options];
    if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
      return nil;
    }
    _callbackQueue = dispatch_queue_create(
        "com.google.mediapipe.tasks.textProofreader.callbackQueue", DISPATCH_QUEUE_SERIAL);
  }
  return self;
}

- (nullable MPPTextProofreaderResult *)proofreadText:(NSString *)text error:(NSError **)error {
  MpTextProofreaderResult *cResult = nullptr;
  char *cError = nullptr;
  MpStatus status =
      MpTextProofreaderProofread(_proofreader, text ? text.UTF8String : "", &cResult, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    return nil;
  }
  NSString *proofreadText = NSStringFromUTF8(cResult->proofread_text);
  NSArray<MPPCorrection *> *corrections =
      ConvertCorrections(cResult->corrections, cResult->corrections_count);
  MPPTextProofreaderResult *result =
      [[MPPTextProofreaderResult alloc] initWithProofreadText:proofreadText
                                                  corrections:corrections];
  MpTextProofreaderCloseResult(cResult);
  return result;
}

- (BOOL)proofreadStreamingWithText:(NSString *)text
                 completionHandler:(void (^)(MPPTextProofreaderStreamResult *_Nullable result,
                                             NSError *_Nullable error))completionHandler
                             error:(NSError **)error {
  if (!completionHandler) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"INVALID_ARGUMENT: completionHandler cannot be nil."];
    return NO;
  }
  auto *context = new ProofreaderStreamContext{[completionHandler copy], _callbackQueue};
  char *cError = nullptr;
  MpStatus status = MpTextProofreaderProofreadStreaming(
      _proofreader, text ? text.UTF8String : "", ProofreaderStreamingCallback, context, &cError);
  if (![MPPCommonUtils checkCError:status errorMessage:cError toError:error]) {
    delete context;
    return NO;
  }
  return YES;
}

- (BOOL)closeWithError:(NSError **)error {
  if (!_proofreader) {
    return YES;
  }
  char *cError = nullptr;
  MpStatus status = MpTextProofreaderClose(_proofreader, &cError);
  _proofreader = nullptr;
  return [MPPCommonUtils checkCError:status errorMessage:cError toError:error];
}

@end
