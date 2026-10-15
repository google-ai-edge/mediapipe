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

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/text/text_proofreader/text_proofreader.h"

namespace {
using TextProofreaderCpp = ::mediapipe::tasks::text::text_proofreader::TextProofreader;
using TextProofreaderOptionsCpp =
    ::mediapipe::tasks::text::text_proofreader::TextProofreaderOptions;
using CorrectionCpp = ::mediapipe::tasks::text::text_proofreader::Correction;
using CorrectionTypeCpp = ::mediapipe::tasks::text::text_proofreader::CorrectionType;

NSArray<MPPCorrection *> *ConvertCorrections(const std::vector<CorrectionCpp> &cppCorrections) {
  NSMutableArray<MPPCorrection *> *corrections =
      [NSMutableArray arrayWithCapacity:cppCorrections.size()];
  for (const auto &cppCorrection : cppCorrections) {
    MPPCorrectionType type = MPPCorrectionTypeSame;
    switch (cppCorrection.type) {
      case CorrectionTypeCpp::kSame:
        type = MPPCorrectionTypeSame;
        break;
      case CorrectionTypeCpp::kInsertion:
        type = MPPCorrectionTypeInsertion;
        break;
      case CorrectionTypeCpp::kDeletion:
        type = MPPCorrectionTypeDeletion;
        break;
    }
    NSString *text = [NSString stringWithCppString:cppCorrection.text];
    [corrections addObject:[[MPPCorrection alloc] initWithType:type text:text]];
  }
  return corrections;
}

}  // namespace

@interface MPPTextProofreader () {
  std::unique_ptr<TextProofreaderCpp> _textProofreader;
}
@end

@implementation MPPTextProofreader

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPTextProofreaderOptions *options = [[MPPTextProofreaderOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable instancetype)initWithOptions:(MPPTextProofreaderOptions *)options
                                   error:(NSError **)error {
  self = [super init];
  if (self) {
    auto cppOptions = std::make_unique<TextProofreaderOptionsCpp>();
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
                description:@"INVALID_ARGUMENT: TextProofreader::The delegate must be CPU."];
      return nil;
    }
    cppOptions->base_options.delegate = ::mediapipe::tasks::core::BaseOptions::Delegate::CPU;

    cppOptions->max_num_tokens = static_cast<int>(options.maxNumTokens);
    if (options.cacheDir) {
      cppOptions->cache_dir = options.cacheDir.cppString;
    }

    auto textProofreaderOr = TextProofreaderCpp::Create(std::move(cppOptions));
    if (![MPPCommonUtils checkCppError:textProofreaderOr.status() toError:error]) {
      return nil;
    }
    _textProofreader = std::move(*textProofreaderOr);
  }
  return self;
}

- (nullable MPPTextProofreaderResult *)proofreadText:(NSString *)text error:(NSError **)error {
  auto resultOr = _textProofreader->Proofread(text.cppString);
  if (![MPPCommonUtils checkCppError:resultOr.status() toError:error]) {
    return nil;
  }
  NSString *proofreadText = [NSString stringWithCppString:resultOr->proofread_text];
  NSArray<MPPCorrection *> *corrections = ConvertCorrections(resultOr->corrections);
  return [[MPPTextProofreaderResult alloc] initWithProofreadText:proofreadText
                                                     corrections:corrections];
}

- (BOOL)proofreadStreamingWithText:(NSString *)text
                 completionHandler:(void (^)(MPPTextProofreaderStreamResult *_Nullable result,
                                             NSError *_Nullable error))completionHandler
                             error:(NSError **)error {
  auto cppCallback =
      [completionHandler](
          absl::StatusOr<mediapipe::tasks::text::text_proofreader::TextProofreaderStreamResult>
              streamResultOr) {
        NSError *callbackError = nil;
        if (![MPPCommonUtils checkCppError:streamResultOr.status() toError:&callbackError]) {
          completionHandler(nil, callbackError);
          return;
        }

        NSString *chunk = [NSString stringWithCppString:streamResultOr->chunk];
        NSArray<MPPCorrection *> *corrections = nil;
        if (streamResultOr->corrections.has_value()) {
          corrections = ConvertCorrections(streamResultOr->corrections.value());
        }

        MPPTextProofreaderStreamResult *result =
            [[MPPTextProofreaderStreamResult alloc] initWithChunk:chunk
                                                      corrections:corrections
                                                             done:streamResultOr->done];
        completionHandler(result, nil);
      };

  absl::Status status =
      _textProofreader->ProofreadStreaming(text.cppString, std::move(cppCallback));
  return [MPPCommonUtils checkCppError:status toError:error];
}

- (BOOL)closeWithError:(NSError **)error {
  absl::Status status = _textProofreader->Close();
  return [MPPCommonUtils checkCppError:status toError:error];
}

@end
