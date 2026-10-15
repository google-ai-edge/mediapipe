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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * The type of correction.
 */
typedef NS_ENUM(NSInteger, MPPCorrectionType) {
  /** The text is the same in the original and proofread text. */
  MPPCorrectionTypeSame = 0,
  /** The text was inserted in the proofread text. */
  MPPCorrectionTypeInsertion,
  /** The text was deleted from the original text. */
  MPPCorrectionTypeDeletion,
} NS_SWIFT_NAME(CorrectionType);

/**
 * A single correction.
 */
NS_SWIFT_NAME(Correction)
@interface MPPCorrection : NSObject

/**
 * The type of correction.
 */
@property(nonatomic, readonly) MPPCorrectionType type;

/**
 * The text associated with the correction.
 */
@property(nonatomic, readonly) NSString *text;

/**
 * Initializes a new `MPPCorrection` with the given type and text.
 *
 * @param type The type of correction.
 * @param text The text associated with the correction.
 *
 * @return An initialized `MPPCorrection`.
 */
- (instancetype)initWithType:(MPPCorrectionType)type text:(NSString *)text;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

/**
 * @brief The result of a proofread operation.
 */
NS_SWIFT_NAME(TextProofreaderResult)
@interface MPPTextProofreaderResult : NSObject

/**
 * The generated proofread text.
 */
@property(nonatomic, readonly) NSString *proofreadText;

/**
 * The list of corrections to transform the input text to the proofread text.
 */
@property(nonatomic, readonly) NSArray<MPPCorrection *> *corrections;

/**
 * Initializes a new `MPPTextProofreaderResult` with the given text and corrections.
 *
 * @param proofreadText The generated proofread text.
 * @param corrections The list of corrections.
 *
 * @return An initialized `MPPTextProofreaderResult`.
 */
- (instancetype)initWithProofreadText:(NSString *)proofreadText
                          corrections:(NSArray<MPPCorrection *> *)corrections;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

/**
 * @brief The result of a streaming proofread operation.
 */
NS_SWIFT_NAME(TextProofreaderStreamResult)
@interface MPPTextProofreaderStreamResult : NSObject

/**
 * The generated proofread text chunk.
 */
@property(nonatomic, readonly) NSString *chunk;

/**
 * The list of corrections to transform the input text to the proofread text.
 * Only populated when `done` is YES.
 */
@property(nonatomic, readonly, nullable) NSArray<MPPCorrection *> *corrections;

/**
 * Whether the proofreading is done.
 */
@property(nonatomic, readonly) BOOL done;

/**
 * Initializes a new `MPPTextProofreaderStreamResult` with the given chunk and done state.
 *
 * @param chunk The generated proofread text chunk.
 * @param corrections The list of corrections.
 * @param done Whether the proofreading is done.
 *
 * @return An initialized `MPPTextProofreaderStreamResult`.
 */
- (instancetype)initWithChunk:(NSString *)chunk
                  corrections:(nullable NSArray<MPPCorrection *> *)corrections
                         done:(BOOL)done;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
