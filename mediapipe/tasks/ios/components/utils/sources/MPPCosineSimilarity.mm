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

#import "mediapipe/tasks/ios/components/utils/sources/MPPCosineSimilarity.h"

#include <math.h>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

@implementation MPPCosineSimilarity

+ (nullable NSNumber *)computeBetweenVector1:(NSArray<NSNumber *> *)u
                                  andVector2:(NSArray<NSNumber *> *)v
                                     isFloat:(BOOL)isFloat
                                       error:(NSError **)error {
  if (u.count != v.count) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"Cannot compute cosine similarity between "
                                                     @"embeddings of different sizes (%lu vs %lu",
                                                     static_cast<u_long>(u.count),
                                                     static_cast<u_long>(v.count)]];
    return nil;
  }

  __block double dotProduct = 0.0;
  __block double normU = 0.0;
  __block double normV = 0.0;

  [u enumerateObjectsUsingBlock:^(NSNumber *num, NSUInteger idx, BOOL *stop) {
    double uVal = 0.0;
    double vVal = 0.0;

    if (isFloat) {
      uVal = num.floatValue;
      vVal = v[idx].floatValue;
    } else {
      uVal = num.charValue;
      vVal = v[idx].charValue;
    }

    dotProduct += uVal * vVal;
    normU += uVal * uVal;
    normV += vVal * vVal;
  }];

  return [NSNumber numberWithDouble:dotProduct / sqrt(normU * normV)];
}

+ (nullable NSNumber *)computeBetweenEmbedding1:(MPPEmbedding *)embedding1
                                  andEmbedding2:(MPPEmbedding *)embedding2
                                          error:(NSError **)error {
  if (embedding1.floatEmbedding && embedding2.floatEmbedding) {
    return [MPPCosineSimilarity computeBetweenVector1:embedding1.floatEmbedding
                                           andVector2:embedding2.floatEmbedding
                                              isFloat:YES
                                                error:error];
  }

  if (embedding1.quantizedEmbedding && embedding2.quantizedEmbedding) {
    return [MPPCosineSimilarity computeBetweenVector1:embedding1.quantizedEmbedding
                                           andVector2:embedding2.quantizedEmbedding
                                              isFloat:NO
                                                error:error];
  }

  [MPPCommonUtils
      createCustomError:error
               withCode:MPPTasksErrorCodeInvalidArgumentError
            description:
                @"Cannot compute cosine similarity between quantized and float embeddings."];
  return nil;
}

@end
