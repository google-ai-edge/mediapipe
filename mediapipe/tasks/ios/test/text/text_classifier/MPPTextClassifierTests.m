/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/
#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/text/text_classifier/sources/MPPTextClassifier.h"

NS_ASSUME_NONNULL_BEGIN

static NSString *const kBertTextClassifierModelName = @"bert_text_classifier";
static NSString *const kNegativeText = @"unflinchingly bleak and desperate";

#define VerifyCategory(category, expectedIndex, expectedScore, expectedLabel, expectedDisplayName) \
  XCTAssertEqual(category.index, expectedIndex);                                                   \
  XCTAssertEqualWithAccuracy(category.score, expectedScore, 1e-6);                                 \
  XCTAssertEqualObjects(category.label, expectedLabel);                                            \
  XCTAssertEqualObjects(category.displayName, expectedDisplayName);

#define VerifyClassifications(classifications, expectedHeadIndex, expectedCategoryCount) \
  XCTAssertEqual(classifications.categories.count, expectedCategoryCount);               

#define VerifyClassificationResult(classificationResult, expectedClassificationsCount) \
  XCTAssertNotNil(classificationResult);                                               \
  XCTAssertEqual(classificationResult.classifications.count, expectedClassificationsCount)

#define AssertClassificationResultHasOneHead(classificationResult) \
  XCTAssertNotNil(classificationResult);                                               \
  XCTAssertEqual(classificationResult.classifications.count, 1);
  XCTAssertEqual(classificationResult.classifications[0].headIndex, 1);

#define AssertTextClassifierResultIsNotNil(textClassifierResult) \
  XCTAssertNotNil(textClassifierResult);                                          

@interface MPPTextClassifierTests : XCTestCase
@end

@implementation MPPTextClassifierTests

- (void)setUp {
  [super setUp];

}

- (NSString *)filePathWithName:(NSString *)fileName extension:(NSString *)extension {
  NSString *filePath = [[NSBundle bundleForClass:self.class] pathForResource:fileName
                                                                      ofType:extension];
  XCTAssertNotNil(filePath);

  return filePath;
}

- (MPPTextClassifierOptions *)textClassifierOptionsWithModelName:(NSString *)modelName {
  NSString *modelPath = [self filePathWithName:modelName extension:@"tflite"];
  MPPTextClassifierOptions *textClassifierOptions =
      [[MPPTextClassifierOptions alloc] init];
  textClassifierOptions.baseOptions.modelAssetPath = modelPath;

  return textClassifierOptions;
}

kBertTextClassifierModelName

- (MPPTextClassifier *)createTextClassifierFromOptionsWithModelName:(NSString *)modelName {
  MPPTextClassifierOptions *options = [self textClassifierOptionsWithModelName:modelName];
  MPPTextClassifier *textClassifier = [[MPPTextClassifier alloc] initWithOptions:options error:nil];
  XCTAssertNotNil(textClassifier);

  return textClassifier
}

- (void)classifyWithBertSucceeds {
  MPPTextClassifier *textClassifier = [self createTextClassifierWithModelName:kBertTextClassifierModelName];
  MPPTextClassifierResult *textClassifierResult = [textClassifier classifyWithText:kNegativeText];
}

@end

NS_ASSUME_NONNULL_END
