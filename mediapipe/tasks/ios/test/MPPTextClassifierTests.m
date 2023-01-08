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

static NSString *const kBertTextClassifierModelName = @"bert_text_classifier";
static NSString *const kNegativeText = @"unflinchingly bleak and desperate";
static NSString *const kPositiveText = @"it's a charming and often affecting journey";

#define AssertCategoriesAre(categories, expectedCategories) \
  XCTAssertEqual(categories.count, expectedCategories.count); \
  for (int i = 0; i < categories.count; i++) { \
    XCTAssertEqual(categories[i].index, expectedCategories[i].index);                                                   \
    XCTAssertEqualWithAccuracy(categories[i].score, expectedCategories[i].score, 1e-6);                                 \
    XCTAssertEqualObjects(categories[i].categoryName, expectedCategories[i].categoryName);                                            \
    XCTAssertEqualObjects(categories[i].displayName, expectedCategories[i].displayName); \
  }              

#define AssertHasOneHead(textClassifierResult) \
  XCTAssertNotNil(textClassifierResult);                           \              
  XCTAssertNotNil(textClassifierResult.classificationResult);      \
  XCTAssertEqual(textClassifierResult.classificationResult.classifications.count, 1);   \
  XCTAssertEqual(textClassifierResult.classificationResult.classifications[0].headIndex, 0);   

@interface MPPTextClassifierTests : XCTestCase
@end

@implementation MPPTextClassifierTests

- (void)setUp {
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
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

- (MPPTextClassifier *)createTextClassifierFromOptionsWithModelName:(NSString *)modelName {
  MPPTextClassifierOptions *options = [self textClassifierOptionsWithModelName:modelName];
  MPPTextClassifier *textClassifier = [[MPPTextClassifier alloc] initWithOptions:options error:nil];
  XCTAssertNotNil(textClassifier);

  return textClassifier;
}

- (void)testClassifyWithBertSucceeds {
  MPPTextClassifier *textClassifier = [self createTextClassifierFromOptionsWithModelName:kBertTextClassifierModelName];
   
  MPPTextClassifierResult *negativeResult = [textClassifier classifyWithText:kNegativeText error:nil];
  AssertHasOneHead(negativeResult);
  
  NSArray<MPPCategory *> *expectedNegativeCategories = @[[[MPPCategory alloc] initWithIndex:0
                                 score:0.956187f
                                 categoryName:@"negative"
                           displayName:nil],
    [[MPPCategory alloc] initWithIndex:1
                                 score:0.043812f
                                 categoryName:@"positive"
                           displayName:nil]];
  
  AssertCategoriesAre(negativeResult.classificationResult.classifications[0].categories,
                      expectedNegativeCategories
  );

  // MPPTextClassifierResult *positiveResult = [textClassifier classifyWithText:kPositiveText error:nil];
  // AssertHasOneHead(positiveResult);
  // NSArray<MPPCategory *> *expectedPositiveCategories = @[[[MPPCategory alloc] initWithIndex:0
  //                                score:0.99997187f
  //                                label:@"positive"
  //                          displayName:nil],
  //   [[MPPCategory alloc] initWithIndex:1
  //                                score:2.8132641E-5f
  //                                label:@"negative"
  //                          displayName:nil]];
  // AssertCategoriesAre(negativeResult.classificationResult.classifications[0].categories,
  //                     expectedPositiveCategories
  // );

}
@end
