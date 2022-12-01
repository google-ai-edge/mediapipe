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
      [[MPPTextClassifierOptions alloc] initWithModelPath:modelPath];

  return textClassifierOptions;
}

- (void)testCreateTextClassifierOptionsSucceeds {
  MPPTextClassifierOptions *options = [self textClassifierOptionsWithModelName:kBertTextClassifierModelName];
  MPPTextClassifier *textClassifier = [[MPPTextClassifier alloc] initWithOptions:options error:nil];
  XCTAssertNotNil(textClassifier);
}

@end

NS_ASSUME_NONNULL_END
