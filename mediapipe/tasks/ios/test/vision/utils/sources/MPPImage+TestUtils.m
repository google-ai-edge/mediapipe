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

#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"

@implementation MPPImage (TestUtils)

+ (nullable MPPImage *)imageFromBundleWithClass:(Class)classObject
                              fileName:(NSString *)name
                                ofType:(NSString *)type
                                error:(NSError **)error {
  NSString *imagePath = [[NSBundle bundleForClass:classObject] pathForResource:name ofType:type];
  if (!imagePath) return nil;

  UIImage *image = [[UIImage alloc] initWithContentsOfFile:imagePath];
  if (!image) return nil;

  return [[MPPImage alloc] initWithUIImage:image error:error];
}

+ (nullable MPPImage *)imageFromBundleWithClass:(Class)classObject
                              fileName:(NSString *)name
                                ofType:(NSString *)type
                                orientation:(UIImageOrientation)imageOrientation
                                error:(NSError **)error {
  NSString *imagePath = [[NSBundle bundleForClass:classObject] pathForResource:name ofType:type];
  if (!imagePath) return nil;

  UIImage *image = [[UIImage alloc] initWithContentsOfFile:imagePath];
  if (!image) return nil;

  return [[MPPImage alloc] initWithUIImage:image orientation:imageOrientation error:error];
}

@end
