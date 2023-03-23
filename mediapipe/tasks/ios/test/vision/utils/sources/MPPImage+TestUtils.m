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

@interface UIImage (FileUtils)

+ (nullable UIImage *)imageFromBundleWithClass:(Class)classObject
                                      fileName:(NSString *)name
                                        ofType:(NSString *)type;

@end

@implementation UIImage (FileUtils)

+ (nullable UIImage *)imageFromBundleWithClass:(Class)classObject
                                      fileName:(NSString *)name
                                        ofType:(NSString *)type {
  NSString *imagePath = [[NSBundle bundleForClass:classObject] pathForResource:name ofType:type];
  if (!imagePath) return nil;

  return [[UIImage alloc] initWithContentsOfFile:imagePath];
}

@end

@implementation MPPImage (TestUtils)

+ (nullable MPPImage *)imageFromBundleWithClass:(Class)classObject
                                       fileName:(NSString *)name
                                         ofType:(NSString *)type {
  UIImage *image = [UIImage imageFromBundleWithClass:classObject fileName:name ofType:type];

  return [[MPPImage alloc] initWithUIImage:image error:nil];
}

+ (nullable MPPImage *)imageFromBundleWithClass:(Class)classObject
                                       fileName:(NSString *)name
                                         ofType:(NSString *)type
                                    orientation:(UIImageOrientation)imageOrientation {
  UIImage *image = [UIImage imageFromBundleWithClass:classObject fileName:name ofType:type];

  return [[MPPImage alloc] initWithUIImage:image orientation:imageOrientation error:nil];
}

@end
