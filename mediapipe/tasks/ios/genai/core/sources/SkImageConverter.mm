#import "mediapipe/tasks/ios/genai/core/sources/SkImageConverter.h"

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

#include "include/core/SkBitmap.h"  // from @skia
#include "include/utils/mac/SkCGUtils.h"  // from @skia

@implementation SkImageConverter

+ (void *)skBitmapFromCGImage:(CGImageRef)image {
  auto bitmap = std::make_unique<SkBitmap>();
  SkCreateBitmapFromCGImage(bitmap.get(), image);
  bitmap->setImmutable();
  return bitmap.release();
}

@end
