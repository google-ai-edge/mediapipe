#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Converts images to SkBitmaps.
@interface SkImageConverter : NSObject

/// Converts a @c UIImage to a @c SkBitmap.
+ (void *)skBitmapFromCGImage:(CGImageRef)image;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
