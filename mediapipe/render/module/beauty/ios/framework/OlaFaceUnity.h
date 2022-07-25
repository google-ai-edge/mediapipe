#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

@interface OlaFaceUnity : NSObject


+ (instancetype)sharedInstance;


- (void)processVideoFrame:(CVPixelBufferRef)pixelbuffer
                timeStamp:(int64_t)timeStamp;

- (void)dispose;

@end
