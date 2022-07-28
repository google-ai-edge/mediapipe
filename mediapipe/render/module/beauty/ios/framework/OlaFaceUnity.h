#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#import <OpenGLES/EAGL.h>

typedef struct {
        int width;
        int height;
        int textureId;
        int ioSurfaceId; // iOS 专属
        int64_t frameTime;
} FaceTextureInfo;

@interface OlaFaceUnity : NSObject

@property (nonatomic) CGFloat whiten;
@property (nonatomic) CGFloat smooth;


+ (instancetype)sharedInstance;

- (EAGLContext *)currentContext;

- (void)resume;

- (void)suspend;

// 算法输入
- (void)processVideoFrame:(CVPixelBufferRef)pixelbuffer
                timeStamp:(int64_t)timeStamp;


- (FaceTextureInfo)render:(FaceTextureInfo)inputTexture;

- (void)dispose;

@end
