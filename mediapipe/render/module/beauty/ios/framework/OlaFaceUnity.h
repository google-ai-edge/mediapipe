#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#import <OpenGLES/EAGL.h>
#import "OlaFURenderView.h"

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
@property (nonatomic) CGFloat slim;
@property (nonatomic) CGFloat nose;
@property (nonatomic) CGFloat eyeFactor;
@property (nonatomic, weak) OlaFURenderView *renderView;
@property (nonatomic) BOOL useGLRender; //测试用开关

- (void)initModule;

+ (instancetype)sharedInstance;

- (EAGLContext *)currentContext;

- (void *)currentGLContext;

- (void)resume;

- (void)suspend;

// 算法输入
- (void)processVideoFrame:(CVPixelBufferRef)pixelbuffer
                timeStamp:(int64_t)timeStamp;

// 相机采集输入 直接渲染到renderView上 rotatedRightFlipVertical YUV420
- (void)renderSampleBuffer:(CMSampleBufferRef)samplebuffer;

// 离屏渲染到目标texture上
- (FaceTextureInfo)render:(FaceTextureInfo)inputTexture;

- (void)dispose;

@end
