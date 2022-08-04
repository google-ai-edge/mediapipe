

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>
#import "OlaShareTexture.h"
#import "OlaCameraRender.h"

@protocol OlaMTLCameraRenderDelegate <NSObject>

/// 闪电拍照准备完毕
- (void)lightningModelPrepared;

/// 首帧渲染完毕
- (void)firstFrameRendered;

@end



@interface OlaMTLCameraRender : OlaCameraRender

/// 输出CameraTexture
@property (nonatomic) id<MTLTexture> outputTexture;
@property (nonatomic) BOOL lightningMode;
@property (nonatomic) BOOL useScan;
@property (nonatomic) BOOL needFlip;
@property (nonatomic) id<MTLBuffer> noRotationBuffer;
@property (nonatomic) CGFloat contentScaleFactor;
@property (nonatomic) id<MTLCommandQueue> commandQueue;

///  renderTarget 可以快速拿到相机的渲染结果
@property (nonatomic, readonly) CVPixelBufferRef renderTarget;


@property (nonatomic, weak) id<OlaMTLCameraRenderDelegate> renderDelegate;
@property (nonatomic, strong) OlaShareTexture *offscreenCameraTexture;

/// 渲染到纹理指令
/// @param displayTexture displayTexture description
/// @param sourceTexture sourceTexture description
/// @param commandBuffer commandBuffer description
/// @param coordinateBuffer coordinateBuffer description
- (void)renderToTexture:(id<MTLTexture>)displayTexture
                   from:(id<MTLTexture>)sourceTexture
          commandBuffer:(id<MTLCommandBuffer>)commandBuffer
      textureCoordinate:(id<MTLBuffer>)coordinateBuffer;

- (void)renderToShareTexture:(id<MTLTexture>)shareTexture commandBuffer:(id<MTLCommandBuffer>)commandBuffer frameTime:(NSTimeInterval)frameTime;

/// 通过Metal处理相机数据
/// @param pixelBuffer pixelBuffer description
/// @param completedHandler completedHandler description
- (void)renderToCameraTextureWithPixelBuffer:(CVPixelBufferRef)pixelBuffer completedHandler:(MTLCommandBufferHandler)completedHandler;

@end
