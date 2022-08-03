//
//  WANativeMTLCameraPreviewView.h
//  
//  Created by wangrenzhu on 2020/11/16.
//  Copyright © 2020 Taobao lnc. All rights reserved.
//

#import <MetalKit/MetalKit.h>
#import <AVFoundation/AVFoundation.h>
#import "OlaCameraRender.h"
#import "OlaMTLCameraRender.h"
#import "OlaShareTexture.h"

@protocol OlaMTLCameraRenderViewDelegate

- (void)draw:(NSTimeInterval)frameTime;


/// 开放离屏相机纹理 用于外部分流
/// @param texture texture description
/// @param onScreenTexture 上屏纹理
/// @param frameTime 帧时间
- (void)bgraCameraTextureReady:(OlaShareTexture *)texture
               onScreenTexture:(OlaShareTexture *)onScreenTexture
                     frameTime:(NSTimeInterval)frameTime;

@optional

/// 渲染到TargetTexture上可以上屏，如果是GL则需要调用GLFlush来同步
/// @param frameTime frameTime description
/// @param targetTexture targetTexture description
/// @param buffer MTL的CommandBuffer
- (IOSurfaceID)externalRender:(NSTimeInterval)frameTime
         targetTexture:(OlaShareTexture *)targetTexture
         commandBuffer:(id<MTLCommandBuffer>)buffer;


/// YUV 相机纹理
/// @param yTexture y纹理
/// @param uvTexture yv纹理
- (void)yuvTextureReady:(OlaShareTexture *)yTexture uvTexture:(OlaShareTexture *)uvTexture;

@end

@interface OlaMTLCameraRenderView : MTKView

/// MetalRender
@property (nonatomic, strong, readonly) OlaMTLCameraRender *mtlRender;

@property (nonatomic, weak) id<OlaMTLCameraRenderViewDelegate> cameraDelegate;

@property (nonatomic) dispatch_queue_t displayRenderQueue;
@property (nonatomic) dispatch_queue_t offlineRenderQueue;

/// 原始相机纹理 可以快速读取
@property (nonatomic, readonly, strong) OlaShareTexture *cameraTexture;
@property (nonatomic, readonly, strong) OlaShareTexture *halfCameraTexture;

/// 不带后处理的相机渲染的原始纹理
@property (nonatomic, readonly) CVPixelBufferRef renderTarget;

/// 是否镜像渲染 默认为NO
@property (nonatomic) BOOL needFlip;
  
/// 恢复/开始渲染
- (void)resume;

/// 暂停/挂起渲染
- (void)suspend;

/// 处理相机采集流
/// @param sampleBuffer 相机采集流
- (void)cameraSampleBufferArrive:(CMSampleBufferRef)sampleBuffer;

- (void)renderPixelbuffer:(CVPixelBufferRef)pixelbuffer;

- (void)addRender:(OlaCameraRender *)render;


/// 是否开启Ola
/// @param frame frame description
- (instancetype)initWithFrame:(CGRect)frame;

- (instancetype)initWithFrame:(CGRect)frame shareContext:(EAGLContext *)context;

@end
