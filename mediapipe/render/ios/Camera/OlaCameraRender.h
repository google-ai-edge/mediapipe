//
//  QuarameraCameraRender.h
//  QuarameraFramework
//
//  Created by wangrenzhu on 2021/1/25.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import "QuarameraShareTexture.h"

@interface OlaCameraRender : NSObject
@property (nonatomic, readonly) CGSize renderSize;
@property (nonatomic, readonly) BOOL enable;

- (instancetype)initWithRenderSize:(CGSize)renderSize
                            device:(id<MTLDevice>)device
                     cameraTexture:(QuarameraShareTexture *)cameraTexture
                    contentScaleFactor:(CGFloat)factor;

- (void)setupWithDevice:(id<MTLDevice>)device shareTexture:(QuarameraShareTexture *)shareTexture useRenderMode:(BOOL)useRenderMode;

/// 重置画布大小
/// @param renderSize 画布大小
- (void)resize:(CGSize)renderSize;

/// 处理相机采集流
/// @param sampleBuffer 相机采集流
- (void)processSampleBuffer:(CMSampleBufferRef)sampleBuffer;


/// 渲染
/// @param frameTime 帧时间
- (void)render:(NSTimeInterval)frameTime;

- (void)updateCameraTexture:(QuarameraShareTexture *)cameraTexture;


@end
