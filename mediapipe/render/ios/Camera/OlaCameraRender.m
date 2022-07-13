//
//  QuarameraCameraRender.m
//  QuarameraFramework
//
//  Created by wangrenzhu on 2021/1/25.
//

#import "QuarameraCameraRender.h"

@implementation QuarameraCameraRender
@synthesize renderSize = _renderSize;

- (instancetype)initWithRenderSize:(CGSize)renderSize
                            device:(id<MTLDevice>)device
                     cameraTexture:(QuarameraShareTexture *)cameraTexture
                    contentScaleFactor:(CGFloat)factor
{
    NSAssert(NO, @"subclass must implement this method");
    return nil;
}


- (void)setupWithDevice:(id<MTLDevice>)device shareTexture:(QuarameraShareTexture *)shareTexture useRenderMode:(BOOL)useRenderMode;
{
    
}

- (void)processSampleBuffer:(CMSampleBufferRef)sampleBuffer
{
    
}

- (void)resize:(CGSize)renderSize
{
    
}

- (void)render:(NSTimeInterval)frameTime
{
    
}

- (void)updateCameraTexture:(QuarameraShareTexture *)cameraTexture
{
	
}

- (BOOL)enable
{
    return NO;
}

@end
