//
//  OlaCameraRender.m
//  OlaFramework
//
//  Created by wangrenzhu on 2021/1/25.
//

#import "OlaCameraRender.h"

@implementation OlaCameraRender
@synthesize renderSize = _renderSize;

- (instancetype)initWithRenderSize:(CGSize)renderSize
                            device:(id<MTLDevice>)device
                     cameraTexture:(OlaShareTexture *)cameraTexture
                    contentScaleFactor:(CGFloat)factor
{
    NSAssert(NO, @"subclass must implement this method");
    return nil;
}


- (void)setupWithDevice:(id<MTLDevice>)device shareTexture:(OlaShareTexture *)shareTexture useRenderMode:(BOOL)useRenderMode;
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

- (void)updateCameraTexture:(OlaShareTexture *)cameraTexture
{
	
}

- (BOOL)enable
{
    return NO;
}

@end
