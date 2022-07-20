//
//  OlaMTLRender.m
//  
//
//  Created by wangrenzhu on 2021/1/22.
//  Copyright © 2021 ola. All rights reserved.
//
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>
#import "OlaMTLCameraRender.h"


typedef struct
{
    int useScan;
    float iRadius;
    float squareWidth;
    float width;
    float height;
    float iTime;
} ScanUniform;

static const float noRotationTextureCoordinates[] = {
    0.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
};

static const float rotate90TextureCoordinates[] = {
    0.0f, 1.0f,
    0.0f, 0.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
};

static const float standardImageVertices[] = {
    -1.0, 1.0,
    1.0, 1.0,
    -1.0, -1.0,
    1.0, -1.0
};

struct TextureScale {
    simd_float3 scaleFlip;
};

@interface OlaMTLCameraRender()

@property (nonatomic) id<MTLRenderPipelineState> colorConvertPipelineState;

@property (nonatomic, strong) MTLRenderPipelineDescriptor *colorConvertRenderPipelineDesc;
@property (nonatomic) id<MTLTexture> cameraTexture;

@property (nonatomic) id<MTLRenderPipelineState> transformRenderPipeline;

@property (nonatomic) id<MTLTexture> luminanceTextureTexture;
@property (nonatomic) id<MTLTexture> chrominanceTextureTexure;

@property (nonatomic, strong) MTLRenderPipelineDescriptor *renderPipelineDesc;

@property (nonatomic) CVMetalTextureCacheRef mtlTextureCacheRef;

@property (nonatomic) MTLSize threadGroups;
@property (nonatomic) MTLSize threadsPerGroups;

@property (nonatomic) id<MTLLibrary> library;
@property (nonatomic) id<MTLDevice> device;
@property (nonatomic) id<MTLBuffer> textureCoordinateBuffer;
@property (nonatomic) id<MTLBuffer> outputVertextBuffer;
@property (nonatomic) id<MTLBuffer> textureScaleBuffer;

@property (nonatomic) float aspect;
@property (nonatomic) TextureScale textureScale;
@property (nonatomic) BOOL isInit;
@property (nonatomic) NSTimeInterval iTime;
@property (nonatomic) BOOL firstFrameRender;

@property (nonatomic) size_t cameraOutputWidth;
@property (nonatomic) size_t cameraOutputHeight;

@property (nonatomic, strong) OlaShareTexture *shareTexture;


@end

@implementation OlaMTLCameraRender
@synthesize renderSize = _renderSize;

- (void)dealloc
{
    self.renderDelegate = nil;
    _luminanceTextureTexture = nil;
    _chrominanceTextureTexure = nil;
    _renderPipelineDesc = nil;
    _cameraTexture = nil;
    _colorConvertPipelineState = nil;
    
    _transformRenderPipeline = nil;
    _commandQueue = nil;
    
    _textureCoordinateBuffer = nil;
    _outputVertextBuffer = nil;
    if (_mtlTextureCacheRef) {
        CFRelease(_mtlTextureCacheRef);
    }

    _offscreenCameraTexture = nil;
    _shareTexture = nil;

}

- (instancetype)initWithRenderSize:(CGSize)renderSize
                            device:(id<MTLDevice>)device
                     cameraTexture:(OlaShareTexture *)cameraTexture
                    contentScaleFactor:(CGFloat)factor
{
    self = [super init];
    if (self) {
        _contentScaleFactor = factor;
        [self resize:renderSize];
        self.device = device;
        __unused NSError *error;
        _offscreenCameraTexture = cameraTexture;
        NSBundle *bundle = [NSBundle bundleForClass:[OlaMTLCameraRender class]];
        
        
        NSURL *shaderURL = [bundle URLForResource:@"OlaCameraMetalLibrary" withExtension:@"metallib"];
        if (@available(iOS 11.0, *)) {
            if (shaderURL) {
                self.library = [self.device newLibraryWithURL:shaderURL error:&error];
            }
        } else {
            NSString *lib = [[NSBundle mainBundle] pathForResource:@"OlaFramework" ofType:@"metallib"];
            if (lib) {
                _library = [_device newLibraryWithFile:lib error:nil];
            }
        }
		NSDictionary * cacheAttributes = @{ (NSString *)kCVMetalTextureCacheMaximumTextureAgeKey: @(0.0) };
        NSAssert(error == nil, @"newDefaultLibraryWithBundle %@", error);
        CVMetalTextureCacheCreate(NULL, NULL, self.device,
								  (__bridge CFDictionaryRef _Nullable)(cacheAttributes),
								  &_mtlTextureCacheRef);
        
        _aspect = self.renderSize.height / self.renderSize.width;
        
        self.cameraOutputWidth = self.renderSize.width;
        self.cameraOutputHeight = self.renderSize.height;
        
        [self initPipeline];
    }
    return self;
}

#pragma mark
#pragma mark Private

- (void)initPipeline
{
    if (self.renderTarget == nil || self.offscreenCameraTexture == nil) {
        return;
    }
    
    self.commandQueue = [self.device newCommandQueue];
    __unused NSError *error;
    NSAssert(error == nil, @"newDefaultLibraryWithBundle %@", error);
    
    id<MTLFunction> colorVertexFunc = [self.library newFunctionWithName:@"twoInputVertex"];
    id<MTLFunction> colorFragFunc = [self.library newFunctionWithName:@"yuvConversionFullRangeFragment"];
    MTLRenderPipelineDescriptor *colorConvertRenderDesc = [MTLRenderPipelineDescriptor new];
    colorConvertRenderDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    colorConvertRenderDesc.rasterSampleCount = 1;
    colorConvertRenderDesc.vertexFunction = colorVertexFunc;
    colorConvertRenderDesc.fragmentFunction = colorFragFunc;
    self.colorConvertRenderPipelineDesc = colorConvertRenderDesc;
    
    self.colorConvertPipelineState = [self.device
                                      newRenderPipelineStateWithDescriptor:self.colorConvertRenderPipelineDesc
                                      error:&error];
    
    NSAssert(error == nil, @"colorConvertPipelineState newRenderPipelineStateWithDescriptor %@", error);
    
    id<MTLFunction> vertexFunc = [self.library newFunctionWithName:@"oneInputVertex"];
    id<MTLFunction> fragFunc = [self.library newFunctionWithName:@"passthroughFragment"];
    
    
    MTLRenderPipelineDescriptor *renderDesc = [MTLRenderPipelineDescriptor new];
    renderDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    renderDesc.rasterSampleCount = 1;
    renderDesc.vertexFunction = vertexFunc;
    renderDesc.fragmentFunction = fragFunc;
    
    self.renderPipelineDesc = renderDesc;
    
    self.transformRenderPipeline = [self.device newRenderPipelineStateWithDescriptor:self.renderPipelineDesc error:&error];
    NSAssert(error == nil, @"transformRenderPipeline newRenderPipelineStateWithDescriptor %@", error);
    
    
    self.textureCoordinateBuffer = [self.device newBufferWithBytes:rotate90TextureCoordinates
                                                       length:sizeof(rotate90TextureCoordinates)
                                                      options:MTLResourceStorageModeShared];
    
    self.noRotationBuffer = [self.device newBufferWithBytes:noRotationTextureCoordinates
                                                length:sizeof(noRotationTextureCoordinates)
                                               options:MTLResourceStorageModeShared];
    self.outputVertextBuffer = [self.device newBufferWithBytes:standardImageVertices
                                                   length:sizeof(standardImageVertices)
                                                  options:MTLResourceStorageModeShared];

    self.isInit = YES;
}

#pragma mark
#pragma mark - Offscreen Render Camera

- (void)renderToCameraTextureWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                            completedHandler:(MTLCommandBufferHandler)completedHandler
{
    [self ensureTexture:pixelBuffer completedHandler:completedHandler];
}

- (void)ensureTexture:(CVPixelBufferRef)cameraBuffer completedHandler:(MTLCommandBufferHandler)completedHandler
{
    size_t outputWidth = CVPixelBufferGetWidth(cameraBuffer);
    size_t outputHeight = CVPixelBufferGetHeight(cameraBuffer);
    
    if (self.renderTarget == nil || self.transformRenderPipeline == nil) {
        return; //renderTarget没创建不渲染
    }
    
    if (@available(iOS 11.0, *)) {
    
        MTLTextureDescriptor *luminanceTextureDesc = [MTLTextureDescriptor
                                                   texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                   width:outputWidth
                                                   height:outputHeight
                                                   mipmapped:NO];
        luminanceTextureDesc.storageMode = MTLStorageModeShared;
        luminanceTextureDesc.usage = MTLTextureUsageShaderRead;

        MTLTextureDescriptor *chrominanceTextureDesc = [MTLTextureDescriptor
                                                   texture2DDescriptorWithPixelFormat:MTLPixelFormatRG8Unorm
                                                   width:outputWidth / 2
                                                   height:outputHeight / 2
                                                   mipmapped:NO];
        chrominanceTextureDesc.storageMode = MTLStorageModeShared;
        chrominanceTextureDesc.usage = MTLTextureUsageShaderRead;
        
        self.luminanceTextureTexture = [self.device newTextureWithDescriptor:luminanceTextureDesc
                                                                   iosurface:CVPixelBufferGetIOSurface(cameraBuffer)
                                                                       plane:0];
        
        self.chrominanceTextureTexure = [self.device newTextureWithDescriptor:chrominanceTextureDesc
                                                                  iosurface:CVPixelBufferGetIOSurface(cameraBuffer)
                                                                      plane:1];
        luminanceTextureDesc = nil;
        chrominanceTextureDesc = nil;
    } else {
    
        CVMetalTextureRef luminanceTextureRef;
        CVMetalTextureRef chrominanceTextureRef;
        
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                  self.mtlTextureCacheRef,
                                                  cameraBuffer,
                                                  NULL,
                                                  MTLPixelFormatR8Unorm, outputWidth, outputHeight, 0,
                                                  &luminanceTextureRef);
        
        
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                  self.mtlTextureCacheRef,
                                                  cameraBuffer,
                                                  NULL,
                                                  MTLPixelFormatRG8Unorm, outputWidth / 2,
                                                  outputHeight / 2, 1, &chrominanceTextureRef);
        
        self.luminanceTextureTexture = CVMetalTextureGetTexture(luminanceTextureRef);
        CFRelease(luminanceTextureRef);
        self.chrominanceTextureTexure = CVMetalTextureGetTexture(chrominanceTextureRef);
        CFRelease(chrominanceTextureRef);
    }
    
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    commandBuffer.label = @"color conversion buffer";
    
    MTLRenderPassDescriptor *colorRenderPass = [MTLRenderPassDescriptor renderPassDescriptor];
    
    colorRenderPass.colorAttachments[0].texture = self.offscreenCameraTexture.metalTexture;
    colorRenderPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    colorRenderPass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    colorRenderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    
    id<MTLRenderCommandEncoder> colorConversionEncoder = [commandBuffer renderCommandEncoderWithDescriptor:colorRenderPass];
    colorConversionEncoder.label = @"Color Encoder";
    float textureAspectRatio = (float)outputWidth / (float)outputHeight;
    float drawAspectRatio = self.renderSize.height / self.renderSize.width;
    
    _textureScale.scaleFlip.x = 1.0;
    _textureScale.scaleFlip.y = 1.0;
    if (drawAspectRatio == textureAspectRatio) {
        _textureScale.scaleFlip.x = 1.0;
        _textureScale.scaleFlip.y = 1.0;
    } else if (self.renderSize.height > self.renderSize.width) {
        //竖屏
        float realHeight = outputHeight > outputWidth ? outputHeight : outputWidth;
        float realWidth = outputHeight < outputWidth ? outputHeight : outputWidth;
        if (drawAspectRatio > textureAspectRatio) {
            //太宽了
            float newWidth = (float)realHeight / drawAspectRatio;
            _textureScale.scaleFlip.y = newWidth / (float)realWidth;
            _textureScale.scaleFlip.x = 1.0;
        } else {
            //太高了
            float newHeight = (float)realWidth * drawAspectRatio;
            _textureScale.scaleFlip.x = newHeight / (float)realHeight;
            _textureScale.scaleFlip.y = 1.0;
        }
    } else {
        //横屏
        float realHeight = outputHeight < outputWidth ? outputHeight : outputWidth;
        float realWidth = outputHeight > outputWidth ? outputHeight : outputWidth;
        if (drawAspectRatio > textureAspectRatio) {
            //太宽了
            float newWidth = (float)realHeight / drawAspectRatio;
            _textureScale.scaleFlip.y = newWidth / (float)realWidth;
            _textureScale.scaleFlip.x = 1.0;
        } else {
            //太高了
            float newHeight = (float)realWidth * drawAspectRatio;
            _textureScale.scaleFlip.x = (float)newHeight / realHeight;
            _textureScale.scaleFlip.y = 1.0;
        }
    }
    
    _textureScale.scaleFlip.z = self.needFlip ? 1.0 : 0.0;
    
    self.textureScaleBuffer = [self.device newBufferWithBytes:&_textureScale length:sizeof(_textureScale) options:MTLResourceStorageModeShared];
    
    [colorConversionEncoder setRenderPipelineState:self.colorConvertPipelineState];
    [colorConversionEncoder setFragmentTexture:self.luminanceTextureTexture atIndex:0];
    [colorConversionEncoder setFragmentTexture:self.chrominanceTextureTexure atIndex:1];
    [colorConversionEncoder setFragmentBuffer:self.textureScaleBuffer offset:0 atIndex:0];
    [colorConversionEncoder setVertexBuffer:self.outputVertextBuffer offset:0 atIndex:0];
    [colorConversionEncoder setVertexBuffer:self.textureCoordinateBuffer offset:0 atIndex:1];
    [colorConversionEncoder setVertexBuffer:self.textureCoordinateBuffer offset:0 atIndex:2];
    
    [colorConversionEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [colorConversionEncoder endEncoding];
    
    __weak OlaMTLCameraRender *weakSelf = self;

    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> commandBuffer) {
        if (weakSelf == nil) {
            return;
        }
        __strong OlaMTLCameraRender *strongSelf = weakSelf;
        if (!strongSelf.firstFrameRender) {
            NSLog(@"相机首帧渲染完毕");
            //这里埋点时机和Android统一，收到相机帧时发送，但实际上还是渲染完后发送比较合适
            if (strongSelf.renderDelegate) {
                [strongSelf.renderDelegate firstFrameRendered];
            }
            strongSelf.firstFrameRender = YES;
        }
        
        if (completedHandler) {
            completedHandler(commandBuffer);
        }
    }];
    
    [commandBuffer commit];
    
    if (self.lightningMode) {
        self.lightningMode = NO;
        if (self.renderDelegate) {
            [self.renderDelegate lightningModelPrepared];
        }
    }
    
}

#pragma mark
#pragma mark - offscreen Render
- (void)renderToShareTexture:(id<MTLTexture>)shareTexture
               commandBuffer:(id<MTLCommandBuffer>)commandBuffer
                   frameTime:(NSTimeInterval)frameTime
{
    if (!self.isInit) {
        return;
    }
    self.iTime = frameTime;
    if (self.offscreenCameraTexture.metalTexture && self.transformRenderPipeline) {
        commandBuffer.label = @"Camera Command Buffer";
        
        id<MTLTexture> sourceTexture = self.offscreenCameraTexture.metalTexture;
        
        [self renderToTexture:shareTexture from:sourceTexture
                commandBuffer:commandBuffer
            textureCoordinate:self.noRotationBuffer];
        
        return;
    }
}

- (void)renderToTexture:(id<MTLTexture>)displayTexture
                   from:(id<MTLTexture>)sourceTexture
          commandBuffer:(id<MTLCommandBuffer>)commandBuffer
      textureCoordinate:(id<MTLBuffer>)coordinateBuffer
{
    if (displayTexture == nil || sourceTexture == nil) {
        return;
    }
    MTLRenderPassDescriptor *renderPass = [MTLRenderPassDescriptor renderPassDescriptor];
    
    renderPass.colorAttachments[0].texture = displayTexture;
    renderPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    renderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPass];
    [renderEncoder setRenderPipelineState:self.transformRenderPipeline];
    [renderEncoder setFragmentTexture:sourceTexture atIndex:0];
    [renderEncoder setVertexBuffer:self.outputVertextBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:coordinateBuffer offset:0 atIndex:1];

    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [renderEncoder endEncoding];
    renderEncoder = nil;
    renderPass = nil;
}

- (void)render:(NSTimeInterval)frameTime
{
    
}

#pragma mark
#pragma mark - Properties

- (void)resize:(CGSize)renderSize
{
    _renderSize = renderSize;
	_aspect = self.renderSize.height / self.renderSize.width;
	
	self.cameraOutputWidth = self.renderSize.width;
	self.cameraOutputHeight = self.renderSize.height;
}

- (void)updateCameraTexture:(OlaShareTexture *)cameraTexture
{
	_offscreenCameraTexture = nil;
	_offscreenCameraTexture = cameraTexture;
}

- (CVPixelBufferRef)renderTarget
{
    return self.offscreenCameraTexture.renderTarget;
}
@end
