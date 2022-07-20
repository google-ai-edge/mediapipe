//
//  OlaShareTexture.m
//  OlaCameraFramework
//
//  Created by wangrenzhu on 2021/1/21.
//  Copyright © 2021 ola. All rights reserved.
//

#import "OlaShareTexture.h"
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#import <MetalKit/MetalKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#if !TARGET_OS_SIMULATOR
#import <OpenGLES/EAGLIOSurface.h>
#endif
 
#define GL_UNSIGNED_INT_8_8_8_8_REV 0x8367

__unused static OlaTextureFormatInfo formatTable[] =
{
    // Core Video Pixel Format,               Metal Pixel Format,            GL internalformat, GL format,   GL type
    { kCVPixelFormatType_32BGRA,              MTLPixelFormatBGRA8Unorm,      GL_RGBA,           GL_BGRA_EXT, GL_UNSIGNED_INT_8_8_8_8_REV },
};

static const float noRotationTextureCoordinates[] = {
    0.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
};

static const float standardImageVertices[] = {
    -1.0, 1.0,
    1.0, 1.0,
    -1.0, -1.0,
    1.0, -1.0
};

static const NSUInteger interopFormats = sizeof(formatTable) / sizeof(OlaTextureFormatInfo);

OlaTextureFormatInfo* textureFormatInfoFromMetalPixelFormat(MTLPixelFormat pixelFormat)
{
    for(int i = 0; i < interopFormats; i++) {
        if(pixelFormat == formatTable[i].mtlFormat) {
            return &formatTable[i];
        }
    }
    return NULL;
}

@interface OlaShareTexture() {
    
}

@property (nonatomic) id<MTLLibrary> library;
@property (nonatomic) id<MTLBuffer> outputVertextBuffer;
@property (nonatomic) id<MTLRenderPipelineState> renderPipeline;
@property (nonatomic) id<MTLBuffer> noRotationBuffer;

@end

@implementation OlaShareTexture
{
    OlaTextureFormatInfo *_formatInfo;
    CVPixelBufferRef _pixelBuffer;
    CVMetalTextureRef _mtlTexture;
    
    CVOpenGLESTextureRef _glTexture;
    CVOpenGLESTextureCacheRef _glTextureCache;
    CVMetalTextureCacheRef _mtlTextureCache;
    IOSurfaceID _ioSurfaceId;
    IOSurfaceRef renderIOSurface;
    CGSize _size;
}
@synthesize surfaceID = _ioSurfaceId;

- (void)dealloc
{
    if (_mtlTexture) {
        CFRelease(_mtlTexture);
    }
    
    if (_glTexture) {
        CFRelease(_glTexture);
    }
    
    if (_glTextureCache) {
        CFRelease(_glTextureCache);
    }
    
    if (_mtlTextureCache) {
        CFRelease(_mtlTextureCache);
    }
    
    if (_pixelBuffer) {
        CVPixelBufferRelease(_pixelBuffer);
        _pixelBuffer = nil;
    }
    
#if !TARGET_OS_SIMULATOR
    if (renderIOSurface) {
        CFRelease(renderIOSurface);
    }
#endif
    
    _formatInfo = nil;
    
    _openGLContext = nil;
    _metalDevice = nil;
    _outputVertextBuffer = nil;
}

- (nonnull instancetype)initWithMetalDevice:(id<MTLDevice>)mtlDevice
                              openGLContext:(EAGLContext *)glContext
                           metalPixelFormat:(MTLPixelFormat)mtlPixelFormat
                                sourceImage:(UIImage *)sourceImage
{
    self = [self initWithMetalDevice:mtlDevice openGLContext:glContext metalPixelFormat:mtlPixelFormat size:sourceImage.size];
    if (self) {
        [self renderUIImageToShareTexture:sourceImage];
    }
    return self;
}


- (nonnull instancetype)initWithMetalDevice:(nonnull id <MTLDevice>) metalevice
                              openGLContext:(nonnull EAGLContext *) glContext
                           metalPixelFormat:(MTLPixelFormat)mtlPixelFormat
                                       size:(CGSize)size
{
    self = [super init];
    if(self) {
        _formatInfo =
        textureFormatInfoFromMetalPixelFormat(mtlPixelFormat);
        
        NSAssert(_formatInfo, @"不支持这个格式");
        
        _size = size;
        _metalDevice = metalevice;
        _openGLContext = glContext;
        NSDictionary* cvBufferProperties = @{
            (__bridge NSString*)kCVPixelBufferIOSurfacePropertiesKey : @{},
            (__bridge NSString*)kCVPixelBufferOpenGLCompatibilityKey : @YES,
            (__bridge NSString*)kCVPixelBufferIOSurfaceOpenGLESTextureCompatibilityKey: @YES,
            (__bridge NSString*)kCVPixelBufferMetalCompatibilityKey : @YES,
        };
        
#if !TARGET_OS_SIMULATOR
        if (@available(iOS 11.0, *)) {
            int _width = size.width;
            int _height = size.height;
            unsigned bytesPerElement = 4;
            
            
            size_t bytesPerRow = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, _width * bytesPerElement);
            size_t totalBytes = IOSurfaceAlignProperty(kIOSurfaceAllocSize,  _height * bytesPerRow);
            NSDictionary *dict = @{
                (id)kIOSurfaceWidth : @(_width),
                (id)kIOSurfaceHeight : @(_height),
                (id)kIOSurfacePixelFormat : @(kCVPixelFormatType_32BGRA),
                (id)kIOSurfaceBytesPerElement : @(bytesPerElement),
                (id)kIOSurfaceBytesPerRow : @(bytesPerRow),
                (id)kIOSurfaceAllocSize : @(totalBytes),
                (id)kIOSurfaceIsGlobal: @YES
            };
            
            renderIOSurface = IOSurfaceCreate((CFDictionaryRef)dict);
            _ioSurfaceId = IOSurfaceGetID(renderIOSurface);
            CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, renderIOSurface,
                                             (__bridge CFDictionaryRef)cvBufferProperties, &_pixelBuffer);
            
            MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor
                                                       texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                       width:_width
                                                       height:_height
                                                       mipmapped:NO];
            textureDescriptor.storageMode = MTLStorageModeShared;
            textureDescriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            _metalTexture = [_metalDevice newTextureWithDescriptor:textureDescriptor
                                                         iosurface:renderIOSurface
                                                             plane:0];
            textureDescriptor = nil;
            [EAGLContext setCurrentContext:_openGLContext];
            
            glGenTextures(1, &_openGLTexture);
            glBindTexture(GL_TEXTURE_2D, _openGLTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                            GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            
            glBindTexture(GL_TEXTURE_2D, _openGLTexture);
#if !TARGET_OS_SIMULATOR
            BOOL rs = [_openGLContext texImageIOSurface:renderIOSurface
                                                 target:GL_TEXTURE_2D
                                          internalFormat:GL_RGBA
                                                   width:_width height:_height
                                                  format:GL_BGRA_EXT
                                                    type:GL_UNSIGNED_BYTE plane:0];
            NSAssert(rs, @"IOSurface binding 失败");
#endif
            glBindTexture(GL_TEXTURE_2D, 0);
            
        } else {
#endif
            
            __unused CVReturn cvret = CVPixelBufferCreate(kCFAllocatorDefault,
                                                          size.width,
                                                          size.height,
                                                          self.formatInfo->cvPixelFormat,
                                                          (__bridge CFDictionaryRef)cvBufferProperties,
                                                          &_pixelBuffer);
            NSAssert(cvret == kCVReturnSuccess, @"Failed to create CVPixelBuffer");
            
            [self createGLTexture];
            [self createMetalTexture];
#if !TARGET_OS_SIMULATOR
        }
#endif
    }

    
    return self;
    
}


- (void)createGLTexture
{
    __unused CVReturn cvret;
    cvret = CVOpenGLESTextureCacheCreate(kCFAllocatorDefault,
                                         nil,
                                         _openGLContext,
                                         nil,
                                         &_glTextureCache);
    
    
    cvret = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                         _glTextureCache,
                                                         _pixelBuffer,
                                                         nil,
                                                         GL_TEXTURE_2D,
                                                         _formatInfo->glInternalFormat,
                                                         _size.width, _size.height,
                                                         _formatInfo->glFormat,
                                                         _formatInfo->glType,
                                                         0,
                                                         &_glTexture);
    
    
    NSAssert(cvret == kCVReturnSuccess, @"OpenGL 纹理创建失败");
    
    _openGLTexture = CVOpenGLESTextureGetName(_glTexture);
}



- (void)createMetalTexture
{
    __unused CVReturn cvret;
    
    cvret = CVMetalTextureCacheCreate(
                                      kCFAllocatorDefault,
                                      nil,
                                      _metalDevice,
                                      nil,
                                      &_mtlTextureCache);
    
    cvret = CVMetalTextureCacheCreateTextureFromImage(
                                                      kCFAllocatorDefault,
                                                      _mtlTextureCache,
                                                      _pixelBuffer, nil,
                                                      _formatInfo->mtlFormat,
                                                      _size.width, _size.height,
                                                      0,
                                                      &_mtlTexture);
    
    NSAssert(cvret == kCVReturnSuccess, @"Metal 纹理创建失败");
    
    _metalTexture = CVMetalTextureGetTexture(_mtlTexture);
    
}

- (CVPixelBufferRef)renderTarget
{
    return _pixelBuffer;
}

- (OlaTextureFormatInfo *)formatInfo
{
    return (OlaTextureFormatInfo *)_formatInfo;
}

- (id<MTLTexture>)loadTextureFromImage:(UIImage *)image
{
    MTKTextureLoader *loader =[[MTKTextureLoader alloc] initWithDevice:self.metalDevice];
    NSError *error;
    id<MTLTexture> sourceTexture = [loader newTextureWithCGImage:image.CGImage options:@{
        MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead),
        MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModeShared),
    } error:&error];
    
    
    if (error) {
        return nil;
    } else {
        return sourceTexture;
    }
}

- (void)renderUIImageToShareTexture:(UIImage *)sourceImage
{
    id<MTLTexture> sourceTexture = [self loadTextureFromImage:sourceImage];
    if (sourceTexture) {
        
        NSError *error;
        NSBundle *bundle = [NSBundle mainBundle];
        NSURL *shaderURL = [bundle URLForResource:@"OlaFramework" withExtension:@"metallib"];
        if (@available(iOS 11.0, *)) {
            if (shaderURL) {
                self.library = [self.metalDevice newLibraryWithURL:shaderURL error:&error];
            }
        } else {
            NSString *lib = [[NSBundle mainBundle] pathForResource:@"OlaFramework" ofType:@"metallib"];
            if (lib) {
                _library = [_metalDevice newLibraryWithFile:lib error:nil];
            }
        }
        
        self.noRotationBuffer = [self.metalDevice newBufferWithBytes:noRotationTextureCoordinates
                                                              length:sizeof(noRotationTextureCoordinates)
                                                             options:MTLResourceStorageModeShared];
        
        self.outputVertextBuffer = [self.metalDevice newBufferWithBytes:standardImageVertices
                                                                 length:sizeof(standardImageVertices)
                                                                options:MTLResourceStorageModeShared];
        
        id<MTLFunction> vertexFunc = [self.library newFunctionWithName:@"oneInputVertex"];
        id<MTLFunction> fragFunc = [self.library newFunctionWithName:@"passthroughFragment"];
        
        id<MTLCommandQueue> commandQueue = [self.metalDevice newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        MTLRenderPipelineDescriptor *renderDesc = [MTLRenderPipelineDescriptor new];
        renderDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        renderDesc.rasterSampleCount = 1;
        renderDesc.vertexFunction = vertexFunc;
        renderDesc.fragmentFunction = fragFunc;
        
        self.renderPipeline = [self.metalDevice newRenderPipelineStateWithDescriptor:renderDesc error:&error];
        
        MTLRenderPassDescriptor *renderPass = [MTLRenderPassDescriptor renderPassDescriptor];
        
        renderPass.colorAttachments[0].texture = self.metalTexture;
        renderPass.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
        renderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
        
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPass];
        [renderEncoder setRenderPipelineState:self.renderPipeline];
        [renderEncoder setFragmentTexture:sourceTexture atIndex:0];
        [renderEncoder setVertexBuffer:self.outputVertextBuffer offset:0 atIndex:0];
        [renderEncoder setVertexBuffer:self.noRotationBuffer offset:0 atIndex:1];
        
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [renderEncoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        commandBuffer = nil;
        renderEncoder = nil;
        
    }
}

@end
