//
//  WANativeMTLRenderView.m
//  WebAR-iOS
//
//  Created by wangrenzhu on 2020/11/16.
//  Copyright © 2020 Taobao lnc. All rights reserved.
//

#import "QuarameraMTLCameraRenderView.h"
#import "QuarameraShareTexture.h"
#import "QuarameraMTLCameraRender.h"
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES3/gl.h>

static const NSUInteger MaxFramesInFlight = 3;
static size_t const kQuarameraDynamicTextureByteAlignment = 16;

NS_INLINE size_t QAAlignSize(size_t size)
{
    return ceil(size / (double)kQuarameraDynamicTextureByteAlignment) * kQuarameraDynamicTextureByteAlignment;
}

@interface QuarameraMTLCameraRenderView()
{
    
}

@property (nonatomic, strong) QuarameraMTLCameraRender *mtlRender;

@property (nonatomic) NSTimeInterval frameTime;
@property (nonatomic, strong) QuarameraShareTexture *shareTexture;
@property (nonatomic, strong) QuarameraShareTexture *cameraTexture;
@property (nonatomic) id<MTLTexture> ioSurfaceTexture;
@property (nonatomic) IOSurfaceID lastIOSurfaceID;
@property (nonatomic, strong) EAGLContext *openGLContext;
@property (nonatomic) dispatch_semaphore_t displayFrameRenderingSemaphore;

@property (nonatomic) dispatch_semaphore_t cameraFrameRenderingSemaphore;

@property (nonatomic, assign) BOOL useRenderMode;

@property (nonatomic, strong) NSMutableArray<QuarameraCameraRender *> *renders;
@property (nonatomic) CGSize lastFrameSize;
@end

@implementation QuarameraMTLCameraRenderView

- (void)dealloc
{
	_openGLContext = nil;
	_mtlRender = nil;
	
	_shareTexture = nil;
	_cameraTexture = nil;
  
}

- (instancetype)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame:frame];
    if (self) {
        _renders = [NSMutableArray arrayWithCapacity:10];
        _openGLContext = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3];
        [self initRender:frame];
        
    }
    return self;
}

- (instancetype)initWithFrame:(CGRect)frame shareContext:(EAGLContext *)context
{
    self = [super initWithFrame:frame];
    if (self) {
        _renders = [NSMutableArray arrayWithCapacity:10];
        _openGLContext = context;
        [self initRender:frame];
        
    }
    return self;
}


- (void)initRender:(const CGRect &)frame
{
    self.enableSetNeedsDisplay = NO;
    self.autoResizeDrawable = YES;
    self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    self.device = MTLCreateSystemDefaultDevice();
	self.lastFrameSize = frame.size;
    
    size_t alighWidth = QAAlignSize(frame.size.width * self.contentScaleFactor);
    
    CGSize textureSize = CGSizeMake(float(alighWidth),
                                    frame.size.height * self.contentScaleFactor);
    
    
    _shareTexture = [[QuarameraShareTexture alloc] initWithMetalDevice:self.device
                                                         openGLContext:self.openGLContext
                                                      metalPixelFormat:self.colorPixelFormat
                                                                  size:textureSize];
    
    _cameraTexture = [[QuarameraShareTexture alloc] initWithMetalDevice:self.device
                                                          openGLContext:self.openGLContext
                                                       metalPixelFormat:self.colorPixelFormat
                                                                   size:textureSize];
    
    _mtlRender = [[QuarameraMTLCameraRender alloc] initWithRenderSize:textureSize
                                                               device:self.device
                                                        cameraTexture:self.cameraTexture
                                                   contentScaleFactor:self.contentScaleFactor];
    
    __unused BOOL isSetCurrent = [EAGLContext setCurrentContext:self.openGLContext];
    NSAssert(isSetCurrent, @"上下文设置失败");
    
    dispatch_queue_attr_t interactive =
    dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,
                                            QOS_CLASS_USER_INTERACTIVE, 0);
    self.displayFrameRenderingSemaphore = dispatch_semaphore_create(MaxFramesInFlight);
    self.displayRenderQueue = dispatch_queue_create("quaramera.ios.displayRenderQueue",
                                                    interactive);

    self.cameraFrameRenderingSemaphore = dispatch_semaphore_create(1);
}

- (void)setNeedFlip:(BOOL)needFlip
{
    [self.mtlRender setNeedFlip:needFlip];
}

- (void)layoutSubviews
{
    [super layoutSubviews];
	if (!CGSizeEqualToSize(self.lastFrameSize, self.frame.size)) {
		CGSize newRenderSize = CGSizeMake(self.frame.size.width * self.contentScaleFactor,
										  self.frame.size.height * self.contentScaleFactor);
		self.lastFrameSize = self.frame.size;
		self.paused = YES;
		dispatch_async(self.displayRenderQueue, ^{
			
			size_t alighWidth = QAAlignSize(newRenderSize.width);
			
			CGSize textureSize = CGSizeMake(float(alighWidth),
											newRenderSize.height);

			[self.mtlRender resize:textureSize];
			
			[self.renders enumerateObjectsUsingBlock:^(QuarameraCameraRender * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
				[obj resize:textureSize];
			}];
			self.paused = NO;
		});
	}
}

#pragma mark
#pragma mark Private Method

- (void)willMoveToSuperview:(UIView *)newSuperview
{
    [self setPaused:NO];
}

- (void)resume
{
    [self setPaused:NO];
}

- (void)suspend
{
    [self setPaused:YES];
}

- (void)draw
{
    
#if !TARGET_OS_SIMULATOR
    
    id<CAMetalDrawable> drawable = [((CAMetalLayer *)self.layer) nextDrawable];
    
    __weak QuarameraMTLCameraRenderView *weakSelf = self;
    
//    dispatch_semaphore_t block_camera_sema = self.cameraFrameRenderingSemaphore;
    dispatch_semaphore_t block_display_sema = self.displayFrameRenderingSemaphore;
    
    void (^renderCompleted)(id<MTLCommandBuffer> buffer) = ^(id<MTLCommandBuffer> buffer)
    {
//        dispatch_semaphore_signal(block_camera_sema);
        dispatch_semaphore_signal(block_display_sema);
    };
    
    NSMutableArray<QuarameraCameraRender *> *renders = [self.renders copy];
    
    dispatch_async(self.displayRenderQueue, ^{
        if (weakSelf == nil) {
            return;
        }
        
        __strong QuarameraMTLCameraRenderView *strongSelf = weakSelf;
        
        strongSelf.frameTime += (1.0 / strongSelf.preferredFramesPerSecond) * 1000.0;
        if (dispatch_semaphore_wait(block_display_sema, DISPATCH_TIME_NOW) != 0)
        {
            return;
        }
        
        id<MTLCommandBuffer> commandBuffer = nil;
        
		commandBuffer = [strongSelf.mtlRender.commandQueue commandBuffer];
		[strongSelf.mtlRender renderToShareTexture:strongSelf.shareTexture.metalTexture
									 commandBuffer:commandBuffer
										 frameTime:strongSelf.frameTime];
		[commandBuffer commit];
        
        commandBuffer = [strongSelf.mtlRender.commandQueue commandBuffer];
        // 然后将渲染结果交给外部引擎 作为输入
        if (renders.count > 0) {
            //这个版本的Quarkit用的OpenGL渲染 所以需要glFlush
            //quarkitRendre把相机渲染到shareTexture上
           
            glFlush();
            [renders enumerateObjectsUsingBlock:^(QuarameraCameraRender * _Nonnull obj,
                                                  NSUInteger idx, BOOL * _Nonnull stop) {
				if (obj.enable) {
					[obj render:weakSelf.frameTime];
				}
            }];

            
        }
        
        if (strongSelf.cameraDelegate && !strongSelf.isPaused) {
            if (@available(iOS 11.0, *)) {
                glFlush();
                [strongSelf.cameraDelegate draw:strongSelf.frameTime];
                
                [strongSelf.cameraDelegate externalRender:strongSelf.frameTime
                                            targetTexture:strongSelf.shareTexture
                                            commandBuffer:commandBuffer];
                [EAGLContext setCurrentContext:self.openGLContext];
                IOSurfaceID surfaceId = [strongSelf.cameraDelegate bgraCameraTextureReady:strongSelf.cameraTexture
                                                  onScreenTexture:strongSelf.shareTexture
                                                      frameTime:strongSelf.frameTime * 1000];
                if (surfaceId != -1) {
                    //这里渲染surfaceId
                    IOSurfaceRef ioSurface = IOSurfaceLookup(surfaceId);
                    IOSurfaceLock(ioSurface, kIOSurfaceLockReadOnly, nil);
                    if (ioSurface) {
                        if (self.lastIOSurfaceID != surfaceId || self.ioSurfaceTexture == nil) {
                            id<MTLTexture> texture;
                            MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor
                                                                       texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                       width:strongSelf.cameraTexture.size.width
                                                                       height:strongSelf.cameraTexture.size.height
                                                                       mipmapped:NO];
                            textureDescriptor.storageMode = MTLStorageModeShared;
                            textureDescriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
                            texture = [self.device newTextureWithDescriptor:textureDescriptor iosurface:ioSurface plane:0];
                            self.ioSurfaceTexture = texture;
                            textureDescriptor = nil;
                        }
                      
                        IOSurfaceUnlock(ioSurface, kIOSurfaceLockReadOnly, nil);
                        CFRelease(ioSurface);
                        
                        self.lastIOSurfaceID = surfaceId;
                        if (self.ioSurfaceTexture) {
                            [strongSelf.mtlRender renderToTexture:drawable.texture
                                                             from:self.ioSurfaceTexture
                                                    commandBuffer:commandBuffer
                                                textureCoordinate:strongSelf.mtlRender.noRotationBuffer];
                            if (drawable) {
                                [commandBuffer presentDrawable:drawable];
                                [commandBuffer addCompletedHandler:renderCompleted];
                                [commandBuffer commit];
                            }
                            return;
                        }
                        
                    }
                }
                
            }
        }
        
        
        glFlush();
        //将混合渲染结果 渲染到屏幕
        [strongSelf.mtlRender renderToTexture:drawable.texture
                                         from:strongSelf.shareTexture.metalTexture
                                commandBuffer:commandBuffer
                            textureCoordinate:strongSelf.mtlRender.noRotationBuffer];
        if (drawable) {
            [commandBuffer presentDrawable:drawable];
            [commandBuffer addCompletedHandler:renderCompleted];
            [commandBuffer commit];
        }
    });
#endif
    
}

- (void)cameraSampleBufferArrive:(CMSampleBufferRef)sampleBuffer
{
    if (self.isPaused) {
        return;
    }
    
    if (dispatch_semaphore_wait(self.cameraFrameRenderingSemaphore, DISPATCH_TIME_NOW) != 0)
    {
        return;
    }
    
    dispatch_semaphore_t block_camera_sema = self.cameraFrameRenderingSemaphore;
    __strong QuarameraMTLCameraRenderView *weakSelf = self;
    void (^renderCompleted)(id<MTLCommandBuffer> buffer) = ^(id<MTLCommandBuffer> buffer)
    {
        dispatch_semaphore_signal(block_camera_sema);
    };
    
    CFRetain(sampleBuffer);
    dispatch_async(self.displayRenderQueue, ^{
        if (weakSelf == nil) {
            CFRelease(sampleBuffer);
            return;
        }
        __strong QuarameraMTLCameraRenderView *strongSelf = weakSelf;
        CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        [strongSelf.mtlRender renderToCameraTextureWithPixelBuffer:pixelBuffer completedHandler:renderCompleted];
        
        CFRelease(sampleBuffer);
    });
}

- (CVPixelBufferRef)renderTarget
{
    return self.cameraTexture.renderTarget;
}

- (void)addRender:(QuarameraCameraRender *)render
{
    NSAssert([NSThread isMainThread], @"call on main Thread");
    
    [render setupWithDevice:self.device shareTexture:self.shareTexture useRenderMode:self.useRenderMode];

    [self.renders addObject:render];
}
@end
