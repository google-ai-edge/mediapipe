#import "OlaFaceUnity.h"
#import "OlaFURenderView+private.h"
#include "mediapipe/render/module/beauty/face_mesh_module.h"
#include "mediapipe/render/core/OlaCameraSource.hpp"
#include "mediapipe/render/core/Context.hpp"
#include "mediapipe/render/core/Filter.hpp"

using namespace Opipe;
@interface OlaFaceUnity() {
    Opipe::FaceMeshModule *_face_module;
    OlaCameraSource *sourceCamera;
    dispatch_queue_t videoQueue;
}

@property (nonatomic) dispatch_semaphore_t cameraFrameRenderingSemaphore;

@end
@implementation OlaFaceUnity

- (void)dealloc
{
    if (sourceCamera) {
        sourceCamera->release();
        sourceCamera = nullptr;
    }
    
    if (_face_module) {
        delete _face_module;
        _face_module = nullptr;
    }
}

- (instancetype)init
{
    self = [super init];
    if (self) {
       
    }
    return self;
}

- (void)initModule
{
    _face_module = Opipe::FaceMeshModule::create();
    NSBundle *bundle = [NSBundle bundleForClass:[self class]];
    NSURL* graphURL = [bundle URLForResource:@"face_mesh_mobile_landmark_gpu" withExtension:@"binarypb"];
    NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:nil];
    if (data) {
        _face_module->init(nullptr, (void *)data.bytes, data.length);
        _face_module->startModule();
    }
    if (_useGLRender) {
        _face_module->runInContextSync([&] {
            OlaContext *context = _face_module->currentContext();

            Context *glContext = context->glContext();

            sourceCamera = new OlaCameraSource(glContext, Opipe::SourceCamera::SourceType_YUV420SP);

            _face_module->setInputSource(sourceCamera);

        });
        self.cameraFrameRenderingSemaphore = dispatch_semaphore_create(1);
        videoQueue = dispatch_queue_create("FaceUnity.videoQueue", 0);
    }
    
}

+ (instancetype)sharedInstance
{
    static OlaFaceUnity *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (FaceTextureInfo)render:(FaceTextureInfo)inputTexture
{
    TextureInfo rs;
    rs.ioSurfaceId = inputTexture.ioSurfaceId;
    rs.width = inputTexture.width;
    rs.height = inputTexture.height;
    rs.textureId = inputTexture.textureId;
    rs.frameTime = inputTexture.frameTime;
    if (_face_module) {
        TextureInfo input;
        input.width = inputTexture.width;
        input.height = inputTexture.height;
        input.ioSurfaceId = inputTexture.ioSurfaceId;
        input.textureId = inputTexture.textureId;
        input.frameTime = inputTexture.frameTime;
        
        rs = _face_module->renderTexture(input);
    }
    FaceTextureInfo result;
    result.width = rs.width;
    result.height = rs.height;
    result.ioSurfaceId = rs.ioSurfaceId;
    result.textureId = rs.textureId;
    result.frameTime = rs.frameTime;
    return result;
}

- (void)renderSampleBuffer:(CMSampleBufferRef)samplebuffer
{
    if (!self.cameraFrameRenderingSemaphore) {
        return;
    }
    if (dispatch_semaphore_wait(self.cameraFrameRenderingSemaphore, DISPATCH_TIME_NOW) != 0)
    {
        return;
    }
    dispatch_semaphore_t block_camera_sema = self.cameraFrameRenderingSemaphore;
    if (_face_module) {
        
        CVPixelBufferRef imagebuffer = CMSampleBufferGetImageBuffer(samplebuffer);
        IOSurfaceRef iosurface = CVPixelBufferGetIOSurface(imagebuffer);
        int surfaceId = IOSurfaceGetID(iosurface);
        
        CMTime time = CMSampleBufferGetOutputPresentationTimeStamp(samplebuffer);
        Float64 frameTime = CMTimeGetSeconds(time) * 1000;
        
        int width = (int)CVPixelBufferGetWidth(imagebuffer);
        int height = (int)CVPixelBufferGetHeight(imagebuffer);
        
        CFRetain(samplebuffer);
        NSLog(@"surfaceId:%@", @(surfaceId));
        dispatch_async(videoQueue, ^{
            _face_module->runInContextSync([&] {
                CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(samplebuffer);
                CVPixelBufferLockBaseAddress(imageBuffer, 0);
                
                sourceCamera->setFrameData(width,
                                           height,
                                           CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0),
                                           GL_RGBA,
                                           -1,
                                           RotationMode::RotateRightFlipVertical,
                                           Opipe::SourceCamera::SourceType_YUV420SP,
                                           CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 1));
                CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
                sourceCamera->updateTargets(frameTime);
                dispatch_semaphore_signal(block_camera_sema);
            });
            CFRelease(samplebuffer);
        });
    }
}

- (void *)currentGLContext {
    if (_face_module) {
        return _face_module->currentContext()->glContext();
    }
}

- (EAGLContext *)currentContext
{
    if (_face_module) {
        return _face_module->currentContext()->currentContext();
    }
}

- (void)processVideoFrame:(CVPixelBufferRef)pixelbuffer
                timeStamp:(int64_t)timeStamp;
{
    if (!_face_module) {
        [self initModule];
    }
    
    _face_module->processVideoFrame(pixelbuffer, timeStamp);
}

- (CGFloat)whiten
{
    return _face_module->getWhitening();
}

- (CGFloat)smooth
{
    return _face_module->getSmoothing();
}

- (void)setWhiten:(CGFloat)whiten
{
    _face_module->setWhitening(whiten);
}

- (void)setSmooth:(CGFloat)smooth
{
    _face_module->setSmoothing(smooth);
}

- (CGFloat)slim
{
    return _face_module->getSlim();
}

- (void)setSlim:(CGFloat)slim
{
    _face_module->setSlim(slim);
}

- (CGFloat)eyeFactor
{
    return _face_module->getEye();
}

- (void)setEyeFactor:(CGFloat)eyeFactor
{
    _face_module->setEye(eyeFactor);
}

- (CGFloat)nose
{
    return _face_module->getNose();
}

- (void)setNose:(CGFloat)nose
{
    _face_module->setNose(nose);
}

- (void)resume
{
    if (!_face_module) {
        [self initModule];
    }
    _face_module->resume();
}

- (void)suspend
{
    if (!_face_module) {
        [self initModule];
    }
    _face_module->suspend();
}

- (void)dispose
{
    _face_module->stopModule();
    _face_module->suspend();
    delete _face_module;
    _face_module = nullptr;
}

- (void)setRenderView:(OlaFURenderView *)renderView
{
    _renderView = renderView;

    if (_face_module && _renderView) {
        Opipe::Filter *filter = _face_module->getOutputFilter();
        if (filter) {
            filter->addTarget(_renderView);
        }
        
    }
}


@end
