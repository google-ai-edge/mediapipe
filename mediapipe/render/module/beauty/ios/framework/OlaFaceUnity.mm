#import "OlaFaceUnity.h"

#include "mediapipe/render/module/beauty/face_mesh_module.h"

@interface OlaFaceUnity() {
    Opipe::FaceMeshModule *_face_module;
}

@end
@implementation OlaFaceUnity

- (void)dealloc
{
    if (_face_module) {
        delete _face_module;
        _face_module = nullptr;
    }
}

- (instancetype)init
{
    self = [super init];
    if (self) {
        [self initModule];
    }
    return self;
}

- (void)initModule
{
    _face_module = Opipe::FaceMeshModule::create();
    NSBundle *bundle = [NSBundle bundleForClass:[self class]];
    NSURL* graphURL = [bundle URLForResource:@"face_mesh_mobile_gpu" withExtension:@"binarypb"];
    NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:nil];
    if (data) {
        _face_module->init(nullptr, (void *)data.bytes, data.length);
        _face_module->startModule();
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
    @autoreleasepool {
        TextureInfo rs;
        rs.ioSurfaceId = inputTexture.ioSurfaceId;
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

@end
