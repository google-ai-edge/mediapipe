#import "OlaFaceUnity.h"

#include "mediapipe/render/module/beauty/face_mesh_module.h"

@interface OlaFaceUnity() {
    Opipe::FaceMeshModule *_face_module;
}

@end
@implementation OlaFaceUnity

- (void)dealloc {
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

- (void)initModule {
    _face_module = Opipe::FaceMeshModule::create();
    NSBundle *bundle = [NSBundle bundleForClass:[self class]];
    NSURL* graphURL = [bundle URLForResource:@"face_mesh_mobile_gpu" withExtension:@"binarypb"];
    NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:nil];
    if (data) {
        _face_module->init(nullptr, (void *)data.bytes, data.length);
        _face_module->startModule();
    }
    
    
}

+ (instancetype)sharedInstance {
    static OlaFaceUnity *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}


- (void)processVideoFrame:(CVPixelBufferRef)pixelbuffer
                timeStamp:(int64_t)timeStamp;
{
    if (!_face_module) {
        [self initModule];
    }
    
    _face_module->processVideoFrame(pixelbuffer, timeStamp);
}

@end
