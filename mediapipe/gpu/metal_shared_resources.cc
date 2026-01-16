#import "mediapipe/gpu/metal_shared_resources.h"

@interface MPPMetalSharedResources ()
@end

@implementation MPPMetalSharedResources {
}

@synthesize mtlDevice = _mtlDevice;
@synthesize mtlCommandQueue = _mtlCommandQueue;
#if COREVIDEO_SUPPORTS_METAL
@synthesize mtlTextureCache = _mtlTextureCache;
#endif

- (instancetype)init {
  self = [super init];
  if (self) {
  }
  return self;
}

- (void)dealloc {
#if COREVIDEO_SUPPORTS_METAL
  if (_mtlTextureCache) {
    CFRelease(_mtlTextureCache);
    _mtlTextureCache = NULL;
  }
#endif
}

- (id<MTLDevice>)mtlDevice {
  @synchronized(self) {
    if (!_mtlDevice) {
      _mtlDevice = MTLCreateSystemDefaultDevice();
    }
  }
  return _mtlDevice;
}

- (id<MTLCommandQueue>)mtlCommandQueue {
  @synchronized(self) {
    if (!_mtlCommandQueue) {
      _mtlCommandQueue = [self.mtlDevice newCommandQueue];
    }
  }
  return _mtlCommandQueue;
}

#if COREVIDEO_SUPPORTS_METAL
- (CVMetalTextureCacheRef)mtlTextureCache {
  @synchronized(self) {
    if (!_mtlTextureCache) {
      CVReturn __unused err = CVMetalTextureCacheCreate(
          NULL, NULL, self.mtlDevice, NULL, &_mtlTextureCache);
      NSAssert(err == kCVReturnSuccess,
               @"Error at CVMetalTextureCacheCreate %d ; device %@", err,
               self.mtlDevice);
      // TODO: register and flush metal caches too.
    }
  }
  return _mtlTextureCache;
}
#endif

@end

namespace mediapipe {

MetalSharedResources::MetalSharedResources() {
  resources_ = [[MPPMetalSharedResources alloc] init];
}
MetalSharedResources::~MetalSharedResources() {}

}  // namespace mediapipe
