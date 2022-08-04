#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <GLKit/GLKTextureLoader.h>

typedef struct {
    int cvPixelFormat;
    MTLPixelFormat  mtlFormat;
    GLuint  glInternalFormat;
    GLuint  glFormat;
    GLuint  glType;
} OlaTextureFormatInfo;

@interface OlaShareTexture : NSObject

- (nonnull instancetype)initWithMetalDevice:(nonnull id<MTLDevice>)mtlDevice
                              openGLContext:(nonnull EAGLContext*)glContext
                           metalPixelFormat:(MTLPixelFormat)mtlPixelFormat
                                       size:(CGSize)size;

- (nonnull instancetype)initWithMetalDevice:(nonnull id<MTLDevice>)mtlDevice
                              openGLContext:(nonnull EAGLContext*)glContext
                           metalPixelFormat:(MTLPixelFormat)mtlPixelFormat
                                sourceImage:(nonnull UIImage *)sourceImage;

@property (readonly, nonnull, nonatomic) CVPixelBufferRef renderTarget;
@property (readonly, nonnull, nonatomic) id<MTLDevice> metalDevice;
@property (readonly, nonnull, nonatomic) id<MTLTexture> metalTexture;

@property (readonly, nonnull, nonatomic) EAGLContext *openGLContext;
@property (readonly, nonatomic) GLuint openGLTexture;

@property (readonly, nonatomic) CGSize size;
@property (strong, nullable, nonatomic) NSString *name;
@property (readonly, nonnull, nonatomic) OlaTextureFormatInfo *formatInfo;
@property (readonly, nonatomic) IOSurfaceID surfaceID;

@end
