#ifndef OlaCameraSource_hpp
#define OlaCameraSource_hpp

#include "SourceCamera.hpp"
#include "OlaYUVTexture.hpp"
#include "OlaYUVTexture420P.hpp"
#include "Framebuffer.hpp"
#include "OlaShareTextureFilter.hpp"

#if defined(__APPLE__)
#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>
#import <CoreVideo/CoreVideo.h>
#elif defined(__ANDROID__) || defined(ANDROID)
#if defined(__ANDROID__) || defined(ANDROID)
// for EGL calls
#define EGL_EGLEXT_PROTOTYPES
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "GLES/gl.h"
#define GL_GLEXT_PROTOTYPES
#include "GLES/glext.h"
#include "android/hardware_buffer.h"

#endif
#endif

using namespace Opipe;
namespace Opipe
{
    class OlaCameraSource : public SourceCamera
    {
        private:
            Filter *_yuvTexture = nullptr;
            OlaShareTextureFilter *_scaleTexture = nullptr;
            SourceType _sourceType;
            int _lastIOSurface = -1;
            #if defined(__APPLE__)
            void _bindIOSurfaceToTexture(int iosurface, RotationMode outputRotation = RotationMode::NoRotation);
            #endif

        public:
            OlaCameraSource(Context *context, SourceType sourceType = SourceType_RGBA);
            virtual ~OlaCameraSource();
            static OlaCameraSource *create(Context *context);

        public:

            virtual void setFrameData(int width,
                                    int height, 
                                    const void* pixels,
                                    GLenum type,
                                    GLuint texture,
                                    RotationMode outputRotation = RotationMode::NoRotation,
                                    SourceType sourceType = SourceType_RGBA,
                                    const void* upixels = NULL,
                                    const void* vpixels = NULL,
                                    bool keep_white = false) override;

            #if defined(__APPLE__)
            virtual void setIORenderTexture(IOSurfaceID surfaceID,
                                    GLuint texture,
                                    int width,
                                    int height,
                                    Opipe::RotationMode outputRotation = RotationMode::NoRotation,
                                    SourceType sourceType = SourceType_RGBA,
                                    const TextureAttributes textureAttributes = Framebuffer::defaultTextureAttribures) override;
            #endif
            // 获取相机渲染后缩小的Framebuffer
            Framebuffer* getScaleFramebuffer();

            virtual Opipe::Source* addTarget(Opipe::Target* target) override;

    };
}
#endif
