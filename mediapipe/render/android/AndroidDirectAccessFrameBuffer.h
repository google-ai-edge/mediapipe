//
// Created by  jormin on 2021/4/30.
//

#ifndef QUARAMERA_ANDROIDDIRECTACCESSFRAMEBUFFER_H
#define QUARAMERA_ANDROIDDIRECTACCESSFRAMEBUFFER_H


// #include "GPUImageMacros.h"
// #include "Framebuffer.hpp"
// #include "GLES/gl.h"
// #include "PlatformEGLAndroidCompat.h"
// #include "android_hardware_buffer_compat.h"
// #include "EGLAndroid.h"


/**
 *  目前仅仅支持hardwarebuffer的高效读取，如果以后要支持upload的话，需要改造
 */

namespace QImage {
    class AndroidDirectAccessFrameBuffer {
        ~AndroidDirectAccessFrameBuffer() {};
        AndroidDirectAccessFrameBuffer() {};
    };
}

// NS_GI_BEGIN


//     class AndroidDirectAccessFrameBuffer : public QImage::Framebuffer {
//     public:


//         AndroidDirectAccessFrameBuffer(Context *context, int width, int height,
//                                        const TextureAttributes textureAttributes = defaultTextureAttribures);


//         virtual ~AndroidDirectAccessFrameBuffer() override;

//         void lockAddress() override;

//         void unlockAddress() override;

//         void *frameBufferGetBaseAddress() override;

//         int getBytesPerRow() override;

//         void _generateTexture() override;

//         void _generateFramebuffer(bool needGenerateTexture = true) override;

//         bool support() {
//             return _support;
//         }

//     private :

//         bool _generateHardwareBuffer();

//         EGLImageKHR _imageEGL = EGL_NO_IMAGE_KHR;
//         void *_hardwareBufferReadData = nullptr;

//         AHardwareBuffer *_graphicBuf = nullptr;
//         AHardwareBuffer_Desc *_graphicBufDes = nullptr;

//         bool _support = false;

//         bool _hasLock = false;


//     };

// NS_GI_END


#endif //QUARAMERA_ANDROIDDIRECTACCESSFRAMEBUFFER_H
