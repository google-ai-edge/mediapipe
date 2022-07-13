//
// Created by  jormin on 2021/4/30.
//

#include "AndroidDirectAccessFrameBuffer.h"

/**
 * {@see https://android.googlesource.com/platform/cts/+/master/tests/tests/nativehardware/jni/AHardwareBufferGLTest.cpp}
 */
namespace QImage {

}

// NS_GI_BEGIN

//     USING_NS_GI
//     using namespace QImage;

//     AndroidDirectAccessFrameBuffer::AndroidDirectAccessFrameBuffer(Context *context, int width,
//                                                                    int height,
//                                                                    const TextureAttributes textureAttributes)
//             : QImage::Framebuffer() {
//         _context = context;
//         useTextureCache = true;
//         _width = width;
//         _height = height;
//         _textureAttributes = textureAttributes;

//         /**
//          * As a general rule, you should never call virtual functions in constructors or destructors.
//          * If you do, those calls will never go to a more derived class than the currently executing constructor or destructor.
//          * In other words, during construction and destruction, virtual functions aren't virtual.
//          */
//         AndroidDirectAccessFrameBuffer::_generateFramebuffer(true/*not use*/);

//         _context->_framebuffers.push_back(this);
//     }


//     AndroidDirectAccessFrameBuffer::~AndroidDirectAccessFrameBuffer() {
//         if (_imageEGL) {
//             QImage::Log("AHardwareBuffer", "release _imageEGL");
//             eglDestroyImageKHR(eglGetDisplay(EGL_DEFAULT_DISPLAY), _imageEGL);
//             _imageEGL = nullptr;
//         }
//         if (_graphicBuf) {
//             QImage::Log("AHardwareBuffer", "release AHardwareBuffer");
//             QImage::AndroidHardwareBufferCompat::GetInstance().Release(_graphicBuf);
//             _graphicBuf = nullptr;

//             if (_graphicBufDes != nullptr) {
//                 delete _graphicBufDes;
//                 _graphicBufDes = nullptr;
//             }

//             //TODO 确认release的时候，是否需要我们自己释放这块内存
// //            _hardwareBufferReadData = nullptr;
//         }
//         //TODO 需要验证FrameBuffer里面的texture和fb是否成功释放了
//     }

//     void AndroidDirectAccessFrameBuffer::lockAddress() {
//         if (!_support) {
//             return;
//         }

//         if (_hasLock) {
//             return;
//         }

//         void *data = nullptr;
//         int result = AndroidHardwareBufferCompat::GetInstance().Lock(_graphicBuf,
//                                                                      AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
//                                                                      -1, nullptr, &data);
//         if (result != 0) {
//             LogE("AHardwareBuffer", "Lock pixel error : %d", result);
//             return;
//         }
//         _hardwareBufferReadData = data;
//         _hasLock = true;
//     }

//     /**
//      * unlock之后，不要继续使用_hardwareBufferReadData，会存在GPU写CPU读并行导致异常问题
//      */
//     void AndroidDirectAccessFrameBuffer::unlockAddress() {
//         if (!_support) {
//             return;
//         }

//         if (!_hasLock) {
//             return;
//         }

//         int unlockResult = AndroidHardwareBufferCompat::GetInstance().Unlock(_graphicBuf, nullptr);

//         if (unlockResult != 0) {
//             LogE("AHardwareBuffer", "Unlock pixel error : %d", unlockResult);
//             return;
//         }

//         _hardwareBufferReadData = nullptr;
//         _hasLock = false;
//     }

//     int AndroidDirectAccessFrameBuffer::getBytesPerRow() {
//         if (!_support) {
//             return 0;
//         }
//         return _graphicBufDes->stride * 4;

//     }

//     void *AndroidDirectAccessFrameBuffer::frameBufferGetBaseAddress() {
//         if (_support) {
//             return _hardwareBufferReadData;
//         }
//         return nullptr;
//     }

//     static void
//     getGLFormat(AHardwareBuffer_Desc *desc, int *internal_format, int *format, int *type) {
//         switch ((*desc).format) {
//             case GL_RGB565:
//                 *internal_format = GL_RGB;
//                 *format = GL_RGB;
//                 *type = GL_UNSIGNED_SHORT_5_6_5;
//                 break;
//             case GL_RGB8:
//                 *internal_format = GL_RGB;
//                 *format = GL_RGB;
//                 *type = GL_UNSIGNED_BYTE;
//                 break;
//             case GL_RGBA8:
//                 *internal_format = GL_RGBA;
//                 *format = GL_RGBA;
//                 *type = GL_UNSIGNED_BYTE;
//                 break;
//             case GL_SRGB8_ALPHA8:
//                 // Available through GL_EXT_sRGB.
//                 *internal_format = GL_SRGB_ALPHA_EXT;
//                 *format = GL_RGBA;
//                 *type = GL_UNSIGNED_BYTE;
//                 break;
//             case GL_DEPTH_COMPONENT16:
//                 // Available through GL_OES_depth_texture.
//                 // Note that these are treated as luminance textures, not as red textures.
//                 *internal_format = GL_DEPTH_COMPONENT;
//                 *format = GL_DEPTH_COMPONENT;
//                 *type = GL_UNSIGNED_SHORT;
//                 break;
//             case GL_DEPTH24_STENCIL8:
//                 // Available through GL_OES_packed_depth_stencil.
//                 *internal_format = GL_DEPTH_STENCIL_OES;
//                 *format = GL_DEPTH_STENCIL;
//                 *type = GL_UNSIGNED_INT_24_8;
//                 break;
//             case AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM :
//                 // only support this format now
//                 *internal_format = GL_RGBA;
//                 *format = GL_RGBA;
//                 *type = GL_UNSIGNED_BYTE;
//                 break;
//             default:
//                 LogE("AHardwareBuffer", "covert to gl  format error");
//                 assert(false);
//         }
//     }

//     void AndroidDirectAccessFrameBuffer::_generateFramebuffer(bool needGenerateTexture) {

//         bool result = _generateHardwareBuffer();

//         if (!result) {
//             LogE("AHardwareBuffer", "not support because of create hardware buffer error ");
//             //check result
//             _support = false;
//             return;
//         }

//         CHECK_GL(glGenFramebuffers(1, &_framebuffer));
//         CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer));

//         _generateTexture();

//         CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));

//         int format = -1;
//         int type = -1;
//         int internal_format = -1;

//         getGLFormat(_graphicBufDes, &internal_format, &format, &type);

//         Log("AHardwareBuffer", "covert to gl format ( %d -> [internal:%d , format:%d , type: %d] )",
//             _graphicBufDes->format, internal_format, format, type);


//         CHECK_GL(glTexImage2D(GL_TEXTURE_2D, 0, internal_format, _width, _height, 0, format, type,
//                               0));

//         CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
//                                         _texture, 0));

//         //bind texture to hardwarebuffer
//         CHECK_GL(PlatformEGLAndroidCompat::GetInstance().glEGLImageTargetTexture2DOES(GL_TEXTURE_2D,
//                                                                                       _imageEGL));

//         CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
//         CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

//         QImage::Log("AHardwareBuffer",
//                       "AHardwareBuffer create finish ( framebuffer:%d, texture:%d , w:%d , h:%d , stride:%d )",
//                       _framebuffer, _texture, _width, _height, _graphicBufDes->stride);

//         _support = true;
//     }


//     void AndroidDirectAccessFrameBuffer::_generateTexture() {

//         CHECK_GL(glGenTextures(1, &_texture));

//         QImage::Log("AHardwareBuffer", "glGenTextures %d", _texture);

//         CHECK_GL(glBindTexture(GL_TEXTURE_2D, _texture));
//         CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
//                                  _textureAttributes.minFilter));
//         CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
//                                  _textureAttributes.magFilter));
//         CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, _textureAttributes.wrapS));
//         CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, _textureAttributes.wrapT));

//         QImage::Log("AHardwareBuffer", "finish glGenTextures");

//         CHECK_GL(glBindTexture(GL_TEXTURE_2D, 0));
//     }

//     bool AndroidDirectAccessFrameBuffer::_generateHardwareBuffer() {

//         AHardwareBuffer_Desc tryDesc;

//         int hardwareBufferFormat;
//         switch (_textureAttributes.format) {
//             case GL_RGBA8:
//             case GL_RGBA:
//                 hardwareBufferFormat = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
//                 break;
//             default:
//                 //not support other format now
//                 assert(false);
//         }

//         //暂时不支持minmap的缩放方法
//         switch (_textureAttributes.magFilter) {
//             case GL_LINEAR:
//             case GL_NEAREST:
//                 break;
//             default:
//                 LogE("AHardwareBuffer", "not support magFilter type :  %d",
//                      _textureAttributes.magFilter);
//                 //not support other format now
//                 assert(false);
//         }

//         switch (_textureAttributes.minFilter) {
//             case GL_LINEAR:
//             case GL_NEAREST:
//                 break;
//             default:
//                 LogE("AHardwareBuffer", "not support magFilter type :  %d",
//                      _textureAttributes.minFilter);
//                 //not support other format now
//                 assert(false);
//         }

//         // filling in the usage for HardwareBuffer
//         tryDesc.format = hardwareBufferFormat;
//         tryDesc.height = _height;
//         tryDesc.width = _width;
//         tryDesc.layers = 1;
//         tryDesc.rfu0 = 0;
//         tryDesc.rfu1 = 0;
//         tryDesc.stride = 0;
//         tryDesc.usage =
//                 AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER |
//                 AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;

//         QImage::Log("AHardwareBuffer", "start create  AHardwareBuffer_Desc (%d, %d) ", _width,
//                       _height);

//         // create GraphicBuffer
//         int errorCode = AndroidHardwareBufferCompat::GetInstance().Allocate(&tryDesc, &_graphicBuf);


//         if (errorCode != 0) {
//             LogE("AHardwareBuffer", "AHardwareBuffer_allocate error = %d ", errorCode);
//             //TODO 不支持,需要进行统计或者降级!!
//             return false;
//         }

//         if (_graphicBufDes == nullptr) {
//             _graphicBufDes = new AHardwareBuffer_Desc;
//         }
//         //实际的hardware buffer的参数
//         AndroidHardwareBufferCompat::GetInstance().Describe(_graphicBuf, _graphicBufDes);

//         QImage::Log("AHardwareBuffer", "AHardwareBuffer_allocate success (%d, %d) stride : %d ",
//                       _graphicBufDes->width, _graphicBufDes->height, _graphicBufDes->stride);

//         // get the native buffer
//         EGLClientBuffer clientBuf = PlatformEGLAndroidCompat::GetInstance().eglGetNativeClientBufferANDROID(
//                 _graphicBuf);

//         // obtaining the EGL display
//         EGLDisplay disp = eglGetDisplay(EGL_DEFAULT_DISPLAY);

//         // specifying the image attributes
//         EGLint eglImageAttributes[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};


//         _imageEGL = PlatformEGLAndroidCompat::GetInstance().eglCreateImageKHR(disp, EGL_NO_CONTEXT,
//                                                                               EGL_NATIVE_BUFFER_ANDROID,
//                                                                               clientBuf,
//                                                                               eglImageAttributes);

//         //AHardwareBuffer allocation succeeded, but binding it to an EGLImage failed. "
//         //This is usually caused by a version mismatch between the gralloc implementation and "
//         //the OpenGL/EGL driver. Please contact your GPU vendor to resolve this problem."
//         if (_imageEGL == EGL_NO_IMAGE_KHR) {
//             LogE("AHardwareBuffer", "eglCreateImageKHR error");
//             assert(false);
//         }

//         return true;
//     }


// NS_GI_END
