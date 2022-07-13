//
// Created by  jormin on 2021/4/22.
//

#ifndef QUARAMERA_PLATFORM_EGL_ANDROID_H
#define QUARAMERA_PLATFORM_EGL_ANDROID_H

#include "EGL/egl.h"
#include "GLES/gl.h"
#include "EGL/egl.h"
#include "GLES/glext.h"
#include <GLES2/gl2ext.h>
#include <sys/system_properties.h>

#include <cstdlib>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#include <GLES/glext.h>
#include "GPUImageMacros.h"

NS_GI_BEGIN class PlatformEGLAndroidCompat {
    public:
        static PlatformEGLAndroidCompat &GetInstance() noexcept;

        virtual ~PlatformEGLAndroidCompat() = default;

        PlatformEGLAndroidCompat() noexcept;

        virtual void glEGLImageTargetTexture2DOES(GLenum target, GLeglImageOES image);

        virtual bool eglDestroyImageKHR(EGLDisplay dpy, EGLImageKHR image);

        virtual EGLImageKHR
        eglCreateImageKHR(EGLDisplay dpy, EGLContext ctx, EGLenum target, EGLClientBuffer buffer,
                          const EGLint *attrib_list);

        virtual EGLClientBuffer
        eglGetNativeClientBufferANDROID(const struct AHardwareBuffer *buffer);

        virtual bool isSupport();

    private:
        static bool _createDriver() noexcept;

        int mOSVersion = 0;

        bool mIsSupport = false;
    };

NS_GI_END
#endif //QUARAMERA_PLATFORM_EGL_ANDROID_H
