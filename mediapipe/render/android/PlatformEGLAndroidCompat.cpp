//
// Created by  jormin on 2021/4/22.
//

#include "PlatformEGLAndroidCompat.h"
//
//
//AHardwareBuffer_Desc


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

#define LOG_TAG "PlatformEGLAndroidCompat"

#include "no_destructor.h"


NS_GI_BEGIN
#define UTILS_PRIVATE __attribute__((visibility("hidden")))

    // The Android NDK doesn't exposes extensions, fake it with eglGetProcAddress
    namespace glext {

        UTILS_PRIVATE PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR = {};

        UTILS_PRIVATE PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBufferANDROID = {};

        UTILS_PRIVATE PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES = {};

        UTILS_PRIVATE PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR = {};

    }
    using namespace glext;

    PlatformEGLAndroidCompat::PlatformEGLAndroidCompat() noexcept {
        mIsSupport = _createDriver();
    }

    bool PlatformEGLAndroidCompat::isSupport() {
        return mIsSupport;
    }


    // static
    PlatformEGLAndroidCompat &PlatformEGLAndroidCompat::GetInstance() noexcept {
        static NoDestructor<PlatformEGLAndroidCompat> compat;
        return *compat;
    }

    void
    PlatformEGLAndroidCompat::glEGLImageTargetTexture2DOES(GLenum target, GLeglImageOES image) {
        glext::glEGLImageTargetTexture2DOES(target, image);
    }


    bool PlatformEGLAndroidCompat::eglDestroyImageKHR(EGLDisplay dpy, EGLImageKHR image) {
        return glext::eglDestroyImageKHR(dpy, image);
    }


    EGLImageKHR
    PlatformEGLAndroidCompat::eglCreateImageKHR(EGLDisplay dpy, EGLContext ctx, EGLenum target,
                                                EGLClientBuffer buffer, const EGLint *attrib_list) {
        return glext::eglCreateImageKHR(dpy, ctx, target, buffer, attrib_list);
    }


    EGLClientBuffer PlatformEGLAndroidCompat::eglGetNativeClientBufferANDROID(
            const struct AHardwareBuffer *buffer) {
        return glext::eglGetNativeClientBufferANDROID(buffer);
    }


    bool PlatformEGLAndroidCompat::_createDriver() noexcept {
        const char *const driver_absolute_path = "/system/lib/egl/libEGL_mali.so";
        // On Gingerbread you have to load symbols manually from Mali driver because
        // Android EGL library has a bug.
        // From  ICE CREAM SANDWICH you can freely use the eglGetProcAddress function.
        // You might be able to get away with just eglGetProcAddress (no dlopen).
        // Try it,  else revert to the following code
        void *dso = dlopen(driver_absolute_path, RTLD_LAZY);
        if (dso != 0) {
            glext::eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC) dlsym(dso, "eglCreateImageKHR");
            glext::eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC) dlsym(dso,
                                                                          "eglDestroyImageKHR");
        } else {
            QImage::Log("PlatformEGLAndroidCompat",
                          "dlopen: FAILED! Loading functions in common way!");
            glext::eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC) eglGetProcAddress(
                    "eglCreateImageKHR");
            glext::eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC) eglGetProcAddress(
                    "eglDestroyImageKHR");
        }

        if (glext::eglCreateImageKHR == nullptr) {
            QImage::Log("PlatformEGLAndroidCompat",
                          "Error: Failed to find eglCreateImageKHR at %s:%in", __FILE__, __LINE__);
            return false;
        }
        if (glext::eglDestroyImageKHR == nullptr) {
            QImage::Log("PlatformEGLAndroidCompat",
                          "Error: Failed to find eglDestroyImageKHR at %s:%in", __FILE__, __LINE__);
            return false;
        }

        glext::eglGetNativeClientBufferANDROID = (PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC) eglGetProcAddress(
                "eglGetNativeClientBufferANDROID");

        glext::glEGLImageTargetTexture2DOES = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC) eglGetProcAddress(
                "glEGLImageTargetTexture2DOES");

        if (glext::eglGetNativeClientBufferANDROID == nullptr) {
            QImage::Log("PlatformEGLAndroidCompat",
                          "Error: Failed to find eglGetNativeClientBufferANDROID at %s:%in",
                          __FILE__, __LINE__);
            return false;
        }

        if (glext::glEGLImageTargetTexture2DOES == nullptr) {
            QImage::Log("PlatformEGLAndroidCompat",
                          "Error: Failed to find glEGLImageTargetTexture2DOES at %s:%in", __FILE__,
                          __LINE__);
            return false;
        }
        return true;
    }

NS_GI_END
