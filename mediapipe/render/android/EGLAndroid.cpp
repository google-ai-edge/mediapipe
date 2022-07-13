//
// Created by  jormin on 2021/4/30.
//

#include "EGLAndroid.h"
#include "android_hardware_buffer_compat.h"
#include "PlatformEGLAndroidCompat.h"

NS_GI_BEGIN

    std::mutex EGLAndroid::mMutex;

    int EGLAndroid::mGLMajorVersion = -1;
    int EGLAndroid::mGLMinorVersion = -1;


    bool EGLAndroid::supportHardwareBuffer() {
        //ios not support
#if defined(__APPLE__)
        if(true){
            return false;
        }
#endif

        //GL版本 > 3.0
        if (EGLAndroid::getGLMajorVersion() < 3) {
            return false;
        }

        //AHardwareBuffer要求android系统>=7.0
        if (!AndroidHardwareBufferCompat::IsSupportAvailable()) {
            return false;
        }

        //EGL接口动态dlopen成功
        if (!PlatformEGLAndroidCompat::GetInstance().isSupport()) {
            return false;
        }

        return true;
    }


    bool EGLAndroid::supportPBO() {
        //GL版本 > 3.0
        if (EGLAndroid::getGLMajorVersion() < 3) {
            return false;
        }
        return true;
    }

    int EGLAndroid::getGLMajorVersion() {
        std::unique_lock<std::mutex> lk(mMutex);
        if (mGLMajorVersion != -1) {
            return mGLMajorVersion;
        }
        _initGLInfo();
        return mGLMajorVersion;
    }

    int EGLAndroid::getGLMinorVersion() {
        std::unique_lock<std::mutex> lk(mMutex);
        if (mGLMinorVersion != -1) {
            return mGLMinorVersion;
        }
        _initGLInfo();
        return mGLMinorVersion;
    }

    void EGLAndroid::_initGLInfo() {
        CHECK_GL(glGetIntegerv(GL_MAJOR_VERSION, &mGLMajorVersion));
        CHECK_GL(glGetIntegerv(GL_MINOR_VERSION, &mGLMinorVersion));
    }


NS_GI_END