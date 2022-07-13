//
// Created by  jormin on 2021/4/30.
//

#ifndef QUARAMERA_EGLANDROID_H
#define QUARAMERA_EGLANDROID_H

#include "GPUImageMacros.h"
#include <mutex>
#import <GLES3/gl3.h>
#include "android_hardware_buffer_compat.h"
#include "PlatformEGLAndroidCompat.h"

NS_GI_BEGIN

    class EGLAndroid {
    public :
        static int getGLMajorVersion();

        static int getGLMinorVersion();

        static bool supportHardwareBuffer();

        static bool supportPBO();

    private :

        static void _initGLInfo();

        static std::mutex mMutex;

        static int mGLMajorVersion;
        static int mGLMinorVersion;


    }; NS_GI_END

#endif //QUARAMERA_EGLANDROID_H
