//
// Created by  jormin on 2021/4/3.
//

#include <jni.h>

extern "C" {
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
    return JNI_VERSION_1_6;
}
}