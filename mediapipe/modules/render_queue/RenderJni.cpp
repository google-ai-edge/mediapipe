//
// Created by Felix Wang on 2022/6/29.
//

#include <jni.h>
#include <malloc.h>
#include "OlaRender.hpp"
#include "image_queue.h"


#include <android/log.h>
#include <jni.h>

#define TAG    "ImageQueue-jni" // 这个是自定义的LOG的标识
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__) // 定义LOGI类型
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__) // 定义LOGE类型


extern "C"
JNIEXPORT jlong JNICALL
Java_com_weatherfish_render_RenderJni_create(JNIEnv *env, jobject thiz) {
    auto *render = OLARender::OlaRender::create();
    return reinterpret_cast<int64_t>(render);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_weatherfish_render_RenderJni_render(JNIEnv *env, jobject thiz, jlong render_context, jint texture_id,
                                             jint width, jint height, jlong timestamp, jboolean exportFlag) {
    auto *render = reinterpret_cast<OLARender::OlaRender *>(render_context);
    OLARender::TextureInfo info;
    info.textureId = texture_id;
    info.width = width;
    info.height = height;
    info.frameTime = timestamp;
    auto res = render->render(info, exportFlag);
    return res.textureId;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_weatherfish_render_RenderJni_release(JNIEnv *env, jobject thiz, jlong renderId) {
    auto *render = reinterpret_cast<OLARender::OlaRender *>(renderId);
    render->release();
}
