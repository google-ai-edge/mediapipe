//
// Created by Felix Wang on 2022/6/29.
//

#include <jni.h>
#include <malloc.h>
#include "OlaRender.h"
#include "image_queue.h"


// #include <android/log.h>
#include <jni.h>

// #define TAG    "ImageQueue-jni" // 这个是自定义的LOG的标识
// #define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__) // 定义LOGI类型
// #define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__) // 定义LOGE类型


extern "C" 
JNIEXPORT jlong JNICALL
Java_com_ola_render_RenderJni_create(JNIEnv *env, jobject thiz) {
    auto *render = Opipe::OlaRender::create();
    return reinterpret_cast<int64_t>(render);
}

extern "C" 
JNIEXPORT jint JNICALL
Java_com_ola_render_RenderJni_render(JNIEnv *env, jobject thiz, jlong render_context, jint texture_id,
                                             jint width, jint height, jlong timestamp, jboolean exportFlag) {
    auto *render = reinterpret_cast<Opipe::OlaRender *>(render_context);
    Opipe::TextureInfo info;
    info.textureId = texture_id;
    info.width = width;
    info.height = height;
    info.frameTime = timestamp;
    auto res = render->render(info, exportFlag);
    return res.textureId;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ola_render_RenderJni_release(JNIEnv *env, jobject thiz, jlong renderId) {
    auto *render = reinterpret_cast<Opipe::OlaRender *>(renderId);
    render->release();
}

static JNINativeMethod methods[] = {
        {"create", "()J", reinterpret_cast<void*>(Java_com_ola_render_RenderJni_create)},
        {"render", "(JIIIJZ)I", reinterpret_cast<void*>(Java_com_ola_render_RenderJni_render)},
        {"release", "(J)V", reinterpret_cast<void*>(Java_com_ola_render_RenderJni_release)}
};

extern "C" 
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
    JNIEnv* env;
    if (JNI_OK != jvm->GetEnv(reinterpret_cast<void**> (&env),JNI_VERSION_1_6)) {
        LOGE("JNI_OnLoad could not get JNI env");
        return JNI_ERR;
    }
    jclass clazz = env->FindClass("com/ola/render/RenderJni");  //获取Java NativeLib类
    if (clazz == nullptr) {
        LOGE ( "find class com.ola.render.RenderJni failed\n");
        return JNI_VERSION_1_6;
    }
	//注册Native方法
    if (env->RegisterNatives(clazz, methods, sizeof(methods)/sizeof((methods)[0])) < 0) {
        LOGE("RegisterNatives error");
        return JNI_ERR;
    }
 
    return JNI_VERSION_1_6;
}

extern "C" 
JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
  
}