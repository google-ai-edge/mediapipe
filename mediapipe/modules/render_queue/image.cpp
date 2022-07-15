#include "image.h"
#include "image_queue.h"
#include <cstdlib>
#include <cstdio>

#ifdef ANDROID

#include <jni.h>

//extern "C" JNIEXPORT void JNICALL
//Java_com_felix_text_1plugin_ImageCache_addImageCache(JNIEnv *env, jclass clazz, jbyteArray img, jint len) {
//
//    auto *arr = (unsigned char *) malloc(len);
//    (*env).GetByteArrayRegion(img, 0, len, (jbyte *) arr);
//    ImageInfo info;
//    ImageQueue.
//}
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_felix_text_1plugin_ImageCache_addImageCache(JNIEnv *env, jclass clazz, jbyteArray img, jint len, jint width,
//                                                     jint height,
//                                                     jlong timestamp) {
//    auto *arr = (unsigned char *) malloc(len);
//    (*env).GetByteArrayRegion(img, 0, len, (jbyte *) arr);
//    ImageQueue::getInstance()->push(arr, len, width, height, (uint32_t) timestamp);
//}
#else

#endif

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void addImageCache(const uint8_t *img, int len, double startX, double startY, double normalWidth, double normalHeight,
                   int width, int height, uint64_t javaTime, uint64_t startT, uint64_t beforeFFi, bool exportFlag) {
    ImageQueue::getInstance()->push(img, len, startX, startY, normalWidth, normalHeight, width, height, javaTime,
                                    startT, beforeFFi, exportFlag);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
void dispose() {
    ImageQueue::getInstance()->dispose();
}