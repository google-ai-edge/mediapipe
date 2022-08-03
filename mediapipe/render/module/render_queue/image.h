
#ifndef IMAGE_H
#define IMAGE_H

#include "image_queue.h"
#include <stdint.h>

// #ifdef __cplusplus
extern "C"
{
// #endif

__attribute__((visibility("default"))) __attribute__((used))
void addImageCache(const uint8_t *img, int len, double startX, double startY, double normalWidth, double normalHeight,
                   int width, int height, bool exportFlag);

__attribute__((visibility("default"))) __attribute__((used))
void disposeImage();

__attribute__((visibility("default"))) __attribute__((used))
void test();

// #ifdef __cplusplus
}
// #endif

#endif