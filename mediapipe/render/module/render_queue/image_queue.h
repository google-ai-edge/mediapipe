//
// Created by Felix Wang on 2022/6/24.
//

#ifndef IMAGE_QUEUE_H
#define IMAGE_QUEUE_H

#include <queue>
#if defined(__ANDROID__) || defined(ANDROID)
#include <semaphore.h>
#else
#import <Foundation/Foundation.h>
#endif
#include <mutex>
#include <atomic>
#include "LockFreeQueue.h"

struct ImageInfo {
    uint8_t *data;
    int len;
    float startX;
    float startY;
    float normalWidth;
    float normalHeight;
    int width;
    int height;
    int flag;
};

class ImageQueue : public LockFreeQueue<ImageInfo> {
private:
    static ImageQueue *instance;
    // sem_t sem;
#if defined(__APPLE__)
    dispatch_semaphore_t sem = 0;
#else
    sem_t sem;
#endif
    ImageInfo emptyInfo;

    ImageQueue(size_t capacity) : LockFreeQueue(capacity) {
        emptyInfo.data = nullptr;
        emptyInfo.len = 0;
#if defined(__APPLE__)
        sem = dispatch_semaphore_create(1);
#else
        sem_init(&sem, 0, 1);
        sem_wait(&sem);
#endif
    };

    ~ImageQueue() {
#if defined(__APPLE__)
        dispatch_semaphore_signal(sem);
        // dispatch_release(sem);
        sem = 0;
#else
        sem_destroy(&sem);
#endif
    };

    ImageQueue(const ImageQueue &);

    ImageQueue &operator=(const ImageQueue &);

public:
    static long getTimeStamp() {
        std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> tp =
                std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now());
        auto tmp = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch());
        long timestamp = tmp.count();
        return timestamp;
    }


    static ImageQueue *getInstance();

//    void push(ImageInfo &imageInfo);

    void
    push(const uint8_t *img, int len, float startX, float startY, float normalWidth, float normalHeight, int width,
         int height, bool exportFlag);

    void pop(ImageInfo &info, bool exportFlag = false);

    void dispose();
};

#endif //IMAGE_QUEUE_H
