//
// Created by Felix Wang on 2022/6/24.
//

#include "image_queue.h"
#include <chrono>
#include <unistd.h>

ImageQueue *ImageQueue::instance = ImageQueue::getInstance();

void
ImageQueue::push(const uint8_t *img, int len, float startX, float startY, float normalWidth, float normalHeight,
                 int width, int height, bool exportFlag) {
    // LOGE("~~~~~~~~~~~~~~~~~~~~push ImageQueue push len: %d", len);

    auto *arr = (uint8_t *) malloc(len);
    memcpy(arr, img, len);
    LOGE("~~~~~~~~~~~~~~~~~~~~startX: %f,startY: %f,normalWidth: %f,normalHeight: %f", startX, startY, normalWidth,
         normalHeight);

    uint64_t afterFFi = getTimeStamp();
    ImageInfo info = {arr, len, startX, startY, normalWidth, normalHeight, width, height, false};

    rawPush(info);
    if (exportFlag) {
#if defined(__APPLE__)
        dispatch_semaphore_signal(sem);
#else
        // LOGE("~~~~~~~~~~~~~~~~~~~~push exportFlag111  true %d", sem.count);

        int rs = sem_post(&sem);
        // LOGE("~~~~~~~~~~~~~~~~~~~~push exportFlag222  true %d ,rs = %d", sem.count, rs);
#endif
    }

    // LOGE("############ push  end  %lu size:%zu  startX:%f startY:%f, normalWidth:%f normalHeight:%f,width:%d,height:%d\n",
    //  getTimeStamp(), instance->rawSize(), startX, startY, normalWidth, normalHeight, width, height);
}


void ImageQueue::pop(ImageInfo &info, bool exportFlag) {
    int g_running_count = 0;
    if (exportFlag) {
        // LOGE("~~~~~~~~~~~~~~~~~~~~~push  sem_wait33  %d", sem.count);
#if defined(__APPLE__)
        dispatch_time_t dispatchtime = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5* NSEC_PER_SEC));
        int rs = (int)dispatch_semaphore_wait(sem, dispatchtime);
#else
        int rs = sem_trywait(&sem);

        while (rs == -1 && g_running_count < 3) {
            usleep(9000);
            rs = sem_trywait(&sem);
            g_running_count++;  // Break here to show we can handle breakpoints
        }
#endif
        // LOGE("~~~~~~~~~~~~~~~~~~~~~push  sem_wait44 %d, rs=%d", sem.count, rs);
        if (instance->rawSize() > 0) {
            instance->rawPop(info, exportFlag);
            //     LOGE("########## pop  end 1  %ld\n", getTimeStamp());
        } else {
//            LOGE("~~~~~~~~~~~~~~33 pop empty  %ld\n", getTimeStamp());
            info = instance->emptyInfo;
        }
    } else {
        if (instance->rawSize() > 0) {
            instance->rawPop(info, exportFlag);
            //     LOGE("########## pop  end 1  %ld\n", getTimeStamp());
        } else {
//            LOGE("~~~~~~~~~~~~~~33 pop empty  %ld\n", getTimeStamp());
            info = instance->emptyInfo;
        }
    }

}

void ImageQueue::dispose() {
#if defined(__APPLE__)
    dispatch_semaphore_signal(sem);
#else
    sem_post(&sem);
#endif
    rawRelease();
}

ImageQueue *ImageQueue::getInstance() {
    if (!instance) {
        instance = new ImageQueue(2);
    } else {
        return instance;
    }
    return instance;
}

