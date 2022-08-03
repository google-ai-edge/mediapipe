//
// Created by Felix Wang on 2022/7/7.
//


#pragma once

#include <atomic>

#ifdef ANDROID
#include <android/log.h>

#define TAG    "LockFreeQueue" // 这个是自定义的LOG的标识
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__) // 定义LOGI类型
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__) // 定义LOGE类型
#else
#define LOGI(...) fprintf(stdout, __VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif

template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue(size_t capacity) {
        _capacityMask = capacity - 1;
        for (size_t i = 1; i <= sizeof(void *) * 4; i <<= 1)
            _capacityMask |= _capacityMask >> i;
        _capacity = _capacityMask + 1;

        _queue = (Node * )
        new char[sizeof(Node) * _capacity];
        for (size_t i = 0; i < _capacity; ++i) {
            _queue[i].tail.store(i, std::memory_order_relaxed);
            _queue[i].head.store(-1, std::memory_order_relaxed);
        }

        _tail.store(0, std::memory_order_relaxed);
        _head.store(0, std::memory_order_relaxed);
    }

    ~LockFreeQueue() {
        rawRelease();
    }

    void releaseNode(T &data) {
        // LOGE("~~~~~~~~~~releaseNode data: %ld", reinterpret_cast<int64_t>(data.data));
        if (data.data)
            free(data.data);
    }

    void rawRelease() {
        size_t head = _head.load(std::memory_order_acquire);
        for (size_t i = head; i != _tail; ++i)
            (&_queue[i & _capacityMask].data)->~T();

        delete[] (char *) _queue;
    }

    size_t rawCapacity() const { return _capacity; }

    size_t rawSize() const {
        size_t head = _head.load(std::memory_order_acquire);
        return _tail.load(std::memory_order_relaxed) - head;
    }

    bool rawPush(const T &data) {
        Node *node;
        size_t tail = _tail.load(std::memory_order_relaxed);
        for (;;) {
            node = &_queue[tail & _capacityMask];
            if (node->tail.load(std::memory_order_relaxed) != tail)
                return false;
            if ((_tail.compare_exchange_weak(tail, tail + 1, std::memory_order_relaxed)))
                break;
        }
        new(&node->data)T(data);
        // LOGE("~~~~~~~~~~rawPush data: %ld", reinterpret_cast<int64_t>(node->data.data));
        node->head.store(tail, std::memory_order_release);
        return true;
    }

    bool rawPop(T &result, bool exportFlag) {
        // LOGE("~~~~~~~~~~33");
        Node *node;
        size_t head = _head.load(std::memory_order_acquire);
        for (;;) {
            node = &_queue[head & _capacityMask];
            if (node->data.len == 0) {
                // LOGE("~~~~~~~~~~33 len == 0 ");
                continue;
            } else {
                // LOGE("~~~~~~~~~~33 node->data.len == %d ", node->data.len);
            }
            if (node->head.load(std::memory_order_relaxed) != head) {
                // LOGE("~~~~~~~~~~33 return false ");
                return false;
            }
            if (_head.compare_exchange_weak(head, head + 1, std::memory_order_relaxed))
                break;
        }
        result = node->data;
//        if (() > 1)
//        (&node->data)->~T();
        node->tail.store(head + _capacity, std::memory_order_release);
        return true;
    }

private:
    struct Node {
        T data;
        std::atomic<size_t> tail;
        std::atomic<size_t> head;
    };

private:
    size_t _capacityMask;
    Node *_queue;
    size_t _capacity;
    std::atomic<size_t> _tail;
    std::atomic<size_t> _head;
};
