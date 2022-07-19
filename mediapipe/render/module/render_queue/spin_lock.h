//
// Created by Felix Wang on 2022/7/12.
//

#ifndef ANDROID_SPIN_LOCK_H
#define ANDROID_SPIN_LOCK_H

class spin_lock {
private:

    std::atomic_flag _atomic;

public:

    spin_lock();

    void lock();

    void unlock();

    bool try_lock();
};

spin_lock::spin_lock() : _atomic(ATOMIC_FLAG_INIT) {}

void spin_lock::lock() {
    while (_atomic.test_and_set(std::memory_order_acquire));
}

void spin_lock::unlock() {
    _atomic.clear(std::memory_order_release);
}

bool spin_lock::try_lock() {
    return _atomic.test() ? false : (_atomic.test_and_set(std::memory_order_acquire));
}

#endif //ANDROID_SPIN_LOCK_H
