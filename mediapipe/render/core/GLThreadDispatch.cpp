//
// Created by  jormin on 2021/6/23.
//

#if defined(__APPLE__)
#include "GLThreadDispatch.h"
#else
#include "QStream/Dispatcher/GLThreadDispatch.h"
#endif
#include <assert.h>


using namespace QStream;


GLThreadDispatch::GLThreadDispatch(std::thread::id glThreadId, DispatchAsyncFunction dispatchAsyncFunction) : _glThreadId(glThreadId), _dispatchAsync(dispatchAsyncFunction) {
}

void GLThreadDispatch::runSync(void *host, std::function<void(void)> func) {
    if (std::this_thread::get_id() == _glThreadId) {
        func();
    } else {
        assert("not support run sync in gl thread now");
    }
}

void GLThreadDispatch::runAsync(void *host, std::function<void(void)> func) {
    if (_dispatchAsync) {
        _dispatchAsync(host, func);
    }
}

