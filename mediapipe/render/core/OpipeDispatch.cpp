//
//  OpipeDispatch.cpp
//  Quaramera
//
//  Created by wangrenzhu2021 on 2021/12/14.
//  Copyright © 2021 ola. All rights reserved.
//

#define LOG_TAG "OpipeDispatch"


#include "dispatch_queue.h"

#include "OpipeDispatch.hpp"


namespace Opipe {

    OpipeDispatch::OpipeDispatch(Opipe::Context *context, void *id, Opipe::GLThreadDispatch *glDispatch) : _context(context) {
        _id = id;
        _contextQueueIO = new dispatch_queue("quaramera_io");
        _contextQueueIO->dispatch_sync([&] {
            _context->useAsCurrent(Opipe::Context::IOContext, true);
        });
#if defined(__APPLE__)
        _contextQueue = new dispatch_queue("com.ola.glContextQueue");
        _contextQueue->dispatch_sync([&]
                                     {
        _context->useAsCurrent(Opipe::Context::GPUImageContext, true);
        });
#endif

        _contextQueueOffline = new dispatch_queue("quaramera_offline");
        _contextQueueOffline->dispatch_sync([&] {
            _context->useAsCurrent(Opipe::Context::OfflineRenderContext, true);
        });

        _glThreadDispatch = glDispatch;
    }

    OpipeDispatch::~OpipeDispatch() {
        delete _contextQueueIO;
        _contextQueueIO = 0;
#if defined(__APPLE__)

        delete _contextQueue;
        _contextQueue = 0;
#endif
        delete _contextQueueOffline;
        _contextQueueOffline = 0;

        delete _glThreadDispatch;
        _glThreadDispatch = 0;
    }

    void OpipeDispatch::flushSharedInstance() {
        this->runSync([&] {});
        this->runSync([&] {}, Opipe::Context::OfflineRenderContext);
        this->runSync([&] {}, Opipe::Context::IOContext);
    }

    void OpipeDispatch::runSync(std::function<void(void)> func, Opipe::Context::ContextType type /* = GPUImageContext*/) {
        if (type == Opipe::Context::IOContext) {
            if (_contextQueueIO->isCurrent()) {
                _context->useAsCurrent(type);
                func();
            } else {
                _contextQueueIO->dispatch_sync([=] {
                    _context->useAsCurrent(type);
                    func();
                });
            }
        } else if (type == Opipe::Context::OfflineRenderContext) {
            if (_contextQueueOffline->isCurrent()) {
                _context->useAsCurrent(type);
                func();
            } else {
                _contextQueueOffline->dispatch_sync([=] {
                    _context->useAsCurrent(type);
                    func();
                });
            }
        } else {


#if defined(__APPLE__)
            if (_contextQueue->isCurrent()) {
                _context->useAsCurrent(type);
                func();
            }
            else {
                _contextQueue->dispatch_sync([=]{
                    _context->useAsCurrent(type);
                    func();
                });
            }
#else
            //优先使用外部注入的实现
            if (_glThreadDispatch) {
                _glThreadDispatch->runSync(_id, func);
                return;
            } else {
                assert("not init gl_thread_dispatcher");
            }
#endif
        }
    }

    void OpipeDispatch::runAsync(std::function<void(void)> func, Opipe::Context::ContextType type /* = GPUImageContext*/, bool async/* = false*/) {
        if (type == Opipe::Context::IOContext) {
            if (!async && _contextQueueIO->isCurrent()) {
                _context->useAsCurrent(type);
                func();
            } else {
                _contextQueueIO->dispatch_async([=] {
                    _context->useAsCurrent(type);
                    func();
                });
            }
        } else if (type == Opipe::Context::OfflineRenderContext) {
            if (!async && _contextQueueOffline->isCurrent()) {
                _context->useAsCurrent(type);
                func();
            } else {
                _contextQueueOffline->dispatch_async([=] {
                    _context->useAsCurrent(type);
                    func();
                });
            }
        } else {
#if defined(__APPLE__)
            if (!async && _contextQueue->isCurrent())
                {
                _context->useAsCurrent(type);
                func();
                }
            else
                {
                _contextQueue->dispatch_async([=]{
                    _context->useAsCurrent(type);
                    func();
                });
                }
#else

            //优先使用外部注入的实现
            if (_glThreadDispatch) {
                _glThreadDispatch->runAsync(_id, func);
                return;
            } else {
                assert("not init gl_thread_dispatcher");
            }
#endif
        }
    }

}
