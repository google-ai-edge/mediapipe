#include "OlaContext.hpp"
#include "Context.hpp"

namespace Opipe {

   
    OlaContext::OlaContext() {
        _currentContext = new Context();
    }

    OlaContext::~OlaContext() {

    }

    #if defined(__APPLE__)
     OlaContext::OlaContext(EAGLContext *context) {
        _currentContext = new Context(context);
    }


    EAGLContext* OlaContext::currentContext() {
        return _currentContext->getEglContext();
    }
    
    #else
    #endif

    Context* OlaContext::glContext() {
        return _currentContext;
    }
}