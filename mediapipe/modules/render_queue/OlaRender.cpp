//
//  OlaRender.cpp
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/17.
//

#include "OlaRender.h"
#if USE_OLARENDER
#include <Context.hpp>
#include <OlaDispatch.hpp>
#endif
#include "OlaRenderIMP.h"



NS_OLA_BEGIN

    OlaRender::~OlaRender() {

    }

    OlaRender::OlaRender() {

    }

#if USE_OLARENDER
    OlaRender* OlaRender::create(void *env, void *context) {
    return nullptr;
}
#endif

    OlaRender* OlaRender::create() {
        OLARenderIMP *instance = new OLARenderIMP();
        return instance;
    }


NS_OLA_END
