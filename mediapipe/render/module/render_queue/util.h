

#ifndef util_h
#define util_h

#include <stdlib.h>
#include <string>
#include "macros.h"

NS_OLA_BEGIN

    std::string str_format(const char *fmt,...);
    void Log(const std::string& tag, const std::string& format, ...);

#define rotationSwapsSize(rotation) ((rotation) == OLARender::RotateLeft || (rotation) == OLARender::RotateRight || (rotation) == OLARender::RotateRightFlipVertical || (rotation) == OLARender::RotateRightFlipHorizontal)

NS_OLA_END

#endif /* util_h */
