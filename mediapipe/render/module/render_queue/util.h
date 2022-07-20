

#ifndef util_h
#define util_h

#include <stdlib.h>
#include <string>
#include "macros.h"

NS_OLA_BEGIN

    std::string str_format(const char *fmt,...);
    void Log(const std::string& tag, const std::string& format, ...);

#define rotationSwapsSize(rotation) ((rotation) == Opipe::RotateLeft || (rotation) == Opipe::RotateRight || (rotation) == Opipe::RotateRightFlipVertical || (rotation) == Opipe::RotateRightFlipHorizontal)

NS_OLA_END

#endif /* util_h */
