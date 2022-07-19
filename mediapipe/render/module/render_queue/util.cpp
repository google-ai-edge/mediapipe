//
//  OlaRender.h
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/17.
//
#include "util.h"

#if PLATFORM == PLATFORM_ANDROID
#include <android/log.h>
#endif


NS_OLA_BEGIN

    std::string str_format(const char *fmt,...) {
        std::string strResult="";
        if (NULL != fmt)
        {
            va_list marker;
            va_start(marker, fmt);
            char *buf = 0;
            int result = vasprintf (&buf, fmt, marker);
            if (!buf)
            {
                va_end(marker);
                return strResult;
            }

            if (result < 0)
            {
                free (buf);
                va_end(marker);
                return strResult;
            }

            result = (int)strlen (buf);
            strResult.append(buf,result);
            free(buf);
            va_end(marker);
        }
        return strResult;
    }

    void Log(const std::string& tag, const std::string& format, ...)
    {
        char buffer[10240];
        va_list args;
        va_start(args, format);
        vsprintf(buffer, format.c_str(), args);
        va_end(args);
#if PLATFORM == PLATFORM_ANDROID
        __android_log_print(ANDROID_LOG_INFO, tag.c_str(), "%s", buffer);
#endif
        
    }

NS_OLA_END
