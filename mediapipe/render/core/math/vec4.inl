//
//  vec4.inl
//  BdiEngine
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#include "vec4.hpp"

namespace Opipe {
    
    inline Vec4 Vec4::operator+(const Vec4& v) const
    {
    Vec4 result(*this);
    result.add(v);
    return result;
    }
    
    inline Vec4& Vec4::operator+=(const Vec4& v)
    {
    add(v);
    return *this;
    }
    
    inline Vec4 Vec4::operator-(const Vec4& v) const
    {
    Vec4 result(*this);
    result.subtract(v);
    return result;
    }
    
    inline Vec4& Vec4::operator-=(const Vec4& v)
    {
    subtract(v);
    return *this;
    }
    
    inline Vec4 Vec4::operator-() const
    {
    Vec4 result(*this);
    result.negate();
    return result;
    }
    
    inline Vec4 Vec4::operator*(float s) const
    {
    Vec4 result(*this);
    result.scale(s);
    return result;
    }
    
    inline Vec4& Vec4::operator*=(float s)
    {
    scale(s);
    return *this;
    }
    
    inline Vec4 Vec4::operator/(const float s) const
    {
    return Vec4(this->x / s, this->y / s, this->z / s, this->w / s);
    }
    
    inline bool Vec4::operator<(const Vec4& v) const
    {
    if (x == v.x) {
        if (y == v.y) {
            if (z == v.z) {
                if (w < v.w) {
                    return w < v.w;
                }
            }
            return z < v.z;
        }
        return y < v.y;
    }
    return x < v.x;
    }
    
    inline bool Vec4::operator==(const Vec4& v) const
    {
    return x == v.x && y == v.y && z == v.z && w == v.w;
    }
    
    inline bool Vec4::operator!=(const Vec4& v) const
    {
    return x != v.x || y != v.y || z != v.z || w != v.w;
    }
    
    inline Vec4 operator*(float x, const Vec4& v)
    {
    Vec4 result(v);
    result.scale(x);
    return result;
    }
    
}
