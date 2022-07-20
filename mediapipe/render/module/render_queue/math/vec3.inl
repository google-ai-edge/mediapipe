//
//  vec3.inl
//  BdiEngine
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#include "vec3.hpp"
#include <cmath>

namespace Opipe {
    
    inline bool Vec3::is_zero() const
    {
    return x == 0.0f && y == 0.0f && z == 0.0f;
    }
    
    inline bool Vec3::is_one() const
    {
    return x == 1.0f && y == 1.0f && z == 1.0f;
    }
    
    inline void Vec3::add(const Vec3& v)
    {
    x += v.x;
    y += v.y;
    z += v.z;
    }
    
    inline void Vec3::add(float xx, float yy, float zz)
    {
    x += xx;
    y += yy;
    z += zz;
    }
    
    inline float Vec3::length() const
    {
    return std::sqrt(x * x + y * y + z * z);
    }
    
    inline float Vec3::length_squared() const
    {
    return (x * x + y * y + z * z);
    }
    
    inline void Vec3::negate()
    {
    x = -x;
    y = -y;
    z = -z;
    }
    
    inline void Vec3::scale(float scalar)
    {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    }
    
    inline Vec3 Vec3::lerp(const Vec3 &target, float alpha) const
    {
    return *this * (1.f - alpha) + target * alpha;
    }
    
    inline void Vec3::set(float xx, float yy, float zz)
    {
    this->x = xx;
    this->y = yy;
    this->z = zz;
    }
    
    inline void Vec3::set(const float* array)
    {
    if (array) {
        x = array[0];
        y = array[1];
        z = array[2];
    }
    }
    
    inline void Vec3::set(const Vec3& v)
    {
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    }
    
    inline void Vec3::set(const Vec3& p1, const Vec3& p2)
    {
    x = p2.x - p1.x;
    y = p2.y - p1.y;
    z = p2.z - p1.z;
    }
    
    inline void Vec3::set_zero()
    {
    x = y = z = 0.0f;
    }
    
    inline void Vec3::subtract(const Vec3& v)
    {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    }
    
    inline Vec3 Vec3::operator+(const Vec3& v) const
    {
    Vec3 result(*this);
    result.add(v);
    return result;
    }
    
    inline Vec3& Vec3::operator+=(const Vec3& v)
    {
    add(v);
    return *this;
    }
    
    inline Vec3 Vec3::operator-(const Vec3& v) const
    {
    Vec3 result(*this);
    result.subtract(v);
    return result;
    }
    
    inline Vec3& Vec3::operator-=(const Vec3& v)
    {
    subtract(v);
    return *this;
    }
    
    inline Vec3 Vec3::operator-() const
    {
    Vec3 result(*this);
    result.negate();
    return result;
    }
    
    inline Vec3 Vec3::operator*(float s) const
    {
    Vec3 result(*this);
    result.scale(s);
    return result;
    }
    
    inline Vec3& Vec3::operator*=(float s)
    {
    scale(s);
    return *this;
    }
    
    inline Vec3 Vec3::operator/(const float s) const
    {
    return Vec3(this->x / s, this->y / s, this->z / s);
    }
    
    inline bool Vec3::operator==(const Vec3& v) const
    {
    return x==v.x && y==v.y && z==v.z;
    }
    
    inline bool Vec3::operator!=(const Vec3& v) const
    {
    return x!=v.x || y!=v.y || z!=v.z;
    }
    
    inline Vec3 operator*(float x, const Vec3& v)
    {
    Vec3 result(v);
    result.scale(x);
    return result;
    }
    
}
