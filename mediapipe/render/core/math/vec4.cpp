//
//  vec4.cpp
//  Opipe
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#include "vec4.hpp"
#include "math_utils.hpp"
#include <cmath>

namespace Opipe {
    
    Vec4::Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f)
    {
    }
    
    Vec4::Vec4(float xx, float yy, float zz, float ww) : x(xx), y(yy), z(zz), w(ww)
    {
    }
    
    Vec4::Vec4(const float* src)
    {
    set(src);
    }
    
    Vec4::Vec4(const Vec4& p1, const Vec4& p2)
    {
    set(p1, p2);
    }
    
    Vec4::Vec4(const Vec4& copy)
    {
    set(copy);
    }
    
    Vec4 Vec4::from_color(unsigned int color)
    {
    float components[4];
    int componentIndex = 0;
    for (int i = 3; i >= 0; --i)
        {
        int component = (color >> i*8) & 0x000000ff;
        
        components[componentIndex++] = static_cast<float>(component) / 255.0f;
        }
    
    Vec4 value(components);
    return value;
    }
    
    bool Vec4::is_zero() const
    {
    return x == 0.0f && y == 0.0f && z == 0.0f && w == 0.0f;
    }
    
    bool Vec4::is_one() const
    {
    return x == 1.0f && y == 1.0f && z == 1.0f && w == 1.0f;
    }
    
    float Vec4::angle(const Vec4& v1, const Vec4& v2)
    {
    float dx = v1.w * v2.x - v1.x * v2.w - v1.y * v2.z + v1.z * v2.y;
    float dy = v1.w * v2.y - v1.y * v2.w - v1.z * v2.x + v1.x * v2.z;
    float dz = v1.w * v2.z - v1.z * v2.w - v1.x * v2.y + v1.y * v2.x;
    
    return std::atan2(std::sqrt(dx * dx + dy * dy + dz * dz) + MATH_FLOAT_SMALL, dot(v1, v2));
    }
    
    void Vec4::add(const Vec4& v)
    {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    }
    
    void Vec4::add(const Vec4& v1, const Vec4& v2, Vec4* dst)
    {
    if (dst) {
        dst->x = v1.x + v2.x;
        dst->y = v1.y + v2.y;
        dst->z = v1.z + v2.z;
        dst->w = v1.w + v2.w;
    }
    }
    
    void Vec4::clamp(const Vec4& min, const Vec4& max)
    {
    // Clamp the x value.
    if (x < min.x)
        x = min.x;
    if (x > max.x)
        x = max.x;
    
    // Clamp the y value.
    if (y < min.y)
        y = min.y;
    if (y > max.y)
        y = max.y;
    
    // Clamp the z value.
    if (z < min.z)
        z = min.z;
    if (z > max.z)
        z = max.z;
    
    // Clamp the z value.
    if (w < min.w)
        w = min.w;
    if (w > max.w)
        w = max.w;
    }
    
    void Vec4::clamp(const Vec4& v, const Vec4& min, const Vec4& max, Vec4* dst)
    {
    if (dst) {
        // Clamp the x value.
        dst->x = v.x;
        if (dst->x < min.x)
            dst->x = min.x;
        if (dst->x > max.x)
            dst->x = max.x;
        
        // Clamp the y value.
        dst->y = v.y;
        if (dst->y < min.y)
            dst->y = min.y;
        if (dst->y > max.y)
            dst->y = max.y;
        
        // Clamp the z value.
        dst->z = v.z;
        if (dst->z < min.z)
            dst->z = min.z;
        if (dst->z > max.z)
            dst->z = max.z;
        
        // Clamp the w value.
        dst->w = v.w;
        if (dst->w < min.w)
            dst->w = min.w;
        if (dst->w > max.w)
            dst->w = max.w;
    }
    }
    
    float Vec4::distance(const Vec4& v) const
    {
    float dx = v.x - x;
    float dy = v.y - y;
    float dz = v.z - z;
    float dw = v.w - w;
    
    return std::sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
    }
    
    float Vec4::distance_squared(const Vec4& v) const
    {
    float dx = v.x - x;
    float dy = v.y - y;
    float dz = v.z - z;
    float dw = v.w - w;
    
    return (dx * dx + dy * dy + dz * dz + dw * dw);
    }
    
    float Vec4::dot(const Vec4& v) const
    {
    return (x * v.x + y * v.y + z * v.z + w * v.w);
    }
    
    float Vec4::dot(const Vec4& v1, const Vec4& v2)
    {
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w);
    }
    
    float Vec4::length() const
    {
    return std::sqrt(x * x + y * y + z * z + w * w);
    }
    
    
    float Vec4::length_squared() const
    {
    return (x * x + y * y + z * z + w * w);
    }
    
    void Vec4::negate()
    {
    x = -x;
    y = -y;
    z = -z;
    w = -w;
    }
    
    void Vec4::normalize()
    {
    float n = x * x + y * y + z * z + w * w;
    // Already normalized.
    if (n == 1.0f)
        return;
    
    n = std::sqrt(n);
    // Too close to zero.
    if (n < MATH_TOLERANCE)
        return;
    
    n = 1.0f / n;
    x *= n;
    y *= n;
    z *= n;
    w *= n;
    }
    
    Vec4 Vec4::get_normalized() const
    {
    Vec4 v(*this);
    v.normalize();
    return v;
    }
    
    void Vec4::scale(float scalar)
    {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    w *= scalar;
    }
    
    void Vec4::set(float xx, float yy, float zz, float ww)
    {
    this->x = xx;
    this->y = yy;
    this->z = zz;
    this->w = ww;
    }
    
    void Vec4::set(const float* array)
    {
    if (array) {
        x = array[0];
        y = array[1];
        z = array[2];
        w = array[3];
    }
    }
    
    void Vec4::set(const Vec4& v)
    {
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->w = v.w;
    }
    
    void Vec4::set(const Vec4& p1, const Vec4& p2)
    {
    x = p2.x - p1.x;
    y = p2.y - p1.y;
    z = p2.z - p1.z;
    w = p2.w - p1.w;
    }
    
    void Vec4::subtract(const Vec4& v)
    {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    }
    
    void Vec4::subtract(const Vec4& v1, const Vec4& v2, Vec4* dst)
    {
    if (dst) {
        dst->x = v1.x - v2.x;
        dst->y = v1.y - v2.y;
        dst->z = v1.z - v2.z;
        dst->w = v1.w - v2.w;
    }
    }
    
    const Vec4 Vec4::ZERO = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
    const Vec4 Vec4::ONE = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
    const Vec4 Vec4::UNIT_X = Vec4(1.0f, 0.0f, 0.0f, 0.0f);
    const Vec4 Vec4::UNIT_Y = Vec4(0.0f, 1.0f, 0.0f, 0.0f);
    const Vec4 Vec4::UNIT_Z = Vec4(0.0f, 0.0f, 1.0f, 0.0f);
    const Vec4 Vec4::UNIT_W = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    
}
