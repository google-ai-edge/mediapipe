//
//  vec2.cpp
//  Opipe
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#include "vec2.hpp"
#include "math_utils.hpp"
namespace Opipe {
    
    float Vec2::angle(const Vec2& v1, const Vec2& v2)
    {
    float dz = v1.x * v2.y - v1.y * v2.x;
    return atan2f(fabsf(dz) + MATH_FLOAT_SMALL, dot(v1, v2));
    }
    
    void Vec2::add(const Vec2& v1, const Vec2& v2, Vec2* dst)
    {
    if (dst) {
        dst->x = v1.x + v2.x;
        dst->y = v1.y + v2.y;
    }
    }
    
    void Vec2::clamp(const Vec2& min, const Vec2& max)
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
    }
    
    void Vec2::clamp(const Vec2& v, const Vec2& min, const Vec2& max, Vec2* dst)
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
    }
    }
    
    float Vec2::distance(const Vec2& v) const
    {
    float dx = v.x - x;
    float dy = v.y - y;
    
    return std::sqrt(dx * dx + dy * dy);
    }
    
    float Vec2::dot(const Vec2& v1, const Vec2& v2)
    {
    return (v1.x * v2.x + v1.y * v2.y);
    }
    
    float Vec2::length() const
    {
    return std::sqrt(x * x + y * y);
    }
    
    void Vec2::normalize()
    {
    float n = x * x + y * y;
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
    }
    
    Vec2 Vec2::get_normalized() const
    {
    Vec2 v(*this);
    v.normalize();
    return v;
    }
    
    void Vec2::rotate(const Vec2& point, float angle)
    {
    float sinAngle = std::sin(angle);
    float cosAngle = std::cos(angle);
    
    if (point.is_zero())
        {
        float tempX = x * cosAngle - y * sinAngle;
        y = y * cosAngle + x * sinAngle;
        x = tempX;
        }
    else
        {
        float tempX = x - point.x;
        float tempY = y - point.y;
        
        x = tempX * cosAngle - tempY * sinAngle + point.x;
        y = tempY * cosAngle + tempX * sinAngle + point.y;
        }
    }
    
    void Vec2::set(const float* array)
    {
    if (array) {
        x = array[0];
        y = array[1];
    }
    }
    
    void Vec2::subtract(const Vec2& v1, const Vec2& v2, Vec2* dst)
    {
    if (dst) {
        dst->x = v1.x - v2.x;
        dst->y = v1.y - v2.y;
    }
    }
    
    bool Vec2::equals(const Vec2& target) const
    {
    return (std::abs(this->x - target.x) < MATH_EPSILON)
    && (std::abs(this->y - target.y) < MATH_EPSILON);
    }
    
    float Vec2::get_angle(const Vec2& other) const
    {
    Vec2 a2 = get_normalized();
    Vec2 b2 = other.get_normalized();
    float angle = atan2f(a2.cross(b2), a2.dot(b2));
    if (std::abs(angle) < MATH_EPSILON) return 0.f;
    return angle;
    }
    
    Vec2 Vec2::rotate_by_angle(const Vec2& pivot, float angle) const
    {
    return pivot + (*this - pivot).rotate(Vec2::for_angle(angle));
    }
    
    const Vec2 Vec2::ZERO(0.0f, 0.0f);
    const Vec2 Vec2::ONE(1.0f, 1.0f);
    const Vec2 Vec2::UNIT_X(1.0f, 0.0f);
    const Vec2 Vec2::UNIT_Y(0.0f, 1.0f);
    const Vec2 Vec2::ANCHOR_MIDDLE(0.5f, 0.5f);
    const Vec2 Vec2::ANCHOR_BOTTOM_LEFT(0.0f, 0.0f);
    const Vec2 Vec2::ANCHOR_TOP_LEFT(0.0f, 1.0f);
    const Vec2 Vec2::ANCHOR_BOTTOM_RIGHT(1.0f, 0.0f);
    const Vec2 Vec2::ANCHOR_TOP_RIGHT(1.0f, 1.0f);
    const Vec2 Vec2::ANCHOR_MIDDLE_RIGHT(1.0f, 0.5f);
    const Vec2 Vec2::ANCHOR_MIDDLE_LEFT(0.0f, 0.5f);
    const Vec2 Vec2::ANCHOR_MIDDLE_TOP(0.5f, 1.0f);
    const Vec2 Vec2::ANCHOR_MIDDLE_BOTTOM(0.5f, 0.0f);
    
}
