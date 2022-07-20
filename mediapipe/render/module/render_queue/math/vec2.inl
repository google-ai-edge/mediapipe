//
//  vec2.inl
//  BdiEngine
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#include "vec2.hpp"

namespace Opipe {
    
    inline Vec2::Vec2() : x(0.0f), y(0.0f)
    {
    }
    
    inline Vec2::Vec2(float xx, float yy) : x(xx), y(yy)
    {
    }
    
    inline Vec2::Vec2(const float* array)
    {
    set(array);
    }
    
    inline Vec2::Vec2(const Vec2& p1, const Vec2& p2)
    {
    set(p1, p2);
    }
    
    inline Vec2::Vec2(const Vec2& copy)
    {
    set(copy);
    }
    
    inline bool Vec2::is_zero() const
    {
    return x == 0.0f && y == 0.0f;
    }
    
    bool Vec2::is_one() const
    {
    return x == 1.0f && y == 1.0f;
    }
    
    inline void Vec2::add(const Vec2& v)
    {
    x += v.x;
    y += v.y;
    }
    
    inline float Vec2::distance_squared(const Vec2& v) const
    {
    float dx = v.x - x;
    float dy = v.y - y;
    return (dx * dx + dy * dy);
    }
    
    inline float Vec2::dot(const Vec2& v) const
    {
    return (x * v.x + y * v.y);
    }
    
    inline float Vec2::length_squared() const
    {
    return (x * x + y * y);
    }
    
    inline void Vec2::negate()
    {
    x = -x;
    y = -y;
    }
    
    inline void Vec2::scale(float scalar)
    {
    x *= scalar;
    y *= scalar;
    }
    
    inline void Vec2::scale(const Vec2& scale)
    {
    x *= scale.x;
    y *= scale.y;
    }
    
    inline void Vec2::set(float xx, float yy)
    {
    this->x = xx;
    this->y = yy;
    }
    
    inline void Vec2::set(const Vec2& v)
    {
    this->x = v.x;
    this->y = v.y;
    }
    
    inline void Vec2::set(const Vec2& p1, const Vec2& p2)
    {
    x = p2.x - p1.x;
    y = p2.y - p1.y;
    }
    
    void Vec2::set_zero()
    {
    x = y = 0.0f;
    }
    
    inline void Vec2::subtract(const Vec2& v)
    {
    x -= v.x;
    y -= v.y;
    }
    
    inline void Vec2::smooth(const Vec2& target, float elapsedTime, float responseTime)
    {
    if (elapsedTime > 0)
        {
        *this += (target - *this) * (elapsedTime / (elapsedTime + responseTime));
        }
    }
    
    inline Vec2 Vec2::operator+(const Vec2& v) const
    {
    Vec2 result(*this);
    result.add(v);
    return result;
    }
    
    inline Vec2& Vec2::operator+=(const Vec2& v)
    {
    add(v);
    return *this;
    }
    
    inline Vec2 Vec2::operator-(const Vec2& v) const
    {
    Vec2 result(*this);
    result.subtract(v);
    return result;
    }
    
    inline Vec2& Vec2::operator-=(const Vec2& v)
    {
    subtract(v);
    return *this;
    }
    
    inline Vec2 Vec2::operator-() const
    {
    Vec2 result(*this);
    result.negate();
    return result;
    }
    
    inline Vec2 Vec2::operator*(float s) const
    {
    Vec2 result(*this);
    result.scale(s);
    return result;
    }
    
    inline Vec2& Vec2::operator*=(float s)
    {
    scale(s);
    return *this;
    }
    
    inline Vec2 Vec2::operator/(const float s) const
    {
    return Vec2(this->x / s, this->y / s);
    }
    
    inline bool Vec2::operator<(const Vec2& v) const
    {
    if (x == v.x)
        {
        return y < v.y;
        }
    return x < v.x;
    }
    
    inline bool Vec2::operator>(const Vec2& v) const
    {
    if (x == v.x)
        {
        return y > v.y;
        }
    return x > v.x;
    }
    
    inline bool Vec2::operator==(const Vec2& v) const
    {
    return x==v.x && y==v.y;
    }
    
    inline bool Vec2::operator!=(const Vec2& v) const
    {
    return x!=v.x || y!=v.y;
    }
    
    inline Vec2 operator*(float x, const Vec2& v)
    {
    Vec2 result(v);
    result.scale(x);
    return result;
    }
    
    void Vec2::set_point(float xx, float yy)
    {
    this->x = xx;
    this->y = yy;
    }
    
}
