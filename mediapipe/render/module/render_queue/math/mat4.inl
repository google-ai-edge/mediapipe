//
//  mat4.inl
//  BdiEngine
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#include "mat4.hpp"

namespace Opipe {
    
    inline Mat4 Mat4::operator+(const Mat4& mat) const
    {
    Mat4 result(*this);
    result.add(mat);
    return result;
    }
    
    inline Mat4& Mat4::operator+=(const Mat4& mat)
    {
    add(mat);
    return *this;
    }
    
    inline Mat4 Mat4::operator-(const Mat4& mat) const
    {
    Mat4 result(*this);
    result.subtract(mat);
    return result;
    }
    
    inline Mat4& Mat4::operator-=(const Mat4& mat)
    {
    subtract(mat);
    return *this;
    }
    
    inline Mat4 Mat4::operator-() const
    {
    Mat4 mat(*this);
    mat.negate();
    return mat;
    }
    
    inline Mat4 Mat4::operator*(const Mat4& mat) const
    {
    Mat4 result(*this);
    result.multiply(mat);
    return result;
    }
    
    inline Mat4& Mat4::operator*=(const Mat4& mat)
    {
    multiply(mat);
    return *this;
    }
    
    inline Vec3& operator*=(Vec3& v, const Mat4& m)
    {
    m.transform_vector(&v);
    return v;
    }
    
    inline Vec3 operator*(const Mat4& m, const Vec3& v)
    {
    Vec3 x;
    m.transform_vector(v, &x);
    return x;
    }
    
    inline Vec4& operator*=(Vec4& v, const Mat4& m)
    {
    m.transform_vector(&v);
    return v;
    }
    
    inline Vec4 operator*(const Mat4& m, const Vec4& v)
    {
    Vec4 x;
    m.transform_vector(v, &x);
    return x;
    }
    
}
