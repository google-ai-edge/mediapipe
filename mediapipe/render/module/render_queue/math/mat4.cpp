//
//  Mat4.cpp
//  Opipe
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#if defined(__APPLE__)
#include "mat4.hpp"
#include "math_utils.hpp"
#else
#include "mat4.hpp"
#include "math_utils.hpp"
#endif
#include <cstring>

namespace Opipe {
    
    static const int MATRIX_SIZE = (sizeof(float) * 16);
    
    Mat4::Mat4()
    {
    *this = IDENTITY;
    }
    
    Mat4::Mat4(float m11, float m12, float m13, float m14, float m21, float m22, float m23, float m24,
               float m31, float m32, float m33, float m34, float m41, float m42, float m43, float m44)
    {
    set(m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44);
    }
    
    Mat4::Mat4(const float* mat)
    {
    set(mat);
    }
    
    Mat4::Mat4(const Mat4& copy)
    {
    memcpy(m, copy.m, MATRIX_SIZE);
    }
    
    void Mat4::create_look_at(const Vec3& eyePosition, const Vec3& targetPosition, const Vec3& up, Mat4* dst)
    {
    create_look_at(eyePosition.x, eyePosition.y, eyePosition.z,
                   targetPosition.x, targetPosition.y, targetPosition.z,
                   up.x, up.y, up.z, dst);
    }
    
    void Mat4::create_look_at(float eyePositionX, float eyePositionY, float eyePositionZ,
                              float targetPositionX, float targetPositionY, float targetPositionZ,
                              float upX, float upY, float upZ, Mat4* dst)
    {
    Vec3 eye(eyePositionX, eyePositionY, eyePositionZ);
    Vec3 target(targetPositionX, targetPositionY, targetPositionZ);
    Vec3 up(upX, upY, upZ);
    up.normalize();
    
    Vec3 zaxis;
    Vec3::subtract(eye, target, &zaxis);
    zaxis.normalize();
    
    Vec3 xaxis;
    Vec3::cross(up, zaxis, &xaxis);
    xaxis.normalize();
    
    Vec3 yaxis;
    Vec3::cross(zaxis, xaxis, &yaxis);
    yaxis.normalize();
    
    dst->m[0] = xaxis.x;
    dst->m[1] = yaxis.x;
    dst->m[2] = zaxis.x;
    dst->m[3] = 0.0f;
    
    dst->m[4] = xaxis.y;
    dst->m[5] = yaxis.y;
    dst->m[6] = zaxis.y;
    dst->m[7] = 0.0f;
    
    dst->m[8] = xaxis.z;
    dst->m[9] = yaxis.z;
    dst->m[10] = zaxis.z;
    dst->m[11] = 0.0f;
    
    dst->m[12] = -Vec3::dot(xaxis, eye);
    dst->m[13] = -Vec3::dot(yaxis, eye);
    dst->m[14] = -Vec3::dot(zaxis, eye);
    dst->m[15] = 1.0f;
    }
    
    void Mat4::create_perspective(float fieldOfView, float aspectRatio,
                                  float zNearPlane, float zFarPlane, Mat4* dst)
    {
    float f_n = 1.0f / (zFarPlane - zNearPlane);
    float theta = MATH_DEG_TO_RAD(fieldOfView) * 0.5f;
    if (std::abs(std::fmod(theta, MATH_PIOVER2)) < MATH_EPSILON)
        {
        return;
        }
    float divisor = std::tan(theta);
    float factor = 1.0f / divisor;
    
    memset(dst, 0, MATRIX_SIZE);
    
    dst->m[0] = (1.0f / aspectRatio) * factor;
    dst->m[5] = factor;
    dst->m[10] = (-(zFarPlane + zNearPlane)) * f_n;
    dst->m[11] = -1.0f;
    dst->m[14] = -2.0f * zFarPlane * zNearPlane * f_n;
    }
    
    void Mat4::create_orthographic(float width, float height, float zNearPlane, float zFarPlane, Mat4* dst)
    {
    float halfWidth = width / 2.0f;
    float halfHeight = height / 2.0f;
    create_orthographic_off_center(-halfWidth, halfWidth, -halfHeight, halfHeight, zNearPlane, zFarPlane, dst);
    }
    
    void Mat4::create_orthographic_off_center(float left, float right, float bottom, float top,
                                              float zNearPlane, float zFarPlane, Mat4* dst)
    {
    memset(dst, 0, MATRIX_SIZE);
    dst->m[0] = 2 / (right - left);
    dst->m[5] = 2 / (top - bottom);
    dst->m[10] = 2 / (zNearPlane - zFarPlane);
    
    dst->m[12] = (left + right) / (left - right);
    dst->m[13] = (top + bottom) / (bottom - top);
    dst->m[14] = (zNearPlane + zFarPlane) / (zNearPlane - zFarPlane);
    dst->m[15] = 1;
    }
    
    void Mat4::create_scale(const Vec3& scale, Mat4* dst)
    {
    memcpy(dst, &IDENTITY, MATRIX_SIZE);
    
    dst->m[0] = scale.x;
    dst->m[5] = scale.y;
    dst->m[10] = scale.z;
    }
    
    void Mat4::create_scale(float xScale, float yScale, float zScale, Mat4* dst)
    {
    memcpy(dst, &IDENTITY, MATRIX_SIZE);
    
    dst->m[0] = xScale;
    dst->m[5] = yScale;
    dst->m[10] = zScale;
    }
    
    void Mat4::create_rotation(const Vec3& axis, float angle, Mat4* dst)
    {
    float x = axis.x;
    float y = axis.y;
    float z = axis.z;
    
    // Make sure the input axis is normalized.
    float n = x*x + y*y + z*z;
    if (n != 1.0f)
        {
        // Not normalized.
        n = std::sqrt(n);
        // Prevent divide too close to zero.
        if (n > 0.000001f)
            {
            n = 1.0f / n;
            x *= n;
            y *= n;
            z *= n;
            }
        }
    
    float c = std::cos(angle);
    float s = std::sin(angle);
    
    float t = 1.0f - c;
    float tx = t * x;
    float ty = t * y;
    float tz = t * z;
    float txy = tx * y;
    float txz = tx * z;
    float tyz = ty * z;
    float sx = s * x;
    float sy = s * y;
    float sz = s * z;
    
    dst->m[0] = c + tx*x;
    dst->m[1] = txy + sz;
    dst->m[2] = txz - sy;
    dst->m[3] = 0.0f;
    
    dst->m[4] = txy - sz;
    dst->m[5] = c + ty*y;
    dst->m[6] = tyz + sx;
    dst->m[7] = 0.0f;
    
    dst->m[8] = txz + sy;
    dst->m[9] = tyz - sx;
    dst->m[10] = c + tz*z;
    dst->m[11] = 0.0f;
    
    dst->m[12] = 0.0f;
    dst->m[13] = 0.0f;
    dst->m[14] = 0.0f;
    dst->m[15] = 1.0f;
    }
    
    void Mat4::create_rotation_x(float angle, Mat4* dst)
    {
    memcpy(dst, &IDENTITY, MATRIX_SIZE);
    
    float c = std::cos(angle);
    float s = std::sin(angle);
    
    dst->m[5]  = c;
    dst->m[6]  = s;
    dst->m[9]  = -s;
    dst->m[10] = c;
    }
    
    void Mat4::create_rotation_y(float angle, Mat4* dst)
    {
    memcpy(dst, &IDENTITY, MATRIX_SIZE);
    
    float c = std::cos(angle);
    float s = std::sin(angle);
    
    dst->m[0]  = c;
    dst->m[2]  = -s;
    dst->m[8]  = s;
    dst->m[10] = c;
    }
    
    void Mat4::create_rotation_z(float angle, Mat4* dst)
    {
    memcpy(dst, &IDENTITY, MATRIX_SIZE);
    
    float c = std::cos(angle);
    float s = std::sin(angle);
    
    dst->m[0] = c;
    dst->m[1] = s;
    dst->m[4] = -s;
    dst->m[5] = c;
    }
    
    void Mat4::create_translation(const Vec3& translation, Mat4* dst)
    {
    memcpy(dst, &IDENTITY, MATRIX_SIZE);
    
    dst->m[12] = translation.x;
    dst->m[13] = translation.y;
    dst->m[14] = translation.z;
    }
    
    void Mat4::create_translation(float xTranslation, float yTranslation, float zTranslation, Mat4* dst)
    {
    memcpy(dst, &IDENTITY, MATRIX_SIZE);
    
    dst->m[12] = xTranslation;
    dst->m[13] = yTranslation;
    dst->m[14] = zTranslation;
    }
    
    void Mat4::add(float scalar)
    {
    add(scalar, this);
    }
    
    void Mat4::add(float scalar, Mat4* dst)
    {
    MathUtils::add_matrix(m, scalar, dst->m);
    }
    
    void Mat4::add(const Mat4& mat)
    {
    add(*this, mat, this);
    }
    
    void Mat4::add(const Mat4& m1, const Mat4& m2, Mat4* dst)
    {
    MathUtils::add_matrix(m1.m, m2.m, dst->m);
    }
    
    float Mat4::determinant() const
    {
    float a0 = m[0] * m[5] - m[1] * m[4];
    float a1 = m[0] * m[6] - m[2] * m[4];
    float a2 = m[0] * m[7] - m[3] * m[4];
    float a3 = m[1] * m[6] - m[2] * m[5];
    float a4 = m[1] * m[7] - m[3] * m[5];
    float a5 = m[2] * m[7] - m[3] * m[6];
    float b0 = m[8] * m[13] - m[9] * m[12];
    float b1 = m[8] * m[14] - m[10] * m[12];
    float b2 = m[8] * m[15] - m[11] * m[12];
    float b3 = m[9] * m[14] - m[10] * m[13];
    float b4 = m[9] * m[15] - m[11] * m[13];
    float b5 = m[10] * m[15] - m[11] * m[14];
    
    // Calculate the determinant.
    return (a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0);
    }
    
    void Mat4::get_up_vector(Vec3* dst) const
    {
    dst->x = m[4];
    dst->y = m[5];
    dst->z = m[6];
    }
    
    void Mat4::get_down_vector(Vec3* dst) const
    {
    dst->x = -m[4];
    dst->y = -m[5];
    dst->z = -m[6];
    }
    
    void Mat4::get_left_vector(Vec3* dst) const
    {
    dst->x = -m[0];
    dst->y = -m[1];
    dst->z = -m[2];
    }
    
    void Mat4::get_right_vector(Vec3* dst) const
    {
    dst->x = m[0];
    dst->y = m[1];
    dst->z = m[2];
    }
    
    void Mat4::get_forward_vector(Vec3* dst) const
    {
    dst->x = -m[8];
    dst->y = -m[9];
    dst->z = -m[10];
    }
    
    void Mat4::get_back_vector(Vec3* dst) const
    {
    dst->x = m[8];
    dst->y = m[9];
    dst->z = m[10];
    }
    
    Mat4 Mat4::get_inversed() const
    {
    Mat4 mat(*this);
    mat.inverse();
    return mat;
    }
    
    bool Mat4::inverse()
    {
    float a0 = m[0] * m[5] - m[1] * m[4];
    float a1 = m[0] * m[6] - m[2] * m[4];
    float a2 = m[0] * m[7] - m[3] * m[4];
    float a3 = m[1] * m[6] - m[2] * m[5];
    float a4 = m[1] * m[7] - m[3] * m[5];
    float a5 = m[2] * m[7] - m[3] * m[6];
    float b0 = m[8] * m[13] - m[9] * m[12];
    float b1 = m[8] * m[14] - m[10] * m[12];
    float b2 = m[8] * m[15] - m[11] * m[12];
    float b3 = m[9] * m[14] - m[10] * m[13];
    float b4 = m[9] * m[15] - m[11] * m[13];
    float b5 = m[10] * m[15] - m[11] * m[14];
    
    // Calculate the determinant.
    float det = a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;
    
    // Close to zero, can't invert.
    if (std::abs(det) <= MATH_TOLERANCE)
        return false;
    
    // Support the case where m == dst.
    Mat4 inverse;
    inverse.m[0]  = m[5] * b5 - m[6] * b4 + m[7] * b3;
    inverse.m[1]  = -m[1] * b5 + m[2] * b4 - m[3] * b3;
    inverse.m[2]  = m[13] * a5 - m[14] * a4 + m[15] * a3;
    inverse.m[3]  = -m[9] * a5 + m[10] * a4 - m[11] * a3;
    
    inverse.m[4]  = -m[4] * b5 + m[6] * b2 - m[7] * b1;
    inverse.m[5]  = m[0] * b5 - m[2] * b2 + m[3] * b1;
    inverse.m[6]  = -m[12] * a5 + m[14] * a2 - m[15] * a1;
    inverse.m[7]  = m[8] * a5 - m[10] * a2 + m[11] * a1;
    
    inverse.m[8]  = m[4] * b4 - m[5] * b2 + m[7] * b0;
    inverse.m[9]  = -m[0] * b4 + m[1] * b2 - m[3] * b0;
    inverse.m[10] = m[12] * a4 - m[13] * a2 + m[15] * a0;
    inverse.m[11] = -m[8] * a4 + m[9] * a2 - m[11] * a0;
    
    inverse.m[12] = -m[4] * b3 + m[5] * b1 - m[6] * b0;
    inverse.m[13] = m[0] * b3 - m[1] * b1 + m[2] * b0;
    inverse.m[14] = -m[12] * a3 + m[13] * a1 - m[14] * a0;
    inverse.m[15] = m[8] * a3 - m[9] * a1 + m[10] * a0;
    
    multiply(inverse, 1.0f / det, this);
    
    return true;
    }
    
    bool Mat4::is_identity() const
    {
    return (memcmp(m, &IDENTITY, MATRIX_SIZE) == 0);
    }
    
    void Mat4::multiply(float scalar)
    {
    multiply(scalar, this);
    }
    
    void Mat4::multiply(float scalar, Mat4* dst) const
    {
    multiply(*this, scalar, dst);
    }
    
    void Mat4::multiply(const Mat4& m, float scalar, Mat4* dst)
    {
    MathUtils::multiply_matrix(m.m, scalar, dst->m);
    }
    
    void Mat4::multiply(const Mat4& mat)
    {
    multiply(*this, mat, this);
    }
    
    void Mat4::multiply(const Mat4& m1, const Mat4& m2, Mat4* dst)
    {
    MathUtils::multiply_matrix(m1.m, m2.m, dst->m);
    }
    
    void Mat4::negate()
    {
    MathUtils::negate_matrix(m, m);
    }
    
    Mat4 Mat4::get_negated() const
    {
    Mat4 mat(*this);
    mat.negate();
    return mat;
    }
    
    void Mat4::rotate(const Vec3& axis, float angle)
    {
    rotate(axis, angle, this);
    }
    
    void Mat4::rotate(const Vec3& axis, float angle, Mat4* dst) const
    {
    Mat4 r;
    create_rotation(axis, angle, &r);
    multiply(*this, r, dst);
    }
    
    void Mat4::rotate_x(float angle)
    {
    rotate_x(angle, this);
    }
    
    void Mat4::rotate_x(float angle, Mat4* dst) const
    {
    Mat4 r;
    create_rotation_x(angle, &r);
    multiply(*this, r, dst);
    }
    
    void Mat4::rotate_y(float angle)
    {
    rotate_y(angle, this);
    }
    
    void Mat4::rotate_y(float angle, Mat4* dst) const
    {
    Mat4 r;
    create_rotation_y(angle, &r);
    multiply(*this, r, dst);
    }
    
    void Mat4::rotate_z(float angle)
    {
    rotate_z(angle, this);
    }
    
    void Mat4::rotate_z(float angle, Mat4* dst) const
    {
    Mat4 r;
    create_rotation_z(angle, &r);
    multiply(*this, r, dst);
    }
    
    void Mat4::scale(float value)
    {
    scale(value, this);
    }
    
    void Mat4::scale(float value, Mat4* dst) const
    {
    scale(value, value, value, dst);
    }
    
    void Mat4::scale(float xScale, float yScale, float zScale)
    {
    scale(xScale, yScale, zScale, this);
    }
    
    void Mat4::scale(float xScale, float yScale, float zScale, Mat4* dst) const
    {
    Mat4 s;
    create_scale(xScale, yScale, zScale, &s);
    multiply(*this, s, dst);
    }
    
    void Mat4::scale(const Vec3& s)
    {
    scale(s.x, s.y, s.z, this);
    }
    
    void Mat4::scale(const Vec3& s, Mat4* dst) const
    {
    scale(s.x, s.y, s.z, dst);
    }
    
    void Mat4::set(float m11, float m12, float m13, float m14, float m21, float m22, float m23, float m24,
                   float m31, float m32, float m33, float m34, float m41, float m42, float m43, float m44)
    {
    m[0]  = m11;
    m[1]  = m21;
    m[2]  = m31;
    m[3]  = m41;
    m[4]  = m12;
    m[5]  = m22;
    m[6]  = m32;
    m[7]  = m42;
    m[8]  = m13;
    m[9]  = m23;
    m[10] = m33;
    m[11] = m43;
    m[12] = m14;
    m[13] = m24;
    m[14] = m34;
    m[15] = m44;
    }
    
    void Mat4::set(const float* mat)
    {
    memcpy(this->m, mat, MATRIX_SIZE);
    }
    
    void Mat4::set(const Mat4& mat)
    {
    memcpy(this->m, mat.m, MATRIX_SIZE);
    }
    
    void Mat4::set_identity()
    {
    memcpy(m, &IDENTITY, MATRIX_SIZE);
    }
    
    void Mat4::set_zero()
    {
    memset(m, 0, MATRIX_SIZE);
    }
    
    void Mat4::subtract(const Mat4& mat)
    {
    subtract(*this, mat, this);
    }
    
    void Mat4::subtract(const Mat4& m1, const Mat4& m2, Mat4* dst)
    {
    MathUtils::subtract_matrix(m1.m, m2.m, dst->m);
    }
    
    void Mat4::transform_vector(Vec3* vector) const
    {
    transform_vector(vector->x, vector->y, vector->z, 0.0f, vector);
    }
    
    void Mat4::transform_vector(const Vec3& vector, Vec3* dst) const
    {
    transform_vector(vector.x, vector.y, vector.z, 0.0f, dst);
    }
    
    void Mat4::transform_vector(float x, float y, float z, float w, Vec3* dst) const
    {
    MathUtils::transform_vec4(m, x, y, z, w, (float*)dst);
    }
    
    void Mat4::transform_vector(Vec4* vector) const
    {
    transform_vector(*vector, vector);
    }
    
    void Mat4::transform_vector(const Vec4& vector, Vec4* dst) const
    {
    MathUtils::transform_vec4(m, (const float*) &vector, (float*)dst);
    }
    
    void Mat4::translate(float x, float y, float z)
    {
    translate(x, y, z, this);
    }
    
    void Mat4::translate(float x, float y, float z, Mat4* dst) const
    {
    Mat4 t;
    create_translation(x, y, z, &t);
    multiply(*this, t, dst);
    }
    
    void Mat4::translate(const Vec3& t)
    {
    translate(t.x, t.y, t.z, this);
    }
    
    void Mat4::translate(const Vec3& t, Mat4* dst) const
    {
    translate(t.x, t.y, t.z, dst);
    }
    
    void Mat4::transpose()
    {
    MathUtils::transpose_matrix(m, m);
    }
    
    Mat4 Mat4::get_transposed() const
    {
    Mat4 mat(*this);
    mat.transpose();
    return mat;
    }
    
    const Mat4 Mat4::operator+(float scalar) const
    {
    Mat4 result(*this);
    result.add(scalar);
    return result;
    }
    
    Mat4& Mat4::operator+=(float scalar)
    {
    add(scalar);
    return *this;
    }
    
    const Mat4 Mat4::operator-(float scalar) const
    {
    Mat4 result(*this);
    result.add(-scalar);
    return result;
    }
    
    Mat4& Mat4::operator-=(float scalar)
    {
    add(-scalar);
    return *this;
    }
    
    const Mat4 Mat4::operator*(float scalar) const
    {
    Mat4 result(*this);
    result.multiply(scalar);
    return result;
    }
    
    Mat4& Mat4::operator*=(float scalar)
    {
    multiply(scalar);
    return *this;
    }
    
    const Mat4 Mat4::IDENTITY = Mat4(1.0f, 0.0f, 0.0f, 0.0f,
                                     0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 1.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 1.0f);
    
    const Mat4 Mat4::ZERO = Mat4(0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0 );
    
}
