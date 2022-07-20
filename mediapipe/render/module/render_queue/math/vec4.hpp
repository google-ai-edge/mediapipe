//
//  vec4.h
//  Opipe
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#ifndef VEC4_H
#define VEC4_H

namespace Opipe {
    class Vec4
    {
    public:
    float x;
    float y;
    float z;
    float w;
    
    /**
     * Constructs a new vector initialized to all zeros.
     */
    Vec4();
    
    /**
     * Constructs a new vector initialized to the specified values.
     *
     * @param xx The x coordinate.
     * @param yy The y coordinate.
     * @param zz The z coordinate.
     * @param ww The w coordinate.
     */
    Vec4(float xx, float yy, float zz, float ww);
    
    /**
     * Constructs a new vector from the values in the specified array.
     *
     * @param array An array containing the elements of the vector in the order x, y, z, w.
     */
    Vec4(const float* array);
    
    /**
     * Constructs a vector that describes the direction between the specified points.
     *
     * @param p1 The first point.
     * @param p2 The second point.
     */
    Vec4(const Vec4& p1, const Vec4& p2);
    
    /**
     * Constructor.
     *
     * Creates a new vector that is a copy of the specified vector.
     *
     * @param copy The vector to copy.
     */
    Vec4(const Vec4& copy);
    
    /**
     * Creates a new vector from an integer interpreted as an RGBA value.
     * E.g. 0xff0000ff represents opaque red or the vector (1, 0, 0, 1).
     *
     * @param color The integer to interpret as an RGBA value.
     *
     * @return A vector corresponding to the interpreted RGBA color.
     */
    static Vec4 from_color(unsigned int color);
    
    /**
     * Indicates whether this vector contains all zeros.
     *
     * @return true if this vector contains all zeros, false otherwise.
     */
    bool is_zero() const;
    
    /**
     * Indicates whether this vector contains all ones.
     *
     * @return true if this vector contains all ones, false otherwise.
     */
    bool is_one() const;
    
    /**
     * Returns the angle (in radians) between the specified vectors.
     *
     * @param v1 The first vector.
     * @param v2 The second vector.
     *
     * @return The angle between the two vectors (in radians).
     */
    static float angle(const Vec4& v1, const Vec4& v2);
    
    /**
     * Adds the elements of the specified vector to this one.
     *
     * @param v The vector to add.
     */
    void add(const Vec4& v);
    
    /**
     * Adds the specified vectors and stores the result in dst.
     *
     * @param v1 The first vector.
     * @param v2 The second vector.
     * @param dst A vector to store the result in.
     */
    static void add(const Vec4& v1, const Vec4& v2, Vec4* dst);
    
    /**
     * Clamps this vector within the specified range.
     *
     * @param min The minimum value.
     * @param max The maximum value.
     */
    void clamp(const Vec4& min, const Vec4& max);
    
    /**
     * Clamps the specified vector within the specified range and returns it in dst.
     *
     * @param v The vector to clamp.
     * @param min The minimum value.
     * @param max The maximum value.
     * @param dst A vector to store the result in.
     */
    static void clamp(const Vec4& v, const Vec4& min, const Vec4& max, Vec4* dst);
    
    /**
     * Returns the distance between this vector and v.
     *
     * @param v The other vector.
     *
     * @return The distance between this vector and v.
     *
     * @see distanceSquared
     */
    float distance(const Vec4& v) const;
    
    /**
     * Returns the squared distance between this vector and v.
     *
     * When it is not necessary to get the exact distance between
     * two vectors (for example, when simply comparing the
     * distance between different vectors), it is advised to use
     * this method instead of distance.
     *
     * @param v The other vector.
     *
     * @return The squared distance between this vector and v.
     *
     * @see distance
     */
    float distance_squared(const Vec4& v) const;
    
    /**
     * Returns the dot product of this vector and the specified vector.
     *
     * @param v The vector to compute the dot product with.
     *
     * @return The dot product.
     */
    float dot(const Vec4& v) const;
    
    /**
     * Returns the dot product between the specified vectors.
     *
     * @param v1 The first vector.
     * @param v2 The second vector.
     *
     * @return The dot product between the vectors.
     */
    static float dot(const Vec4& v1, const Vec4& v2);
    
    /**
     * Computes the length of this vector.
     *
     * @return The length of the vector.
     *
     * @see lengthSquared
     */
    float length() const;
    
    /**
     * Returns the squared length of this vector.
     *
     * When it is not necessary to get the exact length of a
     * vector (for example, when simply comparing the lengths of
     * different vectors), it is advised to use this method
     * instead of length.
     *
     * @return The squared length of the vector.
     *
     * @see length
     */
    float length_squared() const;
    
    /**
     * Negates this vector.
     */
    void negate();
    
    /**
     * Normalizes this vector.
     *
     * This method normalizes this Vec4 so that it is of
     * unit length (in other words, the length of the vector
     * after calling this method will be 1.0f). If the vector
     * already has unit length or if the length of the vector
     * is zero, this method does nothing.
     */
    void normalize();
    
    /**
     * Get the normalized vector.
     */
    Vec4 get_normalized() const;
    
    /**
     * Scales all elements of this vector by the specified value.
     *
     * @param scalar The scalar value.
     */
    void scale(float scalar);
    
    /**
     * Sets the elements of this vector to the specified values.
     *
     * @param xx The new x coordinate.
     * @param yy The new y coordinate.
     * @param zz The new z coordinate.
     * @param ww The new w coordinate.
     */
    void set(float xx, float yy, float zz, float ww);
    
    /**
     * Sets the elements of this vector from the values in the specified array.
     *
     * @param array An array containing the elements of the vector in the order x, y, z, w.
     */
    void set(const float* array);
    
    /**
     * Sets the elements of this vector to those in the specified vector.
     *
     * @param v The vector to copy.
     */
    void set(const Vec4& v);
    
    /**
     * Sets this vector to the directional vector between the specified points.
     *
     * @param p1 The first point.
     * @param p2 The second point.
     */
    void set(const Vec4& p1, const Vec4& p2);
    
    /**
     * Subtracts this vector and the specified vector as (this - v)
     * and stores the result in this vector.
     *
     * @param v The vector to subtract.
     */
    void subtract(const Vec4& v);
    
    /**
     * Subtracts the specified vectors and stores the result in dst.
     * The resulting vector is computed as (v1 - v2).
     *
     * @param v1 The first vector.
     * @param v2 The second vector.
     * @param dst The destination vector.
     */
    static void subtract(const Vec4& v1, const Vec4& v2, Vec4* dst);
    
    /**
     * Calculates the sum of this vector with the given vector.
     *
     * Note: this does not modify this vector.
     *
     * @param v The vector to add.
     * @return The vector sum.
     */
    inline Vec4 operator+(const Vec4& v) const;
    
    /**
     * Adds the given vector to this vector.
     *
     * @param v The vector to add.
     * @return This vector, after the addition occurs.
     */
    inline Vec4& operator+=(const Vec4& v);
    
    /**
     * Calculates the sum of this vector with the given vector.
     *
     * Note: this does not modify this vector.
     *
     * @param v The vector to add.
     * @return The vector sum.
     */
    inline Vec4 operator-(const Vec4& v) const;
    
    /**
     * Subtracts the given vector from this vector.
     *
     * @param v The vector to subtract.
     * @return This vector, after the subtraction occurs.
     */
    inline Vec4& operator-=(const Vec4& v);
    
    /**
     * Calculates the negation of this vector.
     *
     * Note: this does not modify this vector.
     *
     * @return The negation of this vector.
     */
    inline Vec4 operator-() const;
    
    /**
     * Calculates the scalar product of this vector with the given value.
     *
     * Note: this does not modify this vector.
     *
     * @param s The value to scale by.
     * @return The scaled vector.
     */
    inline Vec4 operator*(float s) const;
    
    /**
     * Scales this vector by the given value.
     *
     * @param s The value to scale by.
     * @return This vector, after the scale occurs.
     */
    inline Vec4& operator*=(float s);
    
    /**
     * Returns the components of this vector divided by the given constant
     *
     * Note: this does not modify this vector.
     *
     * @param s the constant to divide this vector with
     * @return a smaller vector
     */
    inline Vec4 operator/(float s) const;
    
    /**
     * Determines if this vector is less than the given vector.
     *
     * @param v The vector to compare against.
     *
     * @return True if this vector is less than the given vector, false otherwise.
     */
    inline bool operator<(const Vec4& v) const;
    
    /**
     * Determines if this vector is equal to the given vector.
     *
     * @param v The vector to compare against.
     *
     * @return True if this vector is equal to the given vector, false otherwise.
     */
    inline bool operator==(const Vec4& v) const;
    
    /**
     * Determines if this vector is not equal to the given vector.
     *
     * @param v The vector to compare against.
     *
     * @return True if this vector is not equal to the given vector, false otherwise.
     */
    inline bool operator!=(const Vec4& v) const;
    
    /** equals to Vec4(0,0,0,0) */
    static const Vec4 ZERO;
    /** equals to Vec4(1,1,1,1) */
    static const Vec4 ONE;
    /** equals to Vec4(1,0,0,0) */
    static const Vec4 UNIT_X;
    /** equals to Vec4(0,1,0,0) */
    static const Vec4 UNIT_Y;
    /** equals to Vec4(0,0,1,0) */
    static const Vec4 UNIT_Z;
    /** equals to Vec4(0,0,0,1) */
    static const Vec4 UNIT_W;
    };
    
    /**
     * Calculates the scalar product of the given vector with the given value.
     *
     * @param x The value to scale by.
     * @param v The vector to scale.
     * @return The scaled vector.
     */
    inline Vec4 operator*(float x, const Vec4& v);
    
}

#include "vec4.inl"

#endif /* VEC4_H */
