//
//  mat4.h
//  Opipe
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#ifndef MAT4_H
#define MAT4_H

#include "vec3.hpp"
#include "vec4.hpp"

namespace Opipe {
    
class Mat4
{
public:
    float m[16];
    
    /**
     * Default constructor.
     * Constructs a matrix initialized to the identity matrix:
     *
     *     1  0  0  0
     *     0  1  0  0
     *     0  0  1  0
     *     0  0  0  1
     */
    Mat4();
    
    /**
     * Constructs a matrix initialized to the specified value.
     *
     * @param m11 The first element of the first row.
     * @param m12 The second element of the first row.
     * @param m13 The third element of the first row.
     * @param m14 The fourth element of the first row.
     * @param m21 The first element of the second row.
     * @param m22 The second element of the second row.
     * @param m23 The third element of the second row.
     * @param m24 The fourth element of the second row.
     * @param m31 The first element of the third row.
     * @param m32 The second element of the third row.
     * @param m33 The third element of the third row.
     * @param m34 The fourth element of the third row.
     * @param m41 The first element of the fourth row.
     * @param m42 The second element of the fourth row.
     * @param m43 The third element of the fourth row.
     * @param m44 The fourth element of the fourth row.
     */
    Mat4(float m11, float m12, float m13, float m14, float m21, float m22, float m23, float m24,
         float m31, float m32, float m33, float m34, float m41, float m42, float m43, float m44);
    
    /**
     * Creates a matrix initialized to the specified column-major array.
     *
     * The passed-in array is in column-major order, so the memory layout of the array is as follows:
     *
     *     0   4   8   12
     *     1   5   9   13
     *     2   6   10  14
     *     3   7   11  15
     *
     * @param mat An array containing 16 elements in column-major order.
     */
    Mat4(const float* mat);
    
    /**
     * Constructs a new matrix by copying the values from the specified matrix.
     *
     * @param copy The matrix to copy.
     */
    Mat4(const Mat4& copy);
    
    /**
     * Creates a view matrix based on the specified input parameters.
     *
     * @param eyePosition The eye position.
     * @param targetPosition The target's center position.
     * @param up The up vector.
     * @param dst A matrix to store the result in.
     */
    static void create_look_at(const Vec3& eyePosition, const Vec3& targetPosition, const Vec3& up, Mat4* dst);
    
    /**
     * Creates a view matrix based on the specified input parameters.
     *
     * @param eyePositionX The eye x-coordinate position.
     * @param eyePositionY The eye y-coordinate position.
     * @param eyePositionZ The eye z-coordinate position.
     * @param targetCenterX The target's center x-coordinate position.
     * @param targetCenterY The target's center y-coordinate position.
     * @param targetCenterZ The target's center z-coordinate position.
     * @param upX The up vector x-coordinate value.
     * @param upY The up vector y-coordinate value.
     * @param upZ The up vector z-coordinate value.
     * @param dst A matrix to store the result in.
     */
    static void create_look_at(float eyePositionX, float eyePositionY, float eyePositionZ,
                               float targetCenterX, float targetCenterY, float targetCenterZ,
                               float upX, float upY, float upZ, Mat4* dst);
    
    /**
     * Builds a perspective projection matrix based on a field of view and returns by value.
     *
     * Projection space refers to the space after applying projection transformation from view space.
     * After the projection transformation, visible content has x- and y-coordinates ranging from -1 to 1,
     * and a z-coordinate ranging from 0 to 1. To obtain the viewable area (in world space) of a scene,
     * create a BoundingFrustum and pass the combined view and projection matrix to the constructor.
     *
     * @param fieldOfView The field of view in the y direction (in degrees).
     * @param aspectRatio The aspect ratio, defined as view space width divided by height.
     * @param zNearPlane The distance to the near view plane.
     * @param zFarPlane The distance to the far view plane.
     * @param dst A matrix to store the result in.
     */
    static void create_perspective(float fieldOfView, float aspectRatio, float zNearPlane, float zFarPlane, Mat4* dst);
    
    /**
     * Creates an orthographic projection matrix.
     *
     * @param width The width of the view.
     * @param height The height of the view.
     * @param zNearPlane The minimum z-value of the view volume.
     * @param zFarPlane The maximum z-value of the view volume.
     * @param dst A matrix to store the result in.
     */
    static void create_orthographic(float width, float height, float zNearPlane, float zFarPlane, Mat4* dst);
    
    /**
     * Creates an orthographic projection matrix.
     *
     * Projection space refers to the space after applying
     * projection transformation from view space. After the
     * projection transformation, visible content has
     * x and y coordinates ranging from -1 to 1, and z coordinates
     * ranging from 0 to 1.
     *
     * Unlike perspective projection, in orthographic projection
     * there is no perspective foreshortening.
     *
     * The viewable area of this orthographic projection extends
     * from left to right on the x-axis, bottom to top on the y-axis,
     * and zNearPlane to zFarPlane on the z-axis. These values are
     * relative to the position and x, y, and z-axes of the view.
     * To obtain the viewable area (in world space) of a scene,
     * create a BoundingFrustum and pass the combined view and
     * projection matrix to the constructor.
     *
     * @param left The minimum x-value of the view volume.
     * @param right The maximum x-value of the view volume.
     * @param bottom The minimum y-value of the view volume.
     * @param top The maximum y-value of the view volume.
     * @param zNearPlane The minimum z-value of the view volume.
     * @param zFarPlane The maximum z-value of the view volume.
     * @param dst A matrix to store the result in.
     */
    static void create_orthographic_off_center(float left, float right, float bottom, float top,
                                               float zNearPlane, float zFarPlane, Mat4* dst);
    
    //Fills in an existing Mat4 so that it reflects the coordinate system about a specified Plane.
    //plane The Plane about which to create a reflection.
    //dst A matrix to store the result in.
    //static void createReflection(const Plane& plane, Mat4* dst);
    
    /**
     * Creates a scale matrix.
     *
     * @param scale The amount to scale.
     * @param dst A matrix to store the result in.
     */
    static void create_scale(const Vec3& scale, Mat4* dst);
    
    /**
     * Creates a scale matrix.
     *
     * @param xScale The amount to scale along the x-axis.
     * @param yScale The amount to scale along the y-axis.
     * @param zScale The amount to scale along the z-axis.
     * @param dst A matrix to store the result in.
     */
    static void create_scale(float xScale, float yScale, float zScale, Mat4* dst);
    
    /**
     * Creates a rotation matrix from the specified axis and angle.
     *
     * @param axis A vector describing the axis to rotate about.
     * @param angle The angle (in radians).
     * @param dst A matrix to store the result in.
     */
    static void create_rotation(const Vec3& axis, float angle, Mat4* dst);
    
    /**
     * Creates a matrix describing a rotation around the x-axis.
     *
     * @param angle The angle of rotation (in radians).
     * @param dst A matrix to store the result in.
     */
    static void create_rotation_x(float angle, Mat4* dst);
    
    /**
     * Creates a matrix describing a rotation around the y-axis.
     *
     * @param angle The angle of rotation (in radians).
     * @param dst A matrix to store the result in.
     */
    static void create_rotation_y(float angle, Mat4* dst);
    
    /**
     * Creates a matrix describing a rotation around the z-axis.
     *
     * @param angle The angle of rotation (in radians).
     * @param dst A matrix to store the result in.
     */
    static void create_rotation_z(float angle, Mat4* dst);
    
    /**
     * Creates a translation matrix.
     *
     * @param translation The translation.
     * @param dst A matrix to store the result in.
     */
    static void create_translation(const Vec3& translation, Mat4* dst);
    
    /**
     * Creates a translation matrix.
     *
     * @param xTranslation The translation on the x-axis.
     * @param yTranslation The translation on the y-axis.
     * @param zTranslation The translation on the z-axis.
     * @param dst A matrix to store the result in.
     */
    static void create_translation(float xTranslation, float yTranslation, float zTranslation, Mat4* dst);
    
    /**
     * Adds a scalar value to each component of this matrix.
     *
     * @param scalar The scalar to add.
     */
    void add(float scalar);
    
    /**
     * Adds a scalar value to each component of this matrix and stores the result in dst.
     *
     * @param scalar The scalar value to add.
     * @param dst A matrix to store the result in.
     */
    void add(float scalar, Mat4* dst);
    
    /**
     * Adds the specified matrix to this matrix.
     *
     * @param mat The matrix to add.
     */
    void add(const Mat4& mat);
    
    /**
     * Adds the specified matrices and stores the result in dst.
     *
     * @param m1 The first matrix.
     * @param m2 The second matrix.
     * @param dst The destination matrix to add to.
     */
    static void add(const Mat4& m1, const Mat4& m2, Mat4* dst);
    
    /**
     * Computes the determinant of this matrix.
     *
     * @return The determinant.
     */
    float determinant() const;
    
    /**
     * Gets the up vector of this matrix.
     *
     * @param dst The destination vector.
     */
    void get_up_vector(Vec3* dst) const;
    
    /**
     * Gets the down vector of this matrix.
     *
     * @param dst The destination vector.
     */
    void get_down_vector(Vec3* dst) const;
    
    /**
     * Gets the left vector of this matrix.
     *
     * @param dst The destination vector.
     */
    void get_left_vector(Vec3* dst) const;
    
    /**
     * Gets the right vector of this matrix.
     *
     * @param dst The destination vector.
     */
    void get_right_vector(Vec3* dst) const;
    
    /**
     * Gets the forward vector of this matrix.
     *
     * @param dst The destination vector.
     */
    void get_forward_vector(Vec3* dst) const;
    
    /**
     * Gets the backward vector of this matrix.
     *
     * @param dst The destination vector.
     */
    void get_back_vector(Vec3* dst) const;
    
    /**
     * Inverts this matrix.
     *
     * @return true if the matrix can be inverted, false otherwise.
     */
    bool inverse();
    
    /**
     * Get the inversed matrix.
     */
    Mat4 get_inversed() const;
    
    /**
     * Determines if this matrix is equal to the identity matrix.
     *
     * @return true if the matrix is an identity matrix, false otherwise.
     */
    bool is_identity() const;
    
    /**
     * Multiplies the components of this matrix by the specified scalar.
     *
     * @param scalar The scalar value.
     */
    void multiply(float scalar);
    
    /**
     * Multiplies the components of this matrix by a scalar and stores the result in dst.
     *
     * @param scalar The scalar value.
     * @param dst A matrix to store the result in.
     */
    void multiply(float scalar, Mat4* dst) const;
    
    /**
     * Multiplies the components of the specified matrix by a scalar and stores the result in dst.
     *
     * @param mat The matrix.
     * @param scalar The scalar value.
     * @param dst A matrix to store the result in.
     */
    static void multiply(const Mat4& mat, float scalar, Mat4* dst);
    
    /**
     * Multiplies this matrix by the specified one.
     *
     * @param mat The matrix to multiply.
     */
    void multiply(const Mat4& mat);
    
    /**
     * Multiplies m1 by m2 and stores the result in dst.
     *
     * @param m1 The first matrix to multiply.
     * @param m2 The second matrix to multiply.
     * @param dst A matrix to store the result in.
     */
    static void multiply(const Mat4& m1, const Mat4& m2, Mat4* dst);
    
    /**
     * Negates this matrix.
     */
    void negate();
    
    /**
     Get the Negated matrix.
     */
    Mat4 get_negated() const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified rotation about the specified axis.
     *
     * @param axis The axis to rotate about.
     * @param angle The angle (in radians).
     */
    void rotate(const Vec3& axis, float angle);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the specified
     * rotation about the specified axis and stores the result in dst.
     *
     * @param axis The axis to rotate about.
     * @param angle The angle (in radians).
     * @param dst A matrix to store the result in.
     */
    void rotate(const Vec3& axis, float angle, Mat4* dst) const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified rotation around the x-axis.
     *
     * @param angle The angle (in radians).
     */
    void rotate_x(float angle);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified rotation around the x-axis and stores the result in dst.
     *
     * @param angle The angle (in radians).
     * @param dst A matrix to store the result in.
     */
    void rotate_x(float angle, Mat4* dst) const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified rotation around the y-axis.
     *
     * @param angle The angle (in radians).
     */
    void rotate_y(float angle);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified rotation around the y-axis and stores the result in dst.
     *
     * @param angle The angle (in radians).
     * @param dst A matrix to store the result in.
     */
    void rotate_y(float angle, Mat4* dst) const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified rotation around the z-axis.
     *
     * @param angle The angle (in radians).
     */
    void rotate_z(float angle);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified rotation around the z-axis and stores the result in dst.
     *
     * @param angle The angle (in radians).
     * @param dst A matrix to store the result in.
     */
    void rotate_z(float angle, Mat4* dst) const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified scale transformation.
     *
     * @param value The amount to scale along all axes.
     */
    void scale(float value);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified scale transformation and stores the result in dst.
     *
     * @param value The amount to scale along all axes.
     * @param dst A matrix to store the result in.
     */
    void scale(float value, Mat4* dst) const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified scale transformation.
     *
     * @param xScale The amount to scale along the x-axis.
     * @param yScale The amount to scale along the y-axis.
     * @param zScale The amount to scale along the z-axis.
     */
    void scale(float xScale, float yScale, float zScale);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified scale transformation and stores the result in dst.
     *
     * @param xScale The amount to scale along the x-axis.
     * @param yScale The amount to scale along the y-axis.
     * @param zScale The amount to scale along the z-axis.
     * @param dst A matrix to store the result in.
     */
    void scale(float xScale, float yScale, float zScale, Mat4* dst) const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified scale transformation.
     *
     * @param s The scale values along the x, y and z axes.
     */
    void scale(const Vec3& s);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified scale transformation and stores the result in dst.
     *
     * @param s The scale values along the x, y and z axes.
     * @param dst A matrix to store the result in.
     */
    void scale(const Vec3& s, Mat4* dst) const;
    
    /**
     * Sets the values of this matrix.
     *
     * @param m11 The first element of the first row.
     * @param m12 The second element of the first row.
     * @param m13 The third element of the first row.
     * @param m14 The fourth element of the first row.
     * @param m21 The first element of the second row.
     * @param m22 The second element of the second row.
     * @param m23 The third element of the second row.
     * @param m24 The fourth element of the second row.
     * @param m31 The first element of the third row.
     * @param m32 The second element of the third row.
     * @param m33 The third element of the third row.
     * @param m34 The fourth element of the third row.
     * @param m41 The first element of the fourth row.
     * @param m42 The second element of the fourth row.
     * @param m43 The third element of the fourth row.
     * @param m44 The fourth element of the fourth row.
     */
    void set(float m11, float m12, float m13, float m14, float m21, float m22, float m23, float m24,
             float m31, float m32, float m33, float m34, float m41, float m42, float m43, float m44);
    
    /**
     * Sets the values of this matrix to those in the specified column-major array.
     *
     * @param mat An array containing 16 elements in column-major format.
     */
    void set(const float* mat);
    
    /**
     * Sets the values of this matrix to those of the specified matrix.
     *
     * @param mat The source matrix.
     */
    void set(const Mat4& mat);
    
    /**
     * Sets this matrix to the identity matrix.
     */
    void set_identity();
    
    /**
     * Sets all elements of the current matrix to zero.
     */
    void set_zero();
    
    /**
     * Subtracts the specified matrix from the current matrix.
     *
     * @param mat The matrix to subtract.
     */
    void subtract(const Mat4& mat);
    
    /**
     * Subtracts the specified matrix from the current matrix.
     *
     * @param m1 The first matrix.
     * @param m2 The second matrix.
     * @param dst A matrix to store the result in.
     */
    static void subtract(const Mat4& m1, const Mat4& m2, Mat4* dst);
    
    /**
     * Transforms the specified point by this matrix.
     *
     * The result of the transformation is stored directly into point.
     *
     * @param point The point to transform and also a vector to hold the result in.
     */
    inline void transform_point(Vec3* point) const {
        transform_vector(point->x, point->y, point->z, 1.0f, point);
    }
    
    /**
     * Transforms the specified point by this matrix, and stores
     * the result in dst.
     *
     * @param point The point to transform.
     * @param dst A vector to store the transformed point in.
     */
    inline void transform_point(const Vec3& point, Vec3* dst) const {
        transform_vector(point.x, point.y, point.z, 1.0f, dst);
    }
    
    /**
     * Transforms the specified vector by this matrix by
     * treating the fourth (w) coordinate as zero.
     *
     * The result of the transformation is stored directly into vector.
     *
     * @param vector The vector to transform and also a vector to hold the result in.
     */
    void transform_vector(Vec3* vector) const;
    
    /**
     * Transforms the specified vector by this matrix by
     * treating the fourth (w) coordinate as zero, and stores the
     * result in dst.
     *
     * @param vector The vector to transform.
     * @param dst A vector to store the transformed vector in.
     */
    void transform_vector(const Vec3& vector, Vec3* dst) const;
    
    /**
     * Transforms the specified vector by this matrix.
     *
     * @param x The vector x-coordinate to transform by.
     * @param y The vector y-coordinate to transform by.
     * @param z The vector z-coordinate to transform by.
     * @param w The vector w-coordinate to transform by.
     * @param dst A vector to store the transformed point in.
     */
    void transform_vector(float x, float y, float z, float w, Vec3* dst) const;
    
    /**
     * Transforms the specified vector by this matrix.
     *
     * The result of the transformation is stored directly into vector.
     *
     * @param vector The vector to transform.
     */
    void transform_vector(Vec4* vector) const;
    
    /**
     * Transforms the specified vector by this matrix.
     *
     * @param vector The vector to transform.
     * @param dst A vector to store the transformed point in.
     */
    void transform_vector(const Vec4& vector, Vec4* dst) const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified translation.
     *
     * @param x The amount to translate along the x-axis.
     * @param y The amount to translate along the y-axis.
     * @param z The amount to translate along the z-axis.
     */
    void translate(float x, float y, float z);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified translation and stores the result in dst.
     *
     * @param x The amount to translate along the x-axis.
     * @param y The amount to translate along the y-axis.
     * @param z The amount to translate along the z-axis.
     * @param dst A matrix to store the result in.
     */
    void translate(float x, float y, float z, Mat4* dst) const;
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified translation.
     *
     * @param t The translation values along the x, y and z axes.
     */
    void translate(const Vec3& t);
    
    /**
     * Post-multiplies this matrix by the matrix corresponding to the
     * specified translation and stores the result in dst.
     *
     * @param t The translation values along the x, y and z axes.
     * @param dst A matrix to store the result in.
     */
    void translate(const Vec3& t, Mat4* dst) const;
    
    /**
     * Transposes this matrix.
     */
    void transpose();
    
    /**
     * Get the Transposed matrix.
     */
    Mat4 get_transposed() const;
    
    /**
     * Calculates the sum of this matrix with the given matrix.
     *
     * Note: this does not modify this matrix.
     *
     * @param mat The matrix to add.
     * @return The matrix sum.
     */
    inline Mat4 operator+(const Mat4& mat) const;
    
    /**
     * Adds the given matrix to this matrix.
     *
     * @param mat The matrix to add.
     * @return This matrix, after the addition occurs.
     */
    inline Mat4& operator+=(const Mat4& mat);
    
    /**
     * Calculates the difference of this matrix with the given matrix.
     *
     * Note: this does not modify this matrix.
     *
     * @param mat The matrix to subtract.
     * @return The matrix difference.
     */
    inline Mat4 operator-(const Mat4& mat) const;
    
    /**
     * Subtracts the given matrix from this matrix.
     *
     * @param mat The matrix to subtract.
     * @return This matrix, after the subtraction occurs.
     */
    inline Mat4& operator-=(const Mat4& mat);
    
    /**
     * Calculates the negation of this matrix.
     *
     * Note: this does not modify this matrix.
     *
     * @return The negation of this matrix.
     */
    inline Mat4 operator-() const;
    
    /**
     * Calculates the matrix product of this matrix with the given matrix.
     *
     * Note: this does not modify this matrix.
     *
     * @param mat The matrix to multiply by.
     * @return The matrix product.
     */
    inline Mat4 operator*(const Mat4& mat) const;
    
    /**
     * Right-multiplies this matrix by the given matrix.
     *
     * @param mat The matrix to multiply by.
     * @return This matrix, after the multiplication occurs.
     */
    inline Mat4& operator*=(const Mat4& mat);
    
    const Mat4 operator+(float scalar) const;
    Mat4& operator+=(float scalar);
    const Mat4 operator-(float scalar) const;
    Mat4& operator-=(float scalar);
    const Mat4 operator*(float scalar) const;
    Mat4& operator*=(float scalar);
    
    /** equals to a matrix full of zeros */
    static const Mat4 ZERO;
    /** equals to the identity matrix */
    static const Mat4 IDENTITY;
};

/**
 * Transforms the given vector by the given matrix.
 *
 * Note: this treats the given vector as a vector and not as a point.
 *
 * @param v The vector to transform.
 * @param m The matrix to transform by.
 * @return This vector, after the transformation occurs.
 */
inline Vec3& operator*=(Vec3& v, const Mat4& m);

/**
 * Transforms the given vector by the given matrix.
 *
 * Note: this treats the given vector as a vector and not as a point.
 *
 * @param m The matrix to transform by.
 * @param v The vector to transform.
 * @return The resulting transformed vector.
 */
inline Vec3 operator*(const Mat4& m, const Vec3& v);

/**
 * Transforms the given vector by the given matrix.
 *
 * Note: this treats the given vector as a vector and not as a point.
 *
 * @param v The vector to transform.
 * @param m The matrix to transform by.
 * @return This vector, after the transformation occurs.
 */
inline Vec4& operator*=(Vec4& v, const Mat4& m);

/**
 * Transforms the given vector by the given matrix.
 *
 * Note: this treats the given vector as a vector and not as a point.
 *
 * @param m The matrix to transform by.
 * @param v The vector to transform.
 * @return The resulting transformed vector.
 */
inline Vec4 operator*(const Mat4& m, const Vec4& v);

}

#include "mat4.inl"

#endif /* MAT4_H */
