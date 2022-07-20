//
//  math_utils.h
//  Opipe
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#ifndef DI_ENGINE_MATH_UTILS_H
#define DI_ENGINE_MATH_UTILS_H

#include "mat4.hpp"
#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include <cstring>

#define MATH_DEG_TO_RAD(x)          ((x) * 0.0174532925f)
#define MATH_RAD_TO_DEG(x)          ((x)* 57.29577951f)

#define MATH_FLOAT_SMALL            1.0e-37f
#define MATH_TOLERANCE              2e-37f
#define MATH_PIOVER2                1.57079632679489661923f
#define MATH_EPSILON                0.000001f

#define MATH_FLOAT_EQUAL(src, dst)  (((src) >= (dst) - MATH_EPSILON) && ((src) <= (dst) + MATH_EPSILON))


namespace Opipe {
    
    class MathUtils
    {
    public:
    static void add_matrix(const float* m, float scalar, float* dst);
    
    static void add_matrix(const float* m1, const float* m2, float* dst);
    
    static void subtract_matrix(const float* m1, const float* m2, float* dst);
    
    static void multiply_matrix(const float* m, float scalar, float* dst);
    
    static void multiply_matrix(const float* m1, const float* m2, float* dst);
    
    static void negate_matrix(const float* m, float* dst);
    
    static void transpose_matrix(const float* m, float* dst);
    
    static void transform_vec4(const float* m, float x, float y, float z, float w, float* dst);
    
    static void transform_vec4(const float* m, const float* v, float* dst);
    
    static void cross_vec3(const float* v1, const float* v2, float* dst);
    
    public:
    /**
     * Updates the given scalar towards the given target using a smoothing function.
     * The given response time determines the amount of smoothing (lag). A longer
     * response time yields a smoother result and more lag. To force the scalar to
     * follow the target closely, provide a response time that is very small relative
     * to the given elapsed time.
     *
     * @param x the scalar to update.
     * @param target target value.
     * @param elapsedTime elapsed time between calls.
     * @param responseTime response time (in the same units as elapsedTime).
     */
    static void smooth(float* x, float target, float elapsedTime, float responseTime);
    
    /**
     * Updates the given scalar towards the given target using a smoothing function.
     * The given rise and fall times determine the amount of smoothing (lag). Longer
     * rise and fall times yield a smoother result and more lag. To force the scalar to
     * follow the target closely, provide rise and fall times that are very small relative
     * to the given elapsed time.
     *
     * @param x the scalar to update.
     * @param target target value.
     * @param elapsedTime elapsed time between calls.
     * @param riseTime response time for rising slope (in the same units as elapsedTime).
     * @param fallTime response time for falling slope (in the same units as elapsedTime).
     */
    static void smooth(float* x, float target, float elapsedTime, float riseTime, float fallTime);
    
    /**
     * Linearly interpolates between from value to to value by alpha which is in
     * the range [0,1]
     *
     * @param from the from value.
     * @param to the to value.
     * @param alpha the alpha value between [0,1]
     *
     * @return interpolated float value
     */
    static float lerp(float from, float to, float alpha);
    };
    
}

#endif /* DI_ENGINE_MATH_UTILS_H */
