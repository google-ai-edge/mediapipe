//
//  math_utils.cpp
//  Opipe
//
//  Created by Wang,Renzhu on 2018/11/20.
//  Copyright © 2018年 Wang,Renzhu. All rights reserved.
//

#if defined(__APPLE__)
#include "math_utils.hpp"
#else
#include "math_utils.hpp"
#endif
#include <cstring>

namespace Opipe {
    
    static const int MATRIX_SIZE = (sizeof(float) * 16);
    
    void MathUtils::add_matrix(const float* m, float scalar, float* dst) {
        for (int i = 0; i < 16; ++i) {
            dst[i] = m[i] + scalar;
        }
    }
    
    void MathUtils::add_matrix(const float* m1, const float* m2, float* dst) {
        for (int i = 0; i < 16; ++i) {
            dst[i] = m1[i] + m2[i];
        }
    }
    
    void MathUtils::subtract_matrix(const float* m1, const float* m2, float* dst) {
        for (int i = 0; i < 16; ++i) {
            dst[i] = m1[i] - m2[i];
        }
    }
    
    void MathUtils::multiply_matrix(const float* m, float scalar, float* dst) {
        for (int i = 0; i < 16; ++i) {
            dst[i] = m[i] * scalar;
        }
    }
    
    void MathUtils::multiply_matrix(const float* m1, const float* m2, float* dst) {
        // Support the case where m1 or m2 is the same array as dst.
        float product[16];
        
        product[0]  = m1[0] * m2[0]  + m1[4] * m2[1] + m1[8]   * m2[2]  + m1[12] * m2[3];
        product[1]  = m1[1] * m2[0]  + m1[5] * m2[1] + m1[9]   * m2[2]  + m1[13] * m2[3];
        product[2]  = m1[2] * m2[0]  + m1[6] * m2[1] + m1[10]  * m2[2]  + m1[14] * m2[3];
        product[3]  = m1[3] * m2[0]  + m1[7] * m2[1] + m1[11]  * m2[2]  + m1[15] * m2[3];
        
        product[4]  = m1[0] * m2[4]  + m1[4] * m2[5] + m1[8]   * m2[6]  + m1[12] * m2[7];
        product[5]  = m1[1] * m2[4]  + m1[5] * m2[5] + m1[9]   * m2[6]  + m1[13] * m2[7];
        product[6]  = m1[2] * m2[4]  + m1[6] * m2[5] + m1[10]  * m2[6]  + m1[14] * m2[7];
        product[7]  = m1[3] * m2[4]  + m1[7] * m2[5] + m1[11]  * m2[6]  + m1[15] * m2[7];
        
        product[8]  = m1[0] * m2[8]  + m1[4] * m2[9] + m1[8]   * m2[10] + m1[12] * m2[11];
        product[9]  = m1[1] * m2[8]  + m1[5] * m2[9] + m1[9]   * m2[10] + m1[13] * m2[11];
        product[10] = m1[2] * m2[8]  + m1[6] * m2[9] + m1[10]  * m2[10] + m1[14] * m2[11];
        product[11] = m1[3] * m2[8]  + m1[7] * m2[9] + m1[11]  * m2[10] + m1[15] * m2[11];
        
        product[12] = m1[0] * m2[12] + m1[4] * m2[13] + m1[8]  * m2[14] + m1[12] * m2[15];
        product[13] = m1[1] * m2[12] + m1[5] * m2[13] + m1[9]  * m2[14] + m1[13] * m2[15];
        product[14] = m1[2] * m2[12] + m1[6] * m2[13] + m1[10] * m2[14] + m1[14] * m2[15];
        product[15] = m1[3] * m2[12] + m1[7] * m2[13] + m1[11] * m2[14] + m1[15] * m2[15];
        
        memcpy(dst, product, MATRIX_SIZE);
    }
    
    void MathUtils::negate_matrix(const float* m, float* dst) {
        for (int i = 0; i < 16; ++i) {
            dst[i] = -m[i];
        }
    }
    
    void MathUtils::transpose_matrix(const float* m, float* dst) {
        float t[16] = {
            m[0], m[4], m[8], m[12],
            m[1], m[5], m[9], m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15]
        };
        memcpy(dst, t, MATRIX_SIZE);
    }
    
    void MathUtils::transform_vec4(const float* m, float x, float y, float z, float w, float* dst) {
        dst[0] = x * m[0] + y * m[4] + z * m[8] + w * m[12];
        dst[1] = x * m[1] + y * m[5] + z * m[9] + w * m[13];
        dst[2] = x * m[2] + y * m[6] + z * m[10] + w * m[14];
    }
    
    void MathUtils::transform_vec4(const float* m, const float* v, float* dst) {
        // Handle case where v == dst.
        float x = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + v[3] * m[12];
        float y = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + v[3] * m[13];
        float z = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + v[3] * m[14];
        float w = v[0] * m[3] + v[1] * m[7] + v[2] * m[11] + v[3] * m[15];
        
        dst[0] = x;
        dst[1] = y;
        dst[2] = z;
        dst[3] = w;
    }
    
    void MathUtils::cross_vec3(const float* v1, const float* v2, float* dst) {
        float x = (v1[1] * v2[2]) - (v1[2] * v2[1]);
        float y = (v1[2] * v2[0]) - (v1[0] * v2[2]);
        float z = (v1[0] * v2[1]) - (v1[1] * v2[0]);
        
        dst[0] = x;
        dst[1] = y;
        dst[2] = z;
    }
    
    void MathUtils::smooth(float* x, float target, float elapsedTime, float responseTime)
    {
    if (elapsedTime > 0) {
        *x += (target - *x) * elapsedTime / (elapsedTime + responseTime);
    }
    }
    
    void MathUtils::smooth(float* x, float target, float elapsedTime, float riseTime, float fallTime)
    {
    if (elapsedTime > 0) {
        float delta = target - *x;
        *x += delta * elapsedTime / (elapsedTime + (delta > 0 ? riseTime : fallTime));
    }
    }
    
    float MathUtils::lerp(float from, float to, float alpha)
    {
    return from * (1.0f - alpha) + to * alpha;
    }
    
}
