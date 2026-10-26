#pragma once
#ifndef SIMD_OPT_H
#define SIMD_OPT_H

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define PLATFORM_X86
    #include <immintrin.h>
    #include <emmintrin.h>
    #include <smmintrin.h>
    #define SIMD_SSE

#elif defined(__aarch64__) || defined(__arm__)
    #define PLATFORM_ARM
    
    #if defined(__ARM_FEATURE_SVE2)
        #include <arm_sve.h>
        #define SIMD_SVE2
    #elif defined(__ARM_FEATURE_SVE)
        #include <arm_sve.h>
        #define SIMD_SVE
    #else
        #include <arm_neon.h>
        #define SIMD_NEON
    #endif
#else
    #error "Unsupported platform"
#endif

#if defined(SIMD_SSE)
    // SSE instruction set definition
    #define SIMD_TYPE             __m128i
    #define SIMD_ITERATION_SIZE   16
    #define SIMD_POWER            4

    #define SIMD_XOR(a, b)        _mm_xor_si128((a), (b))
    #define SIMD_OR(a, b)         _mm_or_si128((a), (b))
    #define SIMD_ZEROTEST(a, b)   _mm_testz_si128((a), (b))
    #define SIMD_LOAD(ptr)        _mm_load_si128((const __m128i*)(ptr))
    #define SIMD_STORE(ptr, val)  _mm_store_si128((__m128i*)(ptr), (val))
    #define SIMD_LOAD_U(ptr)      _mm_loadu_si128((const __m128i*)(ptr))
    #define SIMD_STORE_U(ptr, val) _mm_storeu_si128((__m128i*)(ptr), (val))
    #define SIMD_SET(value)       _mm_set1_epi8((value))
    #define SIMD_SET_ZERO         _mm_setzero_si128()

    #define SIMD_EOR3_U8(result, a, b, c) \
            do { \
                (result) = _mm_xor_si128(_mm_xor_si128((a), (b)), (c)); \
            } while (0)
    typedef struct { __m128i val[4]; } __m128i_x4_t;
    #define SIMD_TYPE_X4            __m128i_x4_t
    #define SIMD_LOAD_X4(ptr)       ({ \
            __m128i_x4_t _r; \
            const __m128i *_p = (const __m128i*)(ptr); \
            _r.val[0] = _mm_load_si128(_p + 0); \
            _r.val[1] = _mm_load_si128(_p + 1); \
            _r.val[2] = _mm_load_si128(_p + 2); \
            _r.val[3] = _mm_load_si128(_p + 3); \
            _r; \
        })
    #define SIMD_X4_VAL(tuple, idx) ((tuple).val[(idx)])

#elif defined(SIMD_SVE2)
    // SVE2 
    #define SIMD_TYPE             svuint8_t
    #define SIMD_ITERATION_SIZE   svcntb()
    #define SVE_ALL_TRUE          svptrue_b8()

    #define SIMD_XOR(a, b)         sveor_u8_x(SVE_ALL_TRUE, (a), (b))
    #define SIMD_OR(a, b)          svorr_u8_x(SVE_ALL_TRUE, (a), (b))
    #define SIMD_EOR3_U8(a, b, c)  sveor3_u8((a), (b), (c))
    
    #define SIMD_ZEROTEST(a, b)                                      \
    (!svptest_any(                                              \
        SVE_ALL_TRUE,                                           \
        svcmpne_n_u8(                                           \
            SVE_ALL_TRUE,                                       \
            svand_u8_x(SVE_ALL_TRUE, (a), (b)),                  \
            0)))

    #define SIMD_LOAD(ptr)         svld1_u8(SVE_ALL_TRUE, (const uint8_t*)(ptr))
    #define SIMD_STORE(ptr, val)   svst1_u8(SVE_ALL_TRUE, (uint8_t*)(ptr), (val))
    #define SIMD_LOAD_U(ptr)       svld1_u8(SVE_ALL_TRUE, (const uint8_t*)(ptr))
    #define SIMD_STORE_U(ptr, val) svst1_u8(SVE_ALL_TRUE, (uint8_t*)(ptr), (val))
    #define SIMD_SET(value)        svdup_n_u8((value))
    #define SIMD_SET_ZERO          svdup_n_u8(0)
    // 第 0 路
    #define SIMD_LOAD_R0(ptr)          svld1_u8(SVE_ALL_TRUE, (const uint8_t*)(ptr))
    #define SIMD_STORE_R0(ptr, val)    svst1_u8(SVE_ALL_TRUE, (uint8_t*)(ptr), (val))

    // 第 1 路
    #define SIMD_LOAD_R1(ptr)          svld1_vnum_u8(SVE_ALL_TRUE, (const uint8_t*)(ptr), 1)
    #define SIMD_STORE_R1(ptr, val)    svst1_vnum_u8(SVE_ALL_TRUE, (uint8_t*)(ptr), 1, (val))
    #define SIMD_EOR3(result, a, b, c) \
            __asm__ volatile( \
                "eor3 %0.d, %1.d, %2.d, %3.d" \
                : "=w" (result) \
                : "w" (a), "w" (b), "w" (c) \
            )
    #define SIMD_TYPE_X4 svuint8x4_t
    static inline svuint8x4_t
    simd_load_x4_u8(const uint8_t *ptr)
    {
        svbool_t pg = svptrue_b8();

        return svcreate4_u8(
            svld1_u8(pg, ptr),
            svld1_vnum_u8(pg, ptr, 1),
            svld1_vnum_u8(pg, ptr, 2),
            svld1_vnum_u8(pg, ptr, 3));
    }

    #define SIMD_LOAD_X4(ptr) \
        simd_load_x4_u8((const uint8_t *)(ptr))

    #define SIMD_X4_VAL(tuple, idx) \
        svget4_u8((tuple), (idx))

#elif defined(SIMD_SVE)
    // SVE
    #define SIMD_TYPE             svuint8_t
    #define SIMD_ITERATION_SIZE   svcntb()
    #define SVE_ALL_TRUE          svptrue_b8()

    #define SIMD_XOR(a, b)         sveor_u8_x(SVE_ALL_TRUE, (a), (b))
    #define SIMD_OR(a, b)          svorr_u8_x(SVE_ALL_TRUE, (a), (b))
    #define SIMD_EOR3_U8(a, b, c)  sveor_u8_x(SVE_ALL_TRUE, sveor_u8_x(SVE_ALL_TRUE, (a), (b)), (c))
    
    #define SIMD_ZEROTEST(a, b)                                      \
    (!svptest_any(                                              \
        SVE_ALL_TRUE,                                           \
        svcmpne_n_u8(                                           \
            SVE_ALL_TRUE,                                       \
            svand_u8_x(SVE_ALL_TRUE, (a), (b)),                  \
            0)))

    #define SIMD_LOAD(ptr)         svld1_u8(SVE_ALL_TRUE, (const uint8_t*)(ptr))
    #define SIMD_STORE(ptr, val)   svst1_u8(SVE_ALL_TRUE, (uint8_t*)(ptr), (val))
    #define SIMD_LOAD_U(ptr)       svld1_u8(SVE_ALL_TRUE, (const uint8_t*)(ptr))
    #define SIMD_STORE_U(ptr, val) svst1_u8(SVE_ALL_TRUE, (uint8_t*)(ptr), (val))
    #define SIMD_SET(value)        svdup_n_u8((value))
    #define SIMD_SET_ZERO          svdup_n_u8(0)
    // 第 0 路
    #define SIMD_LOAD_R0(ptr)          svld1_u8(SVE_ALL_TRUE, (const uint8_t*)(ptr))
    #define SIMD_STORE_R0(ptr, val)    svst1_u8(SVE_ALL_TRUE, (uint8_t*)(ptr), (val))

    // 第 1 路
    #define SIMD_LOAD_R1(ptr)          svld1_vnum_u8(SVE_ALL_TRUE, (const uint8_t*)(ptr), 1)
    #define SIMD_STORE_R1(ptr, val)    svst1_vnum_u8(SVE_ALL_TRUE, (uint8_t*)(ptr), 1, (val))
    #define SIMD_EOR3(result, a, b, c) \
    (result) = sveor_u8_x(SVE_ALL_TRUE, sveor_u8_x(SVE_ALL_TRUE, (a), (b)), (c))
    #define SIMD_TYPE_X4 svuint8x4_t
    static inline svuint8x4_t
    simd_load_x4_u8(const uint8_t *ptr)
    {
        svbool_t pg = svptrue_b8();

        return svcreate4_u8(
            svld1_u8(pg, ptr),
            svld1_vnum_u8(pg, ptr, 1),
            svld1_vnum_u8(pg, ptr, 2),
            svld1_vnum_u8(pg, ptr, 3));
    }

    #define SIMD_LOAD_X4(ptr) \
        simd_load_x4_u8((const uint8_t *)(ptr))

    #define SIMD_X4_VAL(tuple, idx) \
        svget4_u8((tuple), (idx))

#elif defined(SIMD_NEON)
    // NEON instruction set
    #define SIMD_TYPE             uint8x16_t
    #define SIMD_ITERATION_SIZE   16
    #define SIMD_POWER            4

    #define SIMD_XOR(a, b)        veorq_u8((a), (b))
    #define SIMD_OR(a, b)         vorrq_u8((a), (b))

    #define SIMD_LOAD(ptr)        vld1q_u8((const uint8_t*)(ptr))
    #define SIMD_STORE(ptr, val)  vst1q_u8((uint8_t*)(ptr), (val))
    #define SIMD_LOAD_U(ptr)      vld1q_u8((const uint8_t*)(ptr))
    #define SIMD_STORE_U(ptr, val) vst1q_u8((uint8_t*)(ptr), (val))

    #define SIMD_SET(value)       vdupq_n_u8((value))
    #define SIMD_SET_ZERO         vdupq_n_u8(0)

    // R0/R1: 连续的两个 NEON 向量
    #define SIMD_LOAD_R0(ptr)         SIMD_LOAD(ptr)
    #define SIMD_STORE_R0(ptr, val)   SIMD_STORE(ptr, val)

    #define SIMD_LOAD_R1(ptr) \
        SIMD_LOAD((const uint8_t*)(ptr) + 16)

    #define SIMD_STORE_R1(ptr, val) \
        SIMD_STORE((uint8_t*)(ptr) + 16, val)

    // Zero test
    #define SIMD_ZEROTEST(a, b) \
        ({ \
            uint8x16_t _tz_and = vandq_u8((a), (b)); \
            uint8x8_t _tz_or = vorr_u8( \
                vget_low_u8(_tz_and), \
                vget_high_u8(_tz_and)); \
            uint64x1_t _tz_64 = vreinterpret_u64_u8(_tz_or); \
            (vget_lane_u64(_tz_64, 0) == 0); \
        })

    // EOR3: SHA3 优化与基础 NEON 回退
    #if defined(__ARM_FEATURE_SHA3)
        #define SIMD_EOR3_U8(a, b, c) \
            veor3q_u8((a), (b), (c))
        #define SIMD_EOR3(a, b, c) \
            veor3q_u8((a), (b), (c))
    #else
        #define SIMD_EOR3_U8(a, b, c) \
            veorq_u8(veorq_u8((a), (b)), (c))
        #define SIMD_EOR3(a, b, c) \
            veorq_u8(veorq_u8((a), (b)), (c))
    #endif
    #define SIMD_TYPE_X4            uint8x16x4_t
    #define SIMD_LOAD_X4(ptr)       vld1q_u8_x4((const uint8_t*)(ptr))
    #define SIMD_X4_VAL(tuple, idx) ((tuple).val[(idx)])
#else
    #error "SIMD optimization not supported on this platform"
#endif

// function declaration
void aligned_free(void* ptr);
void* aligned_malloc(size_t size, size_t alignment);

#endif // SIMD_OPT_H