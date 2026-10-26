
#include "erasure_code.h"
#include "simd_opt.h"
#include "test.h"
#include "ring_internal.h"
#include <stdlib.h>

enum {
    RING_DATA_SLOTS = 4,
    RING_TOTAL_SLOTS = 5,
};

#define RING_SIMD_BATCH_BYTES (RING_DATA_SLOTS * SIMD_ITERATION_SIZE)
#define RING_TWO_BATCH_BYTES (2 * RING_SIMD_BATCH_BYTES)


/* The tail keeps four equally sized coefficient slots. */
static inline SIMD_TYPE
ring_tail_load_u8(const unsigned char *ptr, size_t bytes)
{
#if defined(SIMD_SVE) || defined(SIMD_SVE2)
    svbool_t pg = svwhilelt_b8_u64(0, (uint64_t)bytes);
    return svld1_u8(pg, ptr);
#else
    if (bytes == SIMD_ITERATION_SIZE)
        return SIMD_LOAD_R0(ptr);
    unsigned char tmp[SIMD_ITERATION_SIZE] = {0};
    memcpy(tmp, ptr, bytes);
    return SIMD_LOAD_U(tmp);
#endif
}

static inline void
ring_tail_store_u8(unsigned char *ptr, SIMD_TYPE value, size_t bytes)
{
#if defined(SIMD_SVE) || defined(SIMD_SVE2)
    svbool_t pg = svwhilelt_b8_u64(0, (uint64_t)bytes);
    svst1_u8(pg, ptr, value);
#else
    if (bytes == SIMD_ITERATION_SIZE) {
        SIMD_STORE_R0(ptr, value);
        return;
    }
    unsigned char tmp[SIMD_ITERATION_SIZE];
    SIMD_STORE_U(tmp, value);
    memcpy(ptr, tmp, bytes);
#endif
}

static inline size_t
ring_tail_slot_bytes(size_t remaining)
{
    if (remaining % RING_DATA_SLOTS != 0)
        abort();
    return remaining / RING_DATA_SLOTS;
}

static inline void recover_data_pair_tail(unsigned char **p0,
                                          unsigned char **p1,
                                          unsigned char **par4,
                                          unsigned char **par5,
                                          unsigned char **dst0,
                                          unsigned char **dst1,
                                          int move0, int move1,
                                          int pair_key, size_t offset, size_t slot_bytes)
{
    SIMD_TYPE zero = SIMD_SET_ZERO;
    SIMD_TYPE p0v0 = ring_tail_load_u8(p0[0] + offset, slot_bytes);
    SIMD_TYPE p0v1 = ring_tail_load_u8(p0[1] + offset, slot_bytes);
    SIMD_TYPE p0v2 = ring_tail_load_u8(p0[2] + offset, slot_bytes);
    SIMD_TYPE p0v3 = ring_tail_load_u8(p0[3] + offset, slot_bytes);
    SIMD_TYPE p1v0 = ring_tail_load_u8(p1[0] + offset, slot_bytes);
    SIMD_TYPE p1v1 = ring_tail_load_u8(p1[1] + offset, slot_bytes);
    SIMD_TYPE p1v2 = ring_tail_load_u8(p1[2] + offset, slot_bytes);
    SIMD_TYPE p1v3 = ring_tail_load_u8(p1[3] + offset, slot_bytes);
    SIMD_TYPE pv40 = ring_tail_load_u8(par4[0] + offset, slot_bytes);
    SIMD_TYPE pv41 = ring_tail_load_u8(par4[1] + offset, slot_bytes);
    SIMD_TYPE pv42 = ring_tail_load_u8(par4[2] + offset, slot_bytes);
    SIMD_TYPE pv43 = ring_tail_load_u8(par4[3] + offset, slot_bytes);
    SIMD_TYPE pv50 = ring_tail_load_u8(par5[0] + offset, slot_bytes);
    SIMD_TYPE pv51 = ring_tail_load_u8(par5[1] + offset, slot_bytes);
    SIMD_TYPE pv52 = ring_tail_load_u8(par5[2] + offset, slot_bytes);
    SIMD_TYPE pv53 = ring_tail_load_u8(par5[3] + offset, slot_bytes);

    SIMD_TYPE sy0v0 = SIMD_EOR3_U8(p0v0, p1v0, pv40);
    SIMD_TYPE sy0v1 = SIMD_EOR3_U8(p0v1, p1v1, pv41);
    SIMD_TYPE sy0v2 = SIMD_EOR3_U8(p0v2, p1v2, pv42);
    SIMD_TYPE sy0v3 = SIMD_EOR3_U8(p0v3, p1v3, pv43);

    SIMD_TYPE sy1v0 = pv50;
    SIMD_TYPE sy1v1 = pv51;
    SIMD_TYPE sy1v2 = pv52;
    SIMD_TYPE sy1v3 = pv53;
    SIMD_TYPE sy1v4 = zero;

    switch (move0) {
    case 1:
        sy1v1 = SIMD_XOR(sy1v1, p0v0);
        sy1v2 = SIMD_XOR(sy1v2, p0v1);
        sy1v3 = SIMD_XOR(sy1v3, p0v2);
        sy1v4 = SIMD_XOR(sy1v4, p0v3);
        break;
    case 2:
        sy1v0 = SIMD_XOR(sy1v0, p0v3);
        sy1v2 = SIMD_XOR(sy1v2, p0v0);
        sy1v3 = SIMD_XOR(sy1v3, p0v1);
        sy1v4 = SIMD_XOR(sy1v4, p0v2);
        break;
    case 3:
        sy1v0 = SIMD_XOR(sy1v0, p0v2);
        sy1v1 = SIMD_XOR(sy1v1, p0v3);
        sy1v3 = SIMD_XOR(sy1v3, p0v0);
        sy1v4 = SIMD_XOR(sy1v4, p0v1);
        break;
    case 4:
        sy1v0 = SIMD_XOR(sy1v0, p0v1);
        sy1v1 = SIMD_XOR(sy1v1, p0v2);
        sy1v2 = SIMD_XOR(sy1v2, p0v3);
        sy1v4 = SIMD_XOR(sy1v4, p0v0);
        break;
    default:
        break;
    }

    switch (move1) {
    case 1:
        sy1v1 = SIMD_XOR(sy1v1, p1v0);
        sy1v2 = SIMD_XOR(sy1v2, p1v1);
        sy1v3 = SIMD_XOR(sy1v3, p1v2);
        sy1v4 = SIMD_XOR(sy1v4, p1v3);
        break;
    case 2:
        sy1v0 = SIMD_XOR(sy1v0, p1v3);
        sy1v2 = SIMD_XOR(sy1v2, p1v0);
        sy1v3 = SIMD_XOR(sy1v3, p1v1);
        sy1v4 = SIMD_XOR(sy1v4, p1v2);
        break;
    case 3:
        sy1v0 = SIMD_XOR(sy1v0, p1v2);
        sy1v1 = SIMD_XOR(sy1v1, p1v3);
        sy1v3 = SIMD_XOR(sy1v3, p1v0);
        sy1v4 = SIMD_XOR(sy1v4, p1v1);
        break;
    case 4:
        sy1v0 = SIMD_XOR(sy1v0, p1v1);
        sy1v1 = SIMD_XOR(sy1v1, p1v2);
        sy1v2 = SIMD_XOR(sy1v2, p1v3);
        sy1v4 = SIMD_XOR(sy1v4, p1v0);
        break;
    default:
        break;
    }

    sy1v0 = SIMD_XOR(sy1v0, sy1v4);
    sy1v1 = SIMD_XOR(sy1v1, sy1v4);
    sy1v2 = SIMD_XOR(sy1v2, sy1v4);
    sy1v3 = SIMD_XOR(sy1v3, sy1v4);

    SIMD_TYPE a0 = sy0v0;
    SIMD_TYPE a1 = sy0v1;
    SIMD_TYPE a2 = sy0v2;
    SIMD_TYPE a3 = sy0v3;
    SIMD_TYPE b0 = sy1v0;
    SIMD_TYPE b1 = sy1v1;
    SIMD_TYPE b2 = sy1v2;
    SIMD_TYPE b3 = sy1v3;
    SIMD_TYPE o00, o01, o02, o03, o04, o10, o11, o12, o13, o14;

    switch (pair_key) {
    case 1:
        o00 = SIMD_XOR(SIMD_EOR3_U8(a3, a1, b0), b3);
        o01 = SIMD_XOR(a2, b1);
        o02 = SIMD_XOR(SIMD_EOR3_U8(a0, a3, b2), b0);
        o03 = SIMD_EOR3_U8(a1, b3, b1);
        o04 = SIMD_EOR3_U8(a2, a0, b2);
        o10 = SIMD_EOR3_U8(a2, b0, b3);
        o11 = SIMD_EOR3_U8(a0, a3, b1);
        o12 = SIMD_EOR3_U8(a1, b2, b0);
        o13 = SIMD_XOR(SIMD_EOR3_U8(a2, a0, b3), b1);
        o14 = SIMD_EOR3_U8(a3, a1, b2);
        break;
    case 2:
        o00 = SIMD_EOR3_U8(a2, a1, b0);
        o01 = SIMD_XOR(SIMD_EOR3_U8(a3, a2, b1), b0);
        o02 = SIMD_EOR3_U8(a3, b2, b1);
        o03 = SIMD_EOR3_U8(a0, b3, b2);
        o04 = SIMD_EOR3_U8(a1, a0, b3);
        o10 = SIMD_XOR(a3, b0);
        o11 = SIMD_EOR3_U8(a0, b1, b0);
        o12 = SIMD_XOR(SIMD_EOR3_U8(a1, a0, b2), b1);
        o13 = SIMD_XOR(SIMD_EOR3_U8(a2, a1, b3), b2);
        o14 = SIMD_EOR3_U8(a3, a2, b3);
        break;
    case 3:
        o00 = SIMD_EOR3_U8(a3, b3, b2);
        o01 = SIMD_XOR(a0, b3);
        o02 = SIMD_EOR3_U8(a1, a0, b0);
        o03 = SIMD_XOR(SIMD_EOR3_U8(a2, a1, b1), b0);
        o04 = SIMD_XOR(SIMD_EOR3_U8(a3, a2, b2), b1);
        o10 = SIMD_XOR(SIMD_EOR3_U8(a2, a1, b3), b2);
        o11 = SIMD_EOR3_U8(a3, a2, b3);
        o12 = SIMD_XOR(a3, b0);
        o13 = SIMD_EOR3_U8(a0, b1, b0);
        o14 = SIMD_XOR(SIMD_EOR3_U8(a1, a0, b2), b1);
        break;
    case 6:
        o00 = SIMD_EOR3_U8(a3, a1, b1);
        o01 = SIMD_EOR3_U8(a2, b0, b2);
        o02 = SIMD_XOR(SIMD_EOR3_U8(a0, a3, b1), b3);
        o03 = SIMD_XOR(a1, b2);
        o04 = SIMD_XOR(SIMD_EOR3_U8(a2, a0, b3), b0);
        o10 = SIMD_XOR(a2, b1);
        o11 = SIMD_XOR(SIMD_EOR3_U8(a0, a3, b0), b2);
        o12 = SIMD_EOR3_U8(a1, b1, b3);
        o13 = SIMD_EOR3_U8(a2, a0, b2);
        o14 = SIMD_XOR(SIMD_EOR3_U8(a3, a1, b3), b0);
        break;
    case 7:
        o00 = SIMD_XOR(SIMD_EOR3_U8(a2, a1, b0), b1);
        o01 = SIMD_XOR(SIMD_EOR3_U8(a3, a2, b1), b2);
        o02 = SIMD_EOR3_U8(a3, b2, b3);
        o03 = SIMD_XOR(a0, b3);
        o04 = SIMD_EOR3_U8(a1, a0, b0);
        o10 = SIMD_EOR3_U8(a3, b0, b1);
        o11 = SIMD_EOR3_U8(a0, b1, b2);
        o12 = SIMD_XOR(SIMD_EOR3_U8(a1, a0, b2), b3);
        o13 = SIMD_EOR3_U8(a2, a1, b3);
        o14 = SIMD_EOR3_U8(a3, a2, b0);
        break;
    case 11:
        o00 = SIMD_XOR(SIMD_EOR3_U8(a3, a1, b0), b2);
        o01 = SIMD_EOR3_U8(a2, b1, b3);
        o02 = SIMD_EOR3_U8(a0, a3, b2);
        o03 = SIMD_EOR3_U8(a1, b3, b0);
        o04 = SIMD_EOR3_U8(a2, a0, b1);
        o10 = SIMD_EOR3_U8(a2, b0, b2);
        o11 = SIMD_XOR(SIMD_EOR3_U8(a0, a3, b1), b3);
        o12 = SIMD_XOR(a1, b2);
        o13 = SIMD_XOR(SIMD_EOR3_U8(a2, a0, b3), b0);
        o14 = SIMD_EOR3_U8(a3, a1, b1);
        break;
    default:
        return;
    }

    ring_tail_store_u8(dst0[0] + offset, SIMD_XOR(o00, o04), slot_bytes);
    ring_tail_store_u8(dst0[1] + offset, SIMD_XOR(o01, o04), slot_bytes);
    ring_tail_store_u8(dst0[2] + offset, SIMD_XOR(o02, o04), slot_bytes);
    ring_tail_store_u8(dst0[3] + offset, SIMD_XOR(o03, o04), slot_bytes);
    ring_tail_store_u8(dst1[0] + offset, SIMD_XOR(o10, o14), slot_bytes);
    ring_tail_store_u8(dst1[1] + offset, SIMD_XOR(o11, o14), slot_bytes);
    ring_tail_store_u8(dst1[2] + offset, SIMD_XOR(o12, o14), slot_bytes);
    ring_tail_store_u8(dst1[3] + offset, SIMD_XOR(o13, o14), slot_bytes);
}

static void recover_data_pair_fused(unsigned char *out0,
                                     unsigned char *out1,  unsigned char **shards, int missing0,
                                    int missing1, int node_size)
{
    const size_t slot_stride = (size_t)node_size < RING_SIMD_BATCH_BYTES
                                   ? (size_t)node_size / RING_DATA_SLOTS
                                   : SIMD_ITERATION_SIZE;

    // compute sy
    int present[2];
    int move0, move1;
    unsigned char *p0[4], *p1[4], *par4[4], *par5[4];

    // for (int i = 0; i < RING_DATA_SLOTS; i++) {
    //     if (i == missing0 || i == missing1) {
    //         continue;
    //     }
    //     present[present_count++] = i;
    // }
    uint32_t mask = 15;
    mask &= ~(1U << missing0);
    mask &= ~(1U << missing1); // 此时 mask 的二进制中正好只有 2 个比特为 1

    // 硬件加速：利用内置函数直接提取这两个比特的位置
    // __builtin_ctz 在 ARM64 下对应 RBIT + CLZ 指令，单周期执行
    present[0] = __builtin_ctz(mask);

    mask &= mask - 1;          // 消去最低位的 1，让原本的第二个 1 变成最低位
    present[1] = __builtin_ctz(mask);

    move0 = present[0] + 1;
    move1 = present[1] + 1;

    /* Four coefficient vectors in the first interleaved SIMD batch. */
    unsigned char *base_p0   = shards[present[0]];
    unsigned char *base_p1   = shards[present[1]];
    unsigned char *base_par4 = shards[4];
    unsigned char *base_par5 = shards[5];

    size_t offset1 = slot_stride;
    size_t offset2 = 2 * slot_stride;
    size_t offset3 = 3 * slot_stride;

    p0[0] = base_p0;           p1[0] = base_p1;           par4[0] = base_par4;           par5[0] = base_par5;
    p0[1] = base_p0 + offset1; p1[1] = base_p1 + offset1; par4[1] = base_par4 + offset1; par5[1] = base_par5 + offset1;
    p0[2] = base_p0 + offset2; p1[2] = base_p1 + offset2; par4[2] = base_par4 + offset2; par5[2] = base_par5 + offset2;
    p0[3] = base_p0 + offset3; p1[3] = base_p1 + offset3; par4[3] = base_par4 + offset3; par5[3] = base_par5 + offset3;

    //recover data prepare
    unsigned char *dst0[5];
    unsigned char *dst1[5];
    int pair_key = (missing0 << 2) | missing1;

    /* Recovery output uses the same interleaved batch layout. */
    dst0[0] = out0;
    dst1[0] = out1;

    dst0[1] = out0 + slot_stride;
    dst1[1] = out1 + slot_stride;

    dst0[2] = out0 + (2 * slot_stride);
    dst1[2] = out1 + (2 * slot_stride);

    dst0[3] = out0 + (3 * slot_stride);
    dst1[3] = out1 + (3 * slot_stride);

    size_t i = 0;
    size_t length = (size_t)node_size;

    for (; i + RING_TWO_BATCH_BYTES <= length;
         i += RING_TWO_BATCH_BYTES) {
        const size_t second_batch = RING_SIMD_BATCH_BYTES;
        SIMD_TYPE zero = SIMD_SET_ZERO;
        SIMD_TYPE p0v0 = SIMD_LOAD_R0(p0[0] + i);
        SIMD_TYPE p0v1 = SIMD_LOAD_R0(p0[1] + i);
        SIMD_TYPE p0v2 = SIMD_LOAD_R0(p0[2] + i);
        SIMD_TYPE p0v3 = SIMD_LOAD_R0(p0[3] + i);
        SIMD_TYPE p1v0 = SIMD_LOAD_R0(p1[0] + i);
        SIMD_TYPE p1v1 = SIMD_LOAD_R0(p1[1] + i);
        SIMD_TYPE p1v2 = SIMD_LOAD_R0(p1[2] + i);
        SIMD_TYPE p1v3 = SIMD_LOAD_R0(p1[3] + i);
        SIMD_TYPE pv40 = SIMD_LOAD_R0(par4[0] + i);
        SIMD_TYPE pv41 = SIMD_LOAD_R0(par4[1] + i);
        SIMD_TYPE pv42 = SIMD_LOAD_R0(par4[2] + i);
        SIMD_TYPE pv43 = SIMD_LOAD_R0(par4[3] + i);
        SIMD_TYPE pv50 = SIMD_LOAD_R0(par5[0] + i);
        SIMD_TYPE pv51 = SIMD_LOAD_R0(par5[1] + i);
        SIMD_TYPE pv52 = SIMD_LOAD_R0(par5[2] + i);
        SIMD_TYPE pv53 = SIMD_LOAD_R0(par5[3] + i);

        SIMD_TYPE qzero = SIMD_SET_ZERO;
        SIMD_TYPE qp0v0 = SIMD_LOAD_R0(p0[0] + i + second_batch);
        SIMD_TYPE qp0v1 = SIMD_LOAD_R0(p0[1] + i + second_batch);
        SIMD_TYPE qp0v2 = SIMD_LOAD_R0(p0[2] + i + second_batch);
        SIMD_TYPE qp0v3 = SIMD_LOAD_R0(p0[3] + i + second_batch);
        SIMD_TYPE qp1v0 = SIMD_LOAD_R0(p1[0] + i + second_batch);
        SIMD_TYPE qp1v1 = SIMD_LOAD_R0(p1[1] + i + second_batch);
        SIMD_TYPE qp1v2 = SIMD_LOAD_R0(p1[2] + i + second_batch);
        SIMD_TYPE qp1v3 = SIMD_LOAD_R0(p1[3] + i + second_batch);
        SIMD_TYPE qpv40 = SIMD_LOAD_R0(par4[0] + i + second_batch);
        SIMD_TYPE qpv41 = SIMD_LOAD_R0(par4[1] + i + second_batch);
        SIMD_TYPE qpv42 = SIMD_LOAD_R0(par4[2] + i + second_batch);
        SIMD_TYPE qpv43 = SIMD_LOAD_R0(par4[3] + i + second_batch);
        SIMD_TYPE qpv50 = SIMD_LOAD_R0(par5[0] + i + second_batch);
        SIMD_TYPE qpv51 = SIMD_LOAD_R0(par5[1] + i + second_batch);
        SIMD_TYPE qpv52 = SIMD_LOAD_R0(par5[2] + i + second_batch);
        SIMD_TYPE qpv53 = SIMD_LOAD_R0(par5[3] + i + second_batch);

        SIMD_TYPE sy0v0, sy0v1, sy0v2, sy0v3;
        SIMD_TYPE qsy0v0, qsy0v1, qsy0v2, qsy0v3;

        sy0v0 = SIMD_EOR3_U8(p0v0, p1v0, pv40);
        sy0v1 = SIMD_EOR3_U8(p0v1, p1v1, pv41);
        sy0v2 = SIMD_EOR3_U8(p0v2, p1v2, pv42);
        sy0v3 = SIMD_EOR3_U8(p0v3, p1v3, pv43);

        qsy0v0 = SIMD_EOR3_U8(qp0v0, qp1v0, qpv40);
        qsy0v1 = SIMD_EOR3_U8(qp0v1, qp1v1, qpv41);
        qsy0v2 = SIMD_EOR3_U8(qp0v2, qp1v2, qpv42);
        qsy0v3 = SIMD_EOR3_U8(qp0v3, qp1v3, qpv43);

        SIMD_TYPE sy1v0 = pv50;
        SIMD_TYPE sy1v1 = pv51;
        SIMD_TYPE sy1v2 = pv52;
        SIMD_TYPE sy1v3 = pv53;
        SIMD_TYPE sy1v4 = zero;

        SIMD_TYPE qsy1v0 = qpv50;
        SIMD_TYPE qsy1v1 = qpv51;
        SIMD_TYPE qsy1v2 = qpv52;
        SIMD_TYPE qsy1v3 = qpv53;
        SIMD_TYPE qsy1v4 = qzero;

        switch (move0) {
        case 1:
            sy1v1 = SIMD_XOR(sy1v1, p0v0);
            sy1v2 = SIMD_XOR(sy1v2, p0v1);
            sy1v3 = SIMD_XOR(sy1v3, p0v2);
            sy1v4 = SIMD_XOR(sy1v4, p0v3);

            qsy1v1 = SIMD_XOR(qsy1v1, qp0v0);
            qsy1v2 = SIMD_XOR(qsy1v2, qp0v1);
            qsy1v3 = SIMD_XOR(qsy1v3, qp0v2);
            qsy1v4 = SIMD_XOR(qsy1v4, qp0v3);
            break;
        case 2:
            sy1v0 = SIMD_XOR(sy1v0, p0v3);
            sy1v2 = SIMD_XOR(sy1v2, p0v0);
            sy1v3 = SIMD_XOR(sy1v3, p0v1);
            sy1v4 = SIMD_XOR(sy1v4, p0v2);

            qsy1v0 = SIMD_XOR(qsy1v0, qp0v3);
            qsy1v2 = SIMD_XOR(qsy1v2, qp0v0);
            qsy1v3 = SIMD_XOR(qsy1v3, qp0v1);
            qsy1v4 = SIMD_XOR(qsy1v4, qp0v2);
            break;
        case 3:
            sy1v0 = SIMD_XOR(sy1v0, p0v2);
            sy1v1 = SIMD_XOR(sy1v1, p0v3);
            sy1v3 = SIMD_XOR(sy1v3, p0v0);
            sy1v4 = SIMD_XOR(sy1v4, p0v1);

            qsy1v0 = SIMD_XOR(qsy1v0, qp0v2);
            qsy1v1 = SIMD_XOR(qsy1v1, qp0v3);
            qsy1v3 = SIMD_XOR(qsy1v3, qp0v0);
            qsy1v4 = SIMD_XOR(qsy1v4, qp0v1);
            break;
        case 4:
            sy1v0 = SIMD_XOR(sy1v0, p0v1);
            sy1v1 = SIMD_XOR(sy1v1, p0v2);
            sy1v2 = SIMD_XOR(sy1v2, p0v3);
            sy1v4 = SIMD_XOR(sy1v4, p0v0);

            qsy1v0 = SIMD_XOR(qsy1v0, qp0v1);
            qsy1v1 = SIMD_XOR(qsy1v1, qp0v2);
            qsy1v2 = SIMD_XOR(qsy1v2, qp0v3);
            qsy1v4 = SIMD_XOR(qsy1v4, qp0v0);
            break;
        default:
            break;
        }
        switch (move1) {
        case 1:
            sy1v1 = SIMD_XOR(sy1v1, p1v0);
            sy1v2 = SIMD_XOR(sy1v2, p1v1);
            sy1v3 = SIMD_XOR(sy1v3, p1v2);
            sy1v4 = SIMD_XOR(sy1v4, p1v3);

            qsy1v1 = SIMD_XOR(qsy1v1, qp1v0);
            qsy1v2 = SIMD_XOR(qsy1v2, qp1v1);
            qsy1v3 = SIMD_XOR(qsy1v3, qp1v2);
            qsy1v4 = SIMD_XOR(qsy1v4, qp1v3);
            break;
        case 2:
            sy1v0 = SIMD_XOR(sy1v0, p1v3);
            sy1v2 = SIMD_XOR(sy1v2, p1v0);
            sy1v3 = SIMD_XOR(sy1v3, p1v1);
            sy1v4 = SIMD_XOR(sy1v4, p1v2);

            qsy1v0 = SIMD_XOR(qsy1v0, qp1v3);
            qsy1v2 = SIMD_XOR(qsy1v2, qp1v0);
            qsy1v3 = SIMD_XOR(qsy1v3, qp1v1);
            qsy1v4 = SIMD_XOR(qsy1v4, qp1v2);
            break;
        case 3:
            sy1v0 = SIMD_XOR(sy1v0, p1v2);
            sy1v1 = SIMD_XOR(sy1v1, p1v3);
            sy1v3 = SIMD_XOR(sy1v3, p1v0);
            sy1v4 = SIMD_XOR(sy1v4, p1v1);

            qsy1v0 = SIMD_XOR(qsy1v0, qp1v2);
            qsy1v1 = SIMD_XOR(qsy1v1, qp1v3);
            qsy1v3 = SIMD_XOR(qsy1v3, qp1v0);
            qsy1v4 = SIMD_XOR(qsy1v4, qp1v1);
            break;
        case 4:
            sy1v0 = SIMD_XOR(sy1v0, p1v1);
            sy1v1 = SIMD_XOR(sy1v1, p1v2);
            sy1v2 = SIMD_XOR(sy1v2, p1v3);
            sy1v4 = SIMD_XOR(sy1v4, p1v0);

            qsy1v0 = SIMD_XOR(qsy1v0, qp1v1);
            qsy1v1 = SIMD_XOR(qsy1v1, qp1v2);
            qsy1v2 = SIMD_XOR(qsy1v2, qp1v3);
            qsy1v4 = SIMD_XOR(qsy1v4, qp1v0);
            break;
        default:
            break;
        }
        sy1v0 = SIMD_XOR(sy1v0,sy1v4);
        sy1v1 = SIMD_XOR(sy1v1,sy1v4);
        sy1v2 = SIMD_XOR(sy1v2,sy1v4);
        sy1v3 = SIMD_XOR(sy1v3,sy1v4);

        qsy1v0 = SIMD_XOR(qsy1v0,qsy1v4);
        qsy1v1 = SIMD_XOR(qsy1v1,qsy1v4);
        qsy1v2 = SIMD_XOR(qsy1v2,qsy1v4);
        qsy1v3 = SIMD_XOR(qsy1v3,qsy1v4);

        SIMD_TYPE a0 = sy0v0;
        SIMD_TYPE a1 = sy0v1;
        SIMD_TYPE a2 = sy0v2;
        SIMD_TYPE a3 = sy0v3;
        SIMD_TYPE b0 = sy1v0;
        SIMD_TYPE b1 = sy1v1;
        SIMD_TYPE b2 = sy1v2;
        SIMD_TYPE b3 = sy1v3;

        SIMD_TYPE qa0 = qsy0v0;
        SIMD_TYPE qa1 = qsy0v1;
        SIMD_TYPE qa2 = qsy0v2;
        SIMD_TYPE qa3 = qsy0v3;
        SIMD_TYPE qb0 = qsy1v0;
        SIMD_TYPE qb1 = qsy1v1;
        SIMD_TYPE qb2 = qsy1v2;
        SIMD_TYPE qb3 = qsy1v3;

        SIMD_TYPE o00, o01, o02, o03, o04, o10, o11, o12, o13, o14;

        SIMD_TYPE qo00, qo01, qo02, qo03, qo04, qo10, qo11, qo12, qo13, qo14;

    switch (pair_key) {
        case 1:
            o00 = SIMD_XOR(SIMD_EOR3_U8(a3, a1, b0), b3);
            o01 = SIMD_XOR(a2, b1);
            o02 = SIMD_XOR(SIMD_EOR3_U8(a0, a3, b2), b0);
            o03 = SIMD_EOR3_U8(a1, b3, b1);
            o04 = SIMD_EOR3_U8(a2, a0, b2);
            o10 = SIMD_EOR3_U8(a2, b0, b3);
            o11 = SIMD_EOR3_U8(a0, a3, b1);
            o12 = SIMD_EOR3_U8(a1, b2, b0);
            o13 = SIMD_XOR(SIMD_EOR3_U8(a2, a0, b3), b1);
            o14 = SIMD_EOR3_U8(a3, a1, b2);

            qo00 = SIMD_XOR(SIMD_EOR3_U8(qa3, qa1, qb0), qb3);
            qo01 = SIMD_XOR(qa2, qb1);
            qo02 = SIMD_XOR(SIMD_EOR3_U8(qa0, qa3, qb2), qb0);
            qo03 = SIMD_EOR3_U8(qa1, qb3, qb1);
            qo04 = SIMD_EOR3_U8(qa2, qa0, qb2);
            qo10 = SIMD_EOR3_U8(qa2, qb0, qb3);
            qo11 = SIMD_EOR3_U8(qa0, qa3, qb1);
            qo12 = SIMD_EOR3_U8(qa1, qb2, qb0);
            qo13 = SIMD_XOR(SIMD_EOR3_U8(qa2, qa0, qb3), qb1);
            qo14 = SIMD_EOR3_U8(qa3, qa1, qb2);
            break;
        case 2:
            o00 = SIMD_EOR3_U8(a2, a1, b0);
            o01 = SIMD_XOR(SIMD_EOR3_U8(a3, a2, b1), b0);
            o02 = SIMD_EOR3_U8(a3, b2, b1);
            o03 = SIMD_EOR3_U8(a0, b3, b2);
            o04 = SIMD_EOR3_U8(a1, a0, b3);
            o10 = SIMD_XOR(a3, b0);
            o11 = SIMD_EOR3_U8(a0, b1, b0);
            o12 = SIMD_XOR(SIMD_EOR3_U8(a1, a0, b2), b1);
            o13 = SIMD_XOR(SIMD_EOR3_U8(a2, a1, b3), b2);
            o14 = SIMD_EOR3_U8(a3, a2, b3);

            qo00 = SIMD_EOR3_U8(qa2, qa1, qb0);
            qo01 = SIMD_XOR(SIMD_EOR3_U8(qa3, qa2, qb1), qb0);
            qo02 = SIMD_EOR3_U8(qa3, qb2, qb1);
            qo03 = SIMD_EOR3_U8(qa0, qb3, qb2);
            qo04 = SIMD_EOR3_U8(qa1, qa0, qb3);
            qo10 = SIMD_XOR(qa3, qb0);
            qo11 = SIMD_EOR3_U8(qa0, qb1, qb0);
            qo12 = SIMD_XOR(SIMD_EOR3_U8(qa1, qa0, qb2), qb1);
            qo13 = SIMD_XOR(SIMD_EOR3_U8(qa2, qa1, qb3), qb2);
            qo14 = SIMD_EOR3_U8(qa3, qa2, qb3);
            break;
        case 3:
            o00 = SIMD_EOR3_U8(a3, b3, b2);
            o01 = SIMD_XOR(a0, b3);
            o02 = SIMD_EOR3_U8(a1, a0, b0);
            o03 = SIMD_XOR(SIMD_EOR3_U8(a2, a1, b1), b0);
            o04 = SIMD_XOR(SIMD_EOR3_U8(a3, a2, b2), b1);
            o10 = SIMD_XOR(SIMD_EOR3_U8(a2, a1, b3), b2);
            o11 = SIMD_EOR3_U8(a3, a2, b3);
            o12 = SIMD_XOR(a3, b0);
            o13 = SIMD_EOR3_U8(a0, b1, b0);
            o14 = SIMD_XOR(SIMD_EOR3_U8(a1, a0, b2), b1);

            qo00 = SIMD_EOR3_U8(qa3, qb3, qb2);
            qo01 = SIMD_XOR(qa0, qb3);
            qo02 = SIMD_EOR3_U8(qa1, qa0, qb0);
            qo03 = SIMD_XOR(SIMD_EOR3_U8(qa2, qa1, qb1), qb0);
            qo04 = SIMD_XOR(SIMD_EOR3_U8(qa3, qa2, qb2), qb1);
            qo10 = SIMD_XOR(SIMD_EOR3_U8(qa2, qa1, qb3), qb2);
            qo11 = SIMD_EOR3_U8(qa3, qa2, qb3);
            qo12 = SIMD_XOR(qa3, qb0);
            qo13 = SIMD_EOR3_U8(qa0, qb1, qb0);
            qo14 = SIMD_XOR(SIMD_EOR3_U8(qa1, qa0, qb2), qb1);
            break;
        case 6:
            o00 = SIMD_EOR3_U8(a3, a1, b1);
            o01 = SIMD_EOR3_U8(a2, b0, b2);
            o02 = SIMD_XOR(SIMD_EOR3_U8(a0, a3, b1), b3);
            o03 = SIMD_XOR(a1, b2);
            o04 = SIMD_XOR(SIMD_EOR3_U8(a2, a0, b3), b0);
            o10 = SIMD_XOR(a2, b1);
            o11 = SIMD_XOR(SIMD_EOR3_U8(a0, a3, b0), b2);
            o12 = SIMD_EOR3_U8(a1, b1, b3);
            o13 = SIMD_EOR3_U8(a2, a0, b2);
            o14 = SIMD_XOR(SIMD_EOR3_U8(a3, a1, b3), b0);

            qo00 = SIMD_EOR3_U8(qa3, qa1, qb1);
            qo01 = SIMD_EOR3_U8(qa2, qb0, qb2);
            qo02 = SIMD_XOR(SIMD_EOR3_U8(qa0, qa3, qb1), qb3);
            qo03 = SIMD_XOR(qa1, qb2);
            qo04 = SIMD_XOR(SIMD_EOR3_U8(qa2, qa0, qb3), qb0);
            qo10 = SIMD_XOR(qa2, qb1);
            qo11 = SIMD_XOR(SIMD_EOR3_U8(qa0, qa3, qb0), qb2);
            qo12 = SIMD_EOR3_U8(qa1, qb1, qb3);
            qo13 = SIMD_EOR3_U8(qa2, qa0, qb2);
            qo14 = SIMD_XOR(SIMD_EOR3_U8(qa3, qa1, qb3), qb0);
            break;
        case 7:
            o00 = SIMD_XOR(SIMD_EOR3_U8(a2, a1, b0), b1);
            o01 = SIMD_XOR(SIMD_EOR3_U8(a3, a2, b1), b2);
            o02 = SIMD_EOR3_U8(a3, b2, b3);
            o03 = SIMD_XOR(a0, b3);
            o04 = SIMD_EOR3_U8(a1, a0, b0);
            o10 = SIMD_EOR3_U8(a3, b0, b1);
            o11 = SIMD_EOR3_U8(a0, b1, b2);
            o12 = SIMD_XOR(SIMD_EOR3_U8(a1, a0, b2), b3);
            o13 = SIMD_EOR3_U8(a2, a1, b3);
            o14 = SIMD_EOR3_U8(a3, a2, b0);

            qo00 = SIMD_XOR(SIMD_EOR3_U8(qa2, qa1, qb0), qb1);
            qo01 = SIMD_XOR(SIMD_EOR3_U8(qa3, qa2, qb1), qb2);
            qo02 = SIMD_EOR3_U8(qa3, qb2, qb3);
            qo03 = SIMD_XOR(qa0, qb3);
            qo04 = SIMD_EOR3_U8(qa1, qa0, qb0);
            qo10 = SIMD_EOR3_U8(qa3, qb0, qb1);
            qo11 = SIMD_EOR3_U8(qa0, qb1, qb2);
            qo12 = SIMD_XOR(SIMD_EOR3_U8(qa1, qa0, qb2), qb3);
            qo13 = SIMD_EOR3_U8(qa2, qa1, qb3);
            qo14 = SIMD_EOR3_U8(qa3, qa2, qb0);
            break;
        case 11:
            o00 = SIMD_XOR(SIMD_EOR3_U8(a3, a1, b0), b2);
            o01 = SIMD_EOR3_U8(a2, b1, b3);
            o02 = SIMD_EOR3_U8(a0, a3, b2);
            o03 = SIMD_EOR3_U8(a1, b3, b0);
            o04 = SIMD_EOR3_U8(a2, a0, b1);
            o10 = SIMD_EOR3_U8(a2, b0, b2);
            o11 = SIMD_XOR(SIMD_EOR3_U8(a0, a3, b1), b3);
            o12 = SIMD_XOR(a1, b2);
            o13 = SIMD_XOR(SIMD_EOR3_U8(a2, a0, b3), b0);
            o14 = SIMD_EOR3_U8(a3, a1, b1);

            qo00 = SIMD_XOR(SIMD_EOR3_U8(qa3, qa1, qb0), qb2);
            qo01 = SIMD_EOR3_U8(qa2, qb1, qb3);
            qo02 = SIMD_EOR3_U8(qa0, qa3, qb2);
            qo03 = SIMD_EOR3_U8(qa1, qb3, qb0);
            qo04 = SIMD_EOR3_U8(qa2, qa0, qb1);
            qo10 = SIMD_EOR3_U8(qa2, qb0, qb2);
            qo11 = SIMD_XOR(SIMD_EOR3_U8(qa0, qa3, qb1), qb3);
            qo12 = SIMD_XOR(qa1, qb2);
            qo13 = SIMD_XOR(SIMD_EOR3_U8(qa2, qa0, qb3), qb0);
            qo14 = SIMD_EOR3_U8(qa3, qa1, qb1);
            break;
        default:
            return;
        }

        SIMD_STORE_R0(dst0[0] + i, SIMD_XOR(o00, o04));
        SIMD_STORE_R0(dst0[1] + i, SIMD_XOR(o01, o04));
        SIMD_STORE_R0(dst0[2] + i, SIMD_XOR(o02, o04));
        SIMD_STORE_R0(dst0[3] + i, SIMD_XOR(o03, o04));
        SIMD_STORE_R0(dst1[0] + i, SIMD_XOR(o10, o14));
        SIMD_STORE_R0(dst1[1] + i, SIMD_XOR(o11, o14));
        SIMD_STORE_R0(dst1[2] + i, SIMD_XOR(o12, o14));
        SIMD_STORE_R0(dst1[3] + i, SIMD_XOR(o13, o14));

        SIMD_STORE_R0(dst0[0] + i + second_batch, SIMD_XOR(qo00, qo04));
        SIMD_STORE_R0(dst0[1] + i + second_batch, SIMD_XOR(qo01, qo04));
        SIMD_STORE_R0(dst0[2] + i + second_batch, SIMD_XOR(qo02, qo04));
        SIMD_STORE_R0(dst0[3] + i + second_batch, SIMD_XOR(qo03, qo04));
        SIMD_STORE_R0(dst1[0] + i + second_batch, SIMD_XOR(qo10, qo14));
        SIMD_STORE_R0(dst1[1] + i + second_batch, SIMD_XOR(qo11, qo14));
        SIMD_STORE_R0(dst1[2] + i + second_batch, SIMD_XOR(qo12, qo14));
        SIMD_STORE_R0(dst1[3] + i + second_batch, SIMD_XOR(qo13, qo14));
    }

    if (length - i >= RING_SIMD_BATCH_BYTES) {
        recover_data_pair_tail(p0, p1, par4, par5, dst0, dst1,
                               move0, move1, pair_key, i,
                               SIMD_ITERATION_SIZE);
        i += RING_SIMD_BATCH_BYTES;
    }

    if (i < length) {
        const size_t slot_bytes = ring_tail_slot_bytes(length - i);
        for (int slot = 0; slot < RING_DATA_SLOTS; ++slot) {
            const size_t offset = i + (size_t)slot * slot_bytes;
            p0[slot] = base_p0 + offset;
            p1[slot] = base_p1 + offset;
            par4[slot] = base_par4 + offset;
            par5[slot] = base_par5 + offset;
            dst0[slot] = out0 + offset;
            dst1[slot] = out1 + offset;
        }
        recover_data_pair_tail(p0, p1, par4, par5, dst0, dst1,
                               move0, move1, pair_key, 0, slot_bytes);
    }
}

static inline void recover_data_single_tail(
        unsigned char *dst[4], const unsigned char *s0[4],
        const unsigned char *s1[4], const unsigned char *s2[4],
        const unsigned char *s3[4], size_t offset, size_t slot_bytes)
{
    SIMD_TYPE v00 = ring_tail_load_u8(s0[0] + offset, slot_bytes);
    SIMD_TYPE v10 = ring_tail_load_u8(s1[0] + offset, slot_bytes);
    SIMD_TYPE v20 = ring_tail_load_u8(s2[0] + offset, slot_bytes);
    SIMD_TYPE v30 = ring_tail_load_u8(s3[0] + offset, slot_bytes);
    SIMD_TYPE v01 = ring_tail_load_u8(s0[1] + offset, slot_bytes);
    SIMD_TYPE v11 = ring_tail_load_u8(s1[1] + offset, slot_bytes);
    SIMD_TYPE v21 = ring_tail_load_u8(s2[1] + offset, slot_bytes);
    SIMD_TYPE v31 = ring_tail_load_u8(s3[1] + offset, slot_bytes);
    SIMD_TYPE v02 = ring_tail_load_u8(s0[2] + offset, slot_bytes);
    SIMD_TYPE v12 = ring_tail_load_u8(s1[2] + offset, slot_bytes);
    SIMD_TYPE v22 = ring_tail_load_u8(s2[2] + offset, slot_bytes);
    SIMD_TYPE v32 = ring_tail_load_u8(s3[2] + offset, slot_bytes);
    SIMD_TYPE v03 = ring_tail_load_u8(s0[3] + offset, slot_bytes);
    SIMD_TYPE v13 = ring_tail_load_u8(s1[3] + offset, slot_bytes);
    SIMD_TYPE v23 = ring_tail_load_u8(s2[3] + offset, slot_bytes);
    SIMD_TYPE v33 = ring_tail_load_u8(s3[3] + offset, slot_bytes);

    SIMD_TYPE outv0 = SIMD_XOR(SIMD_EOR3_U8(v00, v10, v20), v30);
    SIMD_TYPE outv1 = SIMD_XOR(SIMD_EOR3_U8(v01, v11, v21), v31);
    SIMD_TYPE outv2 = SIMD_XOR(SIMD_EOR3_U8(v02, v12, v22), v32);
    SIMD_TYPE outv3 = SIMD_XOR(SIMD_EOR3_U8(v03, v13, v23), v33);

    ring_tail_store_u8(dst[0] + offset, outv0, slot_bytes);
    ring_tail_store_u8(dst[1] + offset, outv1, slot_bytes);
    ring_tail_store_u8(dst[2] + offset, outv2, slot_bytes);
    ring_tail_store_u8(dst[3] + offset, outv3, slot_bytes);
}

static void recover_data_single_fused( unsigned char *output,
                                      unsigned char **shards, int missing, int node_size)
{
    const size_t slot_stride = (size_t)node_size < RING_SIMD_BATCH_BYTES
                                   ? (size_t)node_size / RING_DATA_SLOTS
                                   : SIMD_ITERATION_SIZE;

    const unsigned char *s0[4], *s1[4], *s2[4], *s3[4];
    unsigned char *dst[4];

    /* Select the four surviving nodes without reading shards[missing]. */
    s0[0] = shards[(missing + 1) % 5];
    s1[0] = shards[(missing + 2) % 5];
    s2[0] = shards[(missing + 3) % 5];
    s3[0] = shards[(missing + 4) % 5];
    dst[0] = output;

    s0[1] = shards[(missing + 1) % 5] + slot_stride;
    s1[1] = shards[(missing + 2) % 5] + slot_stride;
    s2[1] = shards[(missing + 3) % 5] + slot_stride;
    s3[1] = shards[(missing + 4) % 5] + slot_stride;
    dst[1] = output + slot_stride;

    s0[2] = shards[(missing + 1) % 5] + 2 * slot_stride;
    s1[2] = shards[(missing + 2) % 5] + 2 * slot_stride;
    s2[2] = shards[(missing + 3) % 5] + 2 * slot_stride;
    s3[2] = shards[(missing + 4) % 5] + 2 * slot_stride;
    dst[2] = output + 2 * slot_stride;

    s0[3] = shards[(missing + 1) % 5] + 3 * slot_stride;
    s1[3] = shards[(missing + 2) % 5] + 3 * slot_stride;
    s2[3] = shards[(missing + 3) % 5] + 3 * slot_stride;
    s3[3] = shards[(missing + 4) % 5] + 3 * slot_stride;
    dst[3] = output + 3 * slot_stride;

    size_t i = 0;
    size_t length = (size_t)node_size;

    for (; i + RING_TWO_BATCH_BYTES <= length;
         i += RING_TWO_BATCH_BYTES) {
        SIMD_TYPE v00, v10, v20, v30, v01, v11, v21, v31, v02, v12, v22, v32, v03, v13, v23, v33;
        SIMD_TYPE outv0, outv1, outv2, outv3;

        SIMD_TYPE c00, c10, c20, c30, c01, c11, c21, c31, c02, c12, c22, c32, c03, c13, c23, c33;
        SIMD_TYPE outc0, outc1, outc2, outc3;
        v00 = SIMD_LOAD_R0(s0[0] + i);
        v10 = SIMD_LOAD_R0(s1[0] + i);
        v20 = SIMD_LOAD_R0(s2[0] + i);
        v30 = SIMD_LOAD_R0(s3[0] + i);
        v01 = SIMD_LOAD_R0(s0[1] + i);
        v11 = SIMD_LOAD_R0(s1[1] + i);
        v21 = SIMD_LOAD_R0(s2[1] + i);
        v31 = SIMD_LOAD_R0(s3[1] + i);
        v02 = SIMD_LOAD_R0(s0[2] + i);
        v12 = SIMD_LOAD_R0(s1[2] + i);
        v22 = SIMD_LOAD_R0(s2[2] + i);
        v32 = SIMD_LOAD_R0(s3[2] + i);
        v03 = SIMD_LOAD_R0(s0[3] + i);
        v13 = SIMD_LOAD_R0(s1[3] + i);
        v23 = SIMD_LOAD_R0(s2[3] + i);
        v33 = SIMD_LOAD_R0(s3[3] + i);

        c00 = SIMD_LOAD_R0(s0[0] + i + RING_SIMD_BATCH_BYTES);
        c10 = SIMD_LOAD_R0(s1[0] + i + RING_SIMD_BATCH_BYTES);
        c20 = SIMD_LOAD_R0(s2[0] + i + RING_SIMD_BATCH_BYTES);
        c30 = SIMD_LOAD_R0(s3[0] + i + RING_SIMD_BATCH_BYTES);
        c01 = SIMD_LOAD_R0(s0[1] + i + RING_SIMD_BATCH_BYTES);
        c11 = SIMD_LOAD_R0(s1[1] + i + RING_SIMD_BATCH_BYTES);
        c21 = SIMD_LOAD_R0(s2[1] + i + RING_SIMD_BATCH_BYTES);
        c31 = SIMD_LOAD_R0(s3[1] + i + RING_SIMD_BATCH_BYTES);
        c02 = SIMD_LOAD_R0(s0[2] + i + RING_SIMD_BATCH_BYTES);
        c12 = SIMD_LOAD_R0(s1[2] + i + RING_SIMD_BATCH_BYTES);
        c22 = SIMD_LOAD_R0(s2[2] + i + RING_SIMD_BATCH_BYTES);
        c32 = SIMD_LOAD_R0(s3[2] + i + RING_SIMD_BATCH_BYTES);
        c03 = SIMD_LOAD_R0(s0[3] + i + RING_SIMD_BATCH_BYTES);
        c13 = SIMD_LOAD_R0(s1[3] + i + RING_SIMD_BATCH_BYTES);
        c23 = SIMD_LOAD_R0(s2[3] + i + RING_SIMD_BATCH_BYTES);
        c33 = SIMD_LOAD_R0(s3[3] + i + RING_SIMD_BATCH_BYTES);

        outv0 = SIMD_XOR(SIMD_EOR3_U8(v00, v10, v20), v30);
        outv1 = SIMD_XOR(SIMD_EOR3_U8(v01, v11, v21), v31);
        outv2 = SIMD_XOR(SIMD_EOR3_U8(v02, v12, v22), v32);
        outv3 = SIMD_XOR(SIMD_EOR3_U8(v03, v13, v23), v33);

        SIMD_STORE_R0(dst[0] + i, outv0);
        SIMD_STORE_R0(dst[1] + i, outv1);
        SIMD_STORE_R0(dst[2] + i, outv2);
        SIMD_STORE_R0(dst[3] + i, outv3);

        outc0 = SIMD_XOR(SIMD_EOR3_U8(c00, c10, c20), c30);
        outc1 = SIMD_XOR(SIMD_EOR3_U8(c01, c11, c21), c31);
        outc2 = SIMD_XOR(SIMD_EOR3_U8(c02, c12, c22), c32);
        outc3 = SIMD_XOR(SIMD_EOR3_U8(c03, c13, c23), c33);

        SIMD_STORE_R0(dst[0] + i + RING_SIMD_BATCH_BYTES, outc0);
        SIMD_STORE_R0(dst[1] + i + RING_SIMD_BATCH_BYTES, outc1);
        SIMD_STORE_R0(dst[2] + i + RING_SIMD_BATCH_BYTES, outc2);
        SIMD_STORE_R0(dst[3] + i + RING_SIMD_BATCH_BYTES, outc3);
    }

    if (length - i >= RING_SIMD_BATCH_BYTES) {
        recover_data_single_tail(dst, s0, s1, s2, s3, i,
                                 SIMD_ITERATION_SIZE);
        i += RING_SIMD_BATCH_BYTES;
    }

    if (i < length) {
        const size_t slot_bytes = ring_tail_slot_bytes(length - i);
        for (int slot = 0; slot < RING_DATA_SLOTS; ++slot) {
            const size_t offset = i + (size_t)slot * slot_bytes;
            s0[slot] = shards[(missing + 1) % 5] + offset;
            s1[slot] = shards[(missing + 2) % 5] + offset;
            s2[slot] = shards[(missing + 3) % 5] + offset;
            s3[slot] = shards[(missing + 4) % 5] + offset;
            dst[slot] = output + offset;
        }
        recover_data_single_tail(dst, s0, s1, s2, s3, 0, slot_bytes);
    }
}


static inline void ring_encode_tail(const unsigned char *d[4][4],
                                    unsigned char *p4_dst[4],
                                    unsigned char *p5_dst[4], size_t offset, size_t slot_bytes)
{
    SIMD_TYPE d00 = ring_tail_load_u8(d[0][0] + offset, slot_bytes);
    SIMD_TYPE d01 = ring_tail_load_u8(d[0][1] + offset, slot_bytes);
    SIMD_TYPE d02 = ring_tail_load_u8(d[0][2] + offset, slot_bytes);
    SIMD_TYPE d03 = ring_tail_load_u8(d[0][3] + offset, slot_bytes);
    SIMD_TYPE d10 = ring_tail_load_u8(d[1][0] + offset, slot_bytes);
    SIMD_TYPE d11 = ring_tail_load_u8(d[1][1] + offset, slot_bytes);
    SIMD_TYPE d12 = ring_tail_load_u8(d[1][2] + offset, slot_bytes);
    SIMD_TYPE d13 = ring_tail_load_u8(d[1][3] + offset, slot_bytes);
    SIMD_TYPE d20 = ring_tail_load_u8(d[2][0] + offset, slot_bytes);
    SIMD_TYPE d21 = ring_tail_load_u8(d[2][1] + offset, slot_bytes);
    SIMD_TYPE d22 = ring_tail_load_u8(d[2][2] + offset, slot_bytes);
    SIMD_TYPE d23 = ring_tail_load_u8(d[2][3] + offset, slot_bytes);
    SIMD_TYPE d30 = ring_tail_load_u8(d[3][0] + offset, slot_bytes);
    SIMD_TYPE d31 = ring_tail_load_u8(d[3][1] + offset, slot_bytes);
    SIMD_TYPE d32 = ring_tail_load_u8(d[3][2] + offset, slot_bytes);
    SIMD_TYPE d33 = ring_tail_load_u8(d[3][3] + offset, slot_bytes);

    SIMD_TYPE p40 = SIMD_XOR(SIMD_EOR3_U8(d00, d10, d20), d30);
    SIMD_TYPE p41 = SIMD_XOR(SIMD_EOR3_U8(d01, d11, d21), d31);
    SIMD_TYPE p42 = SIMD_XOR(SIMD_EOR3_U8(d02, d12, d22), d32);
    SIMD_TYPE p43 = SIMD_XOR(SIMD_EOR3_U8(d03, d13, d23), d33);
    SIMD_TYPE common = SIMD_XOR(SIMD_EOR3_U8(d03, d12, d21), d30);

    SIMD_TYPE q0 = SIMD_XOR(SIMD_EOR3_U8(d13, d22, d31), common);
    SIMD_TYPE q1 = SIMD_XOR(SIMD_EOR3_U8(d00, d23, d32), common);
    SIMD_TYPE q2 = SIMD_XOR(SIMD_EOR3_U8(d01, d10, d33), common);
    SIMD_TYPE q3 = SIMD_XOR(SIMD_EOR3_U8(d02, d11, d20), common);

    ring_tail_store_u8(p4_dst[0] + offset, p40, slot_bytes);
    ring_tail_store_u8(p4_dst[1] + offset, p41, slot_bytes);
    ring_tail_store_u8(p4_dst[2] + offset, p42, slot_bytes);
    ring_tail_store_u8(p4_dst[3] + offset, p43, slot_bytes);
    ring_tail_store_u8(p5_dst[0] + offset, q0, slot_bytes);
    ring_tail_store_u8(p5_dst[1] + offset, q1, slot_bytes);
    ring_tail_store_u8(p5_dst[2] + offset, q2, slot_bytes);
    ring_tail_store_u8(p5_dst[3] + offset, q3, slot_bytes);
}


void ring_4_2_encode_data(int k, int node_size, unsigned char **data,
                     unsigned char **coding)
{
    const size_t slot_stride = (size_t)node_size < RING_SIMD_BATCH_BYTES
                                   ? (size_t)node_size / RING_DATA_SLOTS
                                   : SIMD_ITERATION_SIZE;

    const unsigned char *d[4][4];
    unsigned char *p4_dst[4];
    unsigned char *p5_dst[4];
    // s = 0
    d[0][0] = data[0];
    d[0][1] = data[0] + slot_stride;
    d[0][2] = data[0] + 2 * slot_stride;
    d[0][3] = data[0] + 3 * slot_stride;

    // s = 1
    d[1][0] = data[1];
    d[1][1] = data[1] + slot_stride;
    d[1][2] = data[1] + 2 * slot_stride;
    d[1][3] = data[1] + 3 * slot_stride;

    // s = 2
    d[2][0] = data[2];
    d[2][1] = data[2] + slot_stride;
    d[2][2] = data[2] + 2 * slot_stride;
    d[2][3] = data[2] + 3 * slot_stride;

    // s = 3
    d[3][0] = data[3];
    d[3][1] = data[3] + slot_stride;
    d[3][2] = data[3] + 2 * slot_stride;
    d[3][3] = data[3] + 3 * slot_stride;


    p4_dst[0] = coding[0];
    p5_dst[0] = coding[1];

    p4_dst[1] = coding[0] + slot_stride;
    p5_dst[1] = coding[1] + slot_stride;

    p4_dst[2] = coding[0] + 2 * slot_stride;
    p5_dst[2] = coding[1] + 2 * slot_stride;

    p4_dst[3] = coding[0] + 3 * slot_stride;
    p5_dst[3] = coding[1] + 3 * slot_stride;

    size_t i = 0;
    size_t length = (size_t)node_size;

    for (; i + RING_TWO_BATCH_BYTES <= length;
         i += RING_TWO_BATCH_BYTES) {
        SIMD_TYPE d00 = SIMD_LOAD_R0(d[0][0] + i);
        SIMD_TYPE d01 = SIMD_LOAD_R0(d[0][1] + i);
        SIMD_TYPE d02 = SIMD_LOAD_R0(d[0][2] + i);
        SIMD_TYPE d03 = SIMD_LOAD_R0(d[0][3] + i);
        SIMD_TYPE d10 = SIMD_LOAD_R0(d[1][0] + i);
        SIMD_TYPE d11 = SIMD_LOAD_R0(d[1][1] + i);
        SIMD_TYPE d12 = SIMD_LOAD_R0(d[1][2] + i);
        SIMD_TYPE d13 = SIMD_LOAD_R0(d[1][3] + i);
        SIMD_TYPE d20 = SIMD_LOAD_R0(d[2][0] + i);
        SIMD_TYPE d21 = SIMD_LOAD_R0(d[2][1] + i);
        SIMD_TYPE d22 = SIMD_LOAD_R0(d[2][2] + i);
        SIMD_TYPE d23 = SIMD_LOAD_R0(d[2][3] + i);
        SIMD_TYPE d30 = SIMD_LOAD_R0(d[3][0] + i);
        SIMD_TYPE d31 = SIMD_LOAD_R0(d[3][1] + i);
        SIMD_TYPE d32 = SIMD_LOAD_R0(d[3][2] + i);
        SIMD_TYPE d33 = SIMD_LOAD_R0(d[3][3] + i);

        SIMD_TYPE e00 = SIMD_LOAD_R0(d[0][0] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e01 = SIMD_LOAD_R0(d[0][1] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e02 = SIMD_LOAD_R0(d[0][2] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e03 = SIMD_LOAD_R0(d[0][3] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e10 = SIMD_LOAD_R0(d[1][0] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e11 = SIMD_LOAD_R0(d[1][1] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e12 = SIMD_LOAD_R0(d[1][2] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e13 = SIMD_LOAD_R0(d[1][3] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e20 = SIMD_LOAD_R0(d[2][0] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e21 = SIMD_LOAD_R0(d[2][1] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e22 = SIMD_LOAD_R0(d[2][2] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e23 = SIMD_LOAD_R0(d[2][3] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e30 = SIMD_LOAD_R0(d[3][0] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e31 = SIMD_LOAD_R0(d[3][1] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e32 = SIMD_LOAD_R0(d[3][2] + i + RING_SIMD_BATCH_BYTES);
        SIMD_TYPE e33 = SIMD_LOAD_R0(d[3][3] + i + RING_SIMD_BATCH_BYTES);

        SIMD_TYPE p40, p41, p42, p43, common, q0, q1, q2, q3;
        SIMD_TYPE g40, g41, g42, g43, gcommon, g0, g1, g2, g3;

        p40 = SIMD_XOR(SIMD_EOR3_U8(d00, d10, d20), d30);
        p41 = SIMD_XOR(SIMD_EOR3_U8(d01, d11, d21), d31);
        p42 = SIMD_XOR(SIMD_EOR3_U8(d02, d12, d22), d32);
        p43 = SIMD_XOR(SIMD_EOR3_U8(d03, d13, d23), d33);
        common = SIMD_XOR(SIMD_EOR3_U8(d03, d12, d21), d30);

        q0 = SIMD_XOR(SIMD_EOR3_U8(d13, d22, d31), common);
        q1 = SIMD_XOR(SIMD_EOR3_U8(d00, d23, d32), common);
        q2 = SIMD_XOR(SIMD_EOR3_U8(d01, d10, d33), common);
        q3 = SIMD_XOR(SIMD_EOR3_U8(d02, d11, d20), common);

        g40 = SIMD_XOR(SIMD_EOR3_U8(e00, e10, e20), e30);
        g41 = SIMD_XOR(SIMD_EOR3_U8(e01, e11, e21), e31);
        g42 = SIMD_XOR(SIMD_EOR3_U8(e02, e12, e22), e32);
        g43 = SIMD_XOR(SIMD_EOR3_U8(e03, e13, e23), e33);
        gcommon = SIMD_XOR(SIMD_EOR3_U8(e03, e12, e21), e30);

        g0 = SIMD_XOR(SIMD_EOR3_U8(e13, e22, e31), gcommon);
        g1 = SIMD_XOR(SIMD_EOR3_U8(e00, e23, e32), gcommon);
        g2 = SIMD_XOR(SIMD_EOR3_U8(e01, e10, e33), gcommon);
        g3 = SIMD_XOR(SIMD_EOR3_U8(e02, e11, e20), gcommon);

        SIMD_STORE_R0(p4_dst[0] + i, p40);
        SIMD_STORE_R0(p4_dst[1] + i, p41);
        SIMD_STORE_R0(p4_dst[2] + i, p42);
        SIMD_STORE_R0(p4_dst[3] + i, p43);
        SIMD_STORE_R0(p5_dst[0] + i, q0);
        SIMD_STORE_R0(p5_dst[1] + i, q1);
        SIMD_STORE_R0(p5_dst[2] + i, q2);
        SIMD_STORE_R0(p5_dst[3] + i, q3);

        SIMD_STORE_R0(p4_dst[0] + i + RING_SIMD_BATCH_BYTES, g40);
        SIMD_STORE_R0(p4_dst[1] + i + RING_SIMD_BATCH_BYTES, g41);
        SIMD_STORE_R0(p4_dst[2] + i + RING_SIMD_BATCH_BYTES, g42);
        SIMD_STORE_R0(p4_dst[3] + i + RING_SIMD_BATCH_BYTES, g43);
        SIMD_STORE_R0(p5_dst[0] + i + RING_SIMD_BATCH_BYTES, g0);
        SIMD_STORE_R0(p5_dst[1] + i + RING_SIMD_BATCH_BYTES, g1);
        SIMD_STORE_R0(p5_dst[2] + i + RING_SIMD_BATCH_BYTES, g2);
        SIMD_STORE_R0(p5_dst[3] + i + RING_SIMD_BATCH_BYTES, g3);
    }

    if (length - i >= RING_SIMD_BATCH_BYTES) {
        ring_encode_tail(d, p4_dst, p5_dst, i, SIMD_ITERATION_SIZE);
        i += RING_SIMD_BATCH_BYTES;
    }

    if (i < length) {
        const size_t slot_bytes = ring_tail_slot_bytes(length - i);
        for (int slot = 0; slot < RING_DATA_SLOTS; ++slot) {
            const size_t offset = i + (size_t)slot * slot_bytes;
            for (int node = 0; node < RING_DATA_SLOTS; ++node)
                d[node][slot] = data[node] + offset;
            p4_dst[slot] = coding[0] + offset;
            p5_dst[slot] = coding[1] + offset;
        }
        ring_encode_tail(d, p4_dst, p5_dst, 0, slot_bytes);
    }

}

void ring_4_2_decode_data(int node_size, unsigned char **buffs,
                     unsigned char *src_err_list, int nerrs, unsigned char **temp_buffs)
{
    int a[2];
    a[0]=0;
    a[1]=1;

    if (nerrs == 2 && src_err_list[0]> src_err_list[1]) {
        int temp = src_err_list[0];
        src_err_list[0] = src_err_list[1];
        src_err_list[1] = temp;
        a[0]=1;
        a[1]=0;
    }

    if (nerrs == 1) {
        recover_data_single_fused(temp_buffs[0], buffs,
                                        src_err_list[0], node_size);

    } else if (nerrs == 2) {
        recover_data_pair_fused(temp_buffs[a[0]], temp_buffs[a[1]],
                                buffs, src_err_list[0], src_err_list[1],
                                node_size);
    }
}