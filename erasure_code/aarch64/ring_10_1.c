#include <stddef.h>
#include "erasure_code.h"
#include "simd_opt.h"
#include "test.h"
#include "ring_internal.h"

static inline __attribute__((always_inline)) SIMD_TYPE
xor_ten_vectors(SIMD_TYPE v0, SIMD_TYPE v1, SIMD_TYPE v2, SIMD_TYPE v3, SIMD_TYPE v4, SIMD_TYPE v5,
                SIMD_TYPE v6, SIMD_TYPE v7, SIMD_TYPE v8, SIMD_TYPE v9)
{
        SIMD_TYPE group0 = SIMD_EOR3_U8(v0, v1, v2);
        SIMD_TYPE group1 = SIMD_EOR3_U8(v3, v4, v5);
        SIMD_TYPE group2 = SIMD_EOR3_U8(v6, v7, v8);

        return SIMD_XOR(SIMD_EOR3_U8(group0, group1, group2), v9);
}
enum {
        RING_10_1_UNROLL = 8,
};

#define RING_10_1_BLOCK_BYTES      RING_10_1_UNROLL *SIMD_ITERATION_SIZE
static void
recover_data_single_fused(unsigned char *output, unsigned char **shards, int missing, int node_size)
{
        const unsigned char *p0 = shards[10];
        const unsigned char *s[10] = { shards[0], shards[1], shards[2], shards[3], shards[4],
                                       shards[5], shards[6], shards[7], shards[8], shards[9] };
        size_t offset = 0;
        size_t length = (size_t) node_size;
        s[missing] = p0;
        /* XOR is commutative: replace the missing data pointer with P0. */
        for (; offset + RING_10_1_BLOCK_BYTES <= length; offset += RING_10_1_BLOCK_BYTES) {
                SIMD_TYPE acc0 = SIMD_LOAD_R0(s[0] + offset + 0 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc1 = SIMD_LOAD_R0(s[0] + offset + 1 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc2 = SIMD_LOAD_R0(s[0] + offset + 2 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc3 = SIMD_LOAD_R0(s[0] + offset + 3 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc4 = SIMD_LOAD_R0(s[0] + offset + 4 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc5 = SIMD_LOAD_R0(s[0] + offset + 5 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc6 = SIMD_LOAD_R0(s[0] + offset + 6 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc7 = SIMD_LOAD_R0(s[0] + offset + 7 * SIMD_ITERATION_SIZE);

                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(s[1] + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(s[1] + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left2 = SIMD_LOAD_R0(s[1] + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left3 = SIMD_LOAD_R0(s[1] + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left4 = SIMD_LOAD_R0(s[1] + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left5 = SIMD_LOAD_R0(s[1] + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left6 = SIMD_LOAD_R0(s[1] + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left7 = SIMD_LOAD_R0(s[1] + offset + 7 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(s[2] + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(s[2] + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right2 = SIMD_LOAD_R0(s[2] + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right3 = SIMD_LOAD_R0(s[2] + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right4 = SIMD_LOAD_R0(s[2] + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right5 = SIMD_LOAD_R0(s[2] + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right6 = SIMD_LOAD_R0(s[2] + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right7 = SIMD_LOAD_R0(s[2] + offset + 7 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                        acc2 = SIMD_EOR3_U8(acc2, left2, right2);
                        acc3 = SIMD_EOR3_U8(acc3, left3, right3);
                        acc4 = SIMD_EOR3_U8(acc4, left4, right4);
                        acc5 = SIMD_EOR3_U8(acc5, left5, right5);
                        acc6 = SIMD_EOR3_U8(acc6, left6, right6);
                        acc7 = SIMD_EOR3_U8(acc7, left7, right7);
                }

                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(s[3] + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(s[3] + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left2 = SIMD_LOAD_R0(s[3] + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left3 = SIMD_LOAD_R0(s[3] + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left4 = SIMD_LOAD_R0(s[3] + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left5 = SIMD_LOAD_R0(s[3] + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left6 = SIMD_LOAD_R0(s[3] + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left7 = SIMD_LOAD_R0(s[3] + offset + 7 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(s[4] + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(s[4] + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right2 = SIMD_LOAD_R0(s[4] + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right3 = SIMD_LOAD_R0(s[4] + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right4 = SIMD_LOAD_R0(s[4] + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right5 = SIMD_LOAD_R0(s[4] + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right6 = SIMD_LOAD_R0(s[4] + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right7 = SIMD_LOAD_R0(s[4] + offset + 7 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                        acc2 = SIMD_EOR3_U8(acc2, left2, right2);
                        acc3 = SIMD_EOR3_U8(acc3, left3, right3);
                        acc4 = SIMD_EOR3_U8(acc4, left4, right4);
                        acc5 = SIMD_EOR3_U8(acc5, left5, right5);
                        acc6 = SIMD_EOR3_U8(acc6, left6, right6);
                        acc7 = SIMD_EOR3_U8(acc7, left7, right7);
                }

                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(s[5] + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(s[5] + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left2 = SIMD_LOAD_R0(s[5] + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left3 = SIMD_LOAD_R0(s[5] + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left4 = SIMD_LOAD_R0(s[5] + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left5 = SIMD_LOAD_R0(s[5] + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left6 = SIMD_LOAD_R0(s[5] + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left7 = SIMD_LOAD_R0(s[5] + offset + 7 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(s[6] + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(s[6] + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right2 = SIMD_LOAD_R0(s[6] + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right3 = SIMD_LOAD_R0(s[6] + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right4 = SIMD_LOAD_R0(s[6] + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right5 = SIMD_LOAD_R0(s[6] + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right6 = SIMD_LOAD_R0(s[6] + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right7 = SIMD_LOAD_R0(s[6] + offset + 7 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                        acc2 = SIMD_EOR3_U8(acc2, left2, right2);
                        acc3 = SIMD_EOR3_U8(acc3, left3, right3);
                        acc4 = SIMD_EOR3_U8(acc4, left4, right4);
                        acc5 = SIMD_EOR3_U8(acc5, left5, right5);
                        acc6 = SIMD_EOR3_U8(acc6, left6, right6);
                        acc7 = SIMD_EOR3_U8(acc7, left7, right7);
                }

                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(s[7] + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(s[7] + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left2 = SIMD_LOAD_R0(s[7] + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left3 = SIMD_LOAD_R0(s[7] + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left4 = SIMD_LOAD_R0(s[7] + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left5 = SIMD_LOAD_R0(s[7] + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left6 = SIMD_LOAD_R0(s[7] + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left7 = SIMD_LOAD_R0(s[7] + offset + 7 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(s[8] + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(s[8] + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right2 = SIMD_LOAD_R0(s[8] + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right3 = SIMD_LOAD_R0(s[8] + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right4 = SIMD_LOAD_R0(s[8] + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right5 = SIMD_LOAD_R0(s[8] + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right6 = SIMD_LOAD_R0(s[8] + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right7 = SIMD_LOAD_R0(s[8] + offset + 7 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                        acc2 = SIMD_EOR3_U8(acc2, left2, right2);
                        acc3 = SIMD_EOR3_U8(acc3, left3, right3);
                        acc4 = SIMD_EOR3_U8(acc4, left4, right4);
                        acc5 = SIMD_EOR3_U8(acc5, left5, right5);
                        acc6 = SIMD_EOR3_U8(acc6, left6, right6);
                        acc7 = SIMD_EOR3_U8(acc7, left7, right7);
                }

                acc0 = SIMD_XOR(acc0, SIMD_LOAD_R0(s[9] + offset + 0 * SIMD_ITERATION_SIZE));
                acc1 = SIMD_XOR(acc1, SIMD_LOAD_R0(s[9] + offset + 1 * SIMD_ITERATION_SIZE));
                acc2 = SIMD_XOR(acc2, SIMD_LOAD_R0(s[9] + offset + 2 * SIMD_ITERATION_SIZE));
                acc3 = SIMD_XOR(acc3, SIMD_LOAD_R0(s[9] + offset + 3 * SIMD_ITERATION_SIZE));
                acc4 = SIMD_XOR(acc4, SIMD_LOAD_R0(s[9] + offset + 4 * SIMD_ITERATION_SIZE));
                acc5 = SIMD_XOR(acc5, SIMD_LOAD_R0(s[9] + offset + 5 * SIMD_ITERATION_SIZE));
                acc6 = SIMD_XOR(acc6, SIMD_LOAD_R0(s[9] + offset + 6 * SIMD_ITERATION_SIZE));
                acc7 = SIMD_XOR(acc7, SIMD_LOAD_R0(s[9] + offset + 7 * SIMD_ITERATION_SIZE));

                SIMD_STORE_R0(output + offset + 0 * SIMD_ITERATION_SIZE, acc0);
                SIMD_STORE_R0(output + offset + 1 * SIMD_ITERATION_SIZE, acc1);
                SIMD_STORE_R0(output + offset + 2 * SIMD_ITERATION_SIZE, acc2);
                SIMD_STORE_R0(output + offset + 3 * SIMD_ITERATION_SIZE, acc3);
                SIMD_STORE_R0(output + offset + 4 * SIMD_ITERATION_SIZE, acc4);
                SIMD_STORE_R0(output + offset + 5 * SIMD_ITERATION_SIZE, acc5);
                SIMD_STORE_R0(output + offset + 6 * SIMD_ITERATION_SIZE, acc6);
                SIMD_STORE_R0(output + offset + 7 * SIMD_ITERATION_SIZE, acc7);
        }

        for (; offset < length; offset += SIMD_ITERATION_SIZE) {
                SIMD_TYPE v0 = SIMD_LOAD_R0(s[0] + offset);
                SIMD_TYPE v1 = SIMD_LOAD_R0(s[1] + offset);
                SIMD_TYPE v2 = SIMD_LOAD_R0(s[2] + offset);
                SIMD_TYPE v3 = SIMD_LOAD_R0(s[3] + offset);
                SIMD_TYPE v4 = SIMD_LOAD_R0(s[4] + offset);
                SIMD_TYPE v5 = SIMD_LOAD_R0(s[5] + offset);
                SIMD_TYPE v6 = SIMD_LOAD_R0(s[6] + offset);
                SIMD_TYPE v7 = SIMD_LOAD_R0(s[7] + offset);
                SIMD_TYPE v8 = SIMD_LOAD_R0(s[8] + offset);
                SIMD_TYPE v9 = SIMD_LOAD_R0(s[9] + offset);

                SIMD_STORE_R0(output + offset,
                              xor_ten_vectors(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9));
        }
}

void
ring_10_1_decode_data(int len, int m, int k, int node_size, unsigned char **buffs,
                      unsigned char *src_err_list, int nerrs, unsigned char **temp_buffs)
{
        recover_data_single_fused(temp_buffs[0], buffs, src_err_list[0], node_size);
}

static inline __attribute__((always_inline)) void
ring10_xor_kernel(int node_size, unsigned char *const *sources,
                  unsigned char *output)
{
        if (node_size <= 0)
                return;
        const unsigned char *d0 = sources[0];
        const unsigned char *d1 = sources[1];
        const unsigned char *d2 = sources[2];
        const unsigned char *d3 = sources[3];
        const unsigned char *d4 = sources[4];
        const unsigned char *d5 = sources[5];
        const unsigned char *d6 = sources[6];
        const unsigned char *d7 = sources[7];
        const unsigned char *d8 = sources[8];
        const unsigned char *d9 = sources[9];
        const size_t length = (size_t) node_size;
        const size_t step = (size_t) SIMD_ITERATION_SIZE;
        const size_t block = 8 * step;
        size_t offset = 0;

        for (; offset + block <= length; offset += block) {
                SIMD_TYPE acc0 = SIMD_LOAD_R0(d0 + offset + 0 * step);
                SIMD_TYPE acc1 = SIMD_LOAD_R0(d0 + offset + 1 * step);
                SIMD_TYPE acc2 = SIMD_LOAD_R0(d0 + offset + 2 * step);
                SIMD_TYPE acc3 = SIMD_LOAD_R0(d0 + offset + 3 * step);
                SIMD_TYPE acc4 = SIMD_LOAD_R0(d0 + offset + 4 * step);
                SIMD_TYPE acc5 = SIMD_LOAD_R0(d0 + offset + 5 * step);
                SIMD_TYPE acc6 = SIMD_LOAD_R0(d0 + offset + 6 * step);
                SIMD_TYPE acc7 = SIMD_LOAD_R0(d0 + offset + 7 * step);

                /* Incorporate sources 1 and 2 into all eight chains. */
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d1 + offset + 0 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d1 + offset + 1 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d2 + offset + 0 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d2 + offset + 1 * step);
                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d1 + offset + 2 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d1 + offset + 3 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d2 + offset + 2 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d2 + offset + 3 * step);
                        acc2 = SIMD_EOR3_U8(acc2, left0, right0);
                        acc3 = SIMD_EOR3_U8(acc3, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d1 + offset + 4 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d1 + offset + 5 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d2 + offset + 4 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d2 + offset + 5 * step);
                        acc4 = SIMD_EOR3_U8(acc4, left0, right0);
                        acc5 = SIMD_EOR3_U8(acc5, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d1 + offset + 6 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d1 + offset + 7 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d2 + offset + 6 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d2 + offset + 7 * step);
                        acc6 = SIMD_EOR3_U8(acc6, left0, right0);
                        acc7 = SIMD_EOR3_U8(acc7, left1, right1);
                }

                /* Incorporate sources 3 and 4 into all eight chains. */
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d3 + offset + 0 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d3 + offset + 1 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d4 + offset + 0 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d4 + offset + 1 * step);
                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d3 + offset + 2 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d3 + offset + 3 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d4 + offset + 2 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d4 + offset + 3 * step);
                        acc2 = SIMD_EOR3_U8(acc2, left0, right0);
                        acc3 = SIMD_EOR3_U8(acc3, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d3 + offset + 4 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d3 + offset + 5 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d4 + offset + 4 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d4 + offset + 5 * step);
                        acc4 = SIMD_EOR3_U8(acc4, left0, right0);
                        acc5 = SIMD_EOR3_U8(acc5, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d3 + offset + 6 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d3 + offset + 7 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d4 + offset + 6 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d4 + offset + 7 * step);
                        acc6 = SIMD_EOR3_U8(acc6, left0, right0);
                        acc7 = SIMD_EOR3_U8(acc7, left1, right1);
                }

                /* Incorporate sources 5 and 6 into all eight chains. */
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d5 + offset + 0 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d5 + offset + 1 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d6 + offset + 0 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d6 + offset + 1 * step);
                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d5 + offset + 2 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d5 + offset + 3 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d6 + offset + 2 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d6 + offset + 3 * step);
                        acc2 = SIMD_EOR3_U8(acc2, left0, right0);
                        acc3 = SIMD_EOR3_U8(acc3, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d5 + offset + 4 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d5 + offset + 5 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d6 + offset + 4 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d6 + offset + 5 * step);
                        acc4 = SIMD_EOR3_U8(acc4, left0, right0);
                        acc5 = SIMD_EOR3_U8(acc5, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d5 + offset + 6 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d5 + offset + 7 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d6 + offset + 6 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d6 + offset + 7 * step);
                        acc6 = SIMD_EOR3_U8(acc6, left0, right0);
                        acc7 = SIMD_EOR3_U8(acc7, left1, right1);
                }

                /* Incorporate sources 7 and 8 into all eight chains. */
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d7 + offset + 0 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d7 + offset + 1 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d8 + offset + 0 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d8 + offset + 1 * step);
                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d7 + offset + 2 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d7 + offset + 3 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d8 + offset + 2 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d8 + offset + 3 * step);
                        acc2 = SIMD_EOR3_U8(acc2, left0, right0);
                        acc3 = SIMD_EOR3_U8(acc3, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d7 + offset + 4 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d7 + offset + 5 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d8 + offset + 4 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d8 + offset + 5 * step);
                        acc4 = SIMD_EOR3_U8(acc4, left0, right0);
                        acc5 = SIMD_EOR3_U8(acc5, left1, right1);
                }
                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d7 + offset + 6 * step);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d7 + offset + 7 * step);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d8 + offset + 6 * step);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d8 + offset + 7 * step);
                        acc6 = SIMD_EOR3_U8(acc6, left0, right0);
                        acc7 = SIMD_EOR3_U8(acc7, left1, right1);
                }

                /* Final source: retain the baseline XOR operand order. */
                acc0 = SIMD_XOR(acc0, SIMD_LOAD_R0(d9 + offset + 0 * step));
                acc1 = SIMD_XOR(acc1, SIMD_LOAD_R0(d9 + offset + 1 * step));
                acc2 = SIMD_XOR(acc2, SIMD_LOAD_R0(d9 + offset + 2 * step));
                acc3 = SIMD_XOR(acc3, SIMD_LOAD_R0(d9 + offset + 3 * step));
                acc4 = SIMD_XOR(acc4, SIMD_LOAD_R0(d9 + offset + 4 * step));
                acc5 = SIMD_XOR(acc5, SIMD_LOAD_R0(d9 + offset + 5 * step));
                acc6 = SIMD_XOR(acc6, SIMD_LOAD_R0(d9 + offset + 6 * step));
                acc7 = SIMD_XOR(acc7, SIMD_LOAD_R0(d9 + offset + 7 * step));

                SIMD_STORE_R0(output + offset + 0 * step, acc0);
                SIMD_STORE_R0(output + offset + 1 * step, acc1);
                SIMD_STORE_R0(output + offset + 2 * step, acc2);
                SIMD_STORE_R0(output + offset + 3 * step, acc3);
                SIMD_STORE_R0(output + offset + 4 * step, acc4);
                SIMD_STORE_R0(output + offset + 5 * step, acc5);
                SIMD_STORE_R0(output + offset + 6 * step, acc6);
                SIMD_STORE_R0(output + offset + 7 * step, acc7);
        }

        /* Full-vector tail follows the same accumulation order. */
        for (; offset + step <= length; offset += step) {
                SIMD_TYPE acc = SIMD_LOAD_R0(d0 + offset);
                acc = SIMD_EOR3_U8(acc, SIMD_LOAD_R0(d1 + offset),
                                  SIMD_LOAD_R0(d2 + offset));
                acc = SIMD_EOR3_U8(acc, SIMD_LOAD_R0(d3 + offset),
                                  SIMD_LOAD_R0(d4 + offset));
                acc = SIMD_EOR3_U8(acc, SIMD_LOAD_R0(d5 + offset),
                                  SIMD_LOAD_R0(d6 + offset));
                acc = SIMD_EOR3_U8(acc, SIMD_LOAD_R0(d7 + offset),
                                  SIMD_LOAD_R0(d8 + offset));
                acc = SIMD_XOR(acc, SIMD_LOAD_R0(d9 + offset));
                SIMD_STORE_R0(output + offset, acc);
        }

        /* Byte tail avoids over-reading sizes not divisible by the SIMD step. */
        for (; offset < length; ++offset)
                output[offset] = d0[offset] ^ d1[offset] ^ d2[offset] ^ d3[offset] ^
                                 d4[offset] ^ d5[offset] ^ d6[offset] ^ d7[offset] ^
                                 d8[offset] ^ d9[offset];
}

void
ring_10_1_encode_data(int len, int k, int rows, int node_size, unsigned char **data,
                      unsigned char **coding)
{
#if defined(__ARM_FEATURE_SVE) 
ring10_xor_kernel(node_size, data, coding[0]);
#else
        const unsigned char *d0 = data[0];
        const unsigned char *d1 = data[1];
        const unsigned char *d2 = data[2];
        const unsigned char *d3 = data[3];
        const unsigned char *d4 = data[4];
        const unsigned char *d5 = data[5];
        const unsigned char *d6 = data[6];
        const unsigned char *d7 = data[7];
        const unsigned char *d8 = data[8];
        const unsigned char *d9 = data[9];
        unsigned char *p0 = coding[0];
        size_t offset = 0;
        size_t length = (size_t) node_size;
        const size_t block_bytes = 16 * SIMD_ITERATION_SIZE;
        const size_t tail_bytes = 8 * SIMD_ITERATION_SIZE;

        for (; offset + block_bytes <= length; offset += block_bytes) {
                {
                        SIMD_TYPE_X4 a0 = SIMD_LOAD_X4(d0 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a1 = SIMD_LOAD_X4(d1 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a2 = SIMD_LOAD_X4(d2 + offset + 0 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g0_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 0), SIMD_X4_VAL(a1, 0), SIMD_X4_VAL(a2, 0));
                        SIMD_TYPE g0_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 1), SIMD_X4_VAL(a1, 1), SIMD_X4_VAL(a2, 1));
                        SIMD_TYPE g0_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 2), SIMD_X4_VAL(a1, 2), SIMD_X4_VAL(a2, 2));
                        SIMD_TYPE g0_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 3), SIMD_X4_VAL(a1, 3), SIMD_X4_VAL(a2, 3));

                        SIMD_TYPE_X4 a3 = SIMD_LOAD_X4(d3 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a4 = SIMD_LOAD_X4(d4 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a5 = SIMD_LOAD_X4(d5 + offset + 0 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g1_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 0), SIMD_X4_VAL(a4, 0), SIMD_X4_VAL(a5, 0));
                        SIMD_TYPE g1_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 1), SIMD_X4_VAL(a4, 1), SIMD_X4_VAL(a5, 1));
                        SIMD_TYPE g1_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 2), SIMD_X4_VAL(a4, 2), SIMD_X4_VAL(a5, 2));
                        SIMD_TYPE g1_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 3), SIMD_X4_VAL(a4, 3), SIMD_X4_VAL(a5, 3));

                        SIMD_TYPE_X4 a6 = SIMD_LOAD_X4(d6 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a7 = SIMD_LOAD_X4(d7 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a8 = SIMD_LOAD_X4(d8 + offset + 0 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g2_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 0), SIMD_X4_VAL(a7, 0), SIMD_X4_VAL(a8, 0));
                        SIMD_TYPE g2_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 1), SIMD_X4_VAL(a7, 1), SIMD_X4_VAL(a8, 1));
                        SIMD_TYPE g2_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 2), SIMD_X4_VAL(a7, 2), SIMD_X4_VAL(a8, 2));
                        SIMD_TYPE g2_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 3), SIMD_X4_VAL(a7, 3), SIMD_X4_VAL(a8, 3));

                        SIMD_TYPE acc0 = SIMD_EOR3_U8(g0_0, g1_0, g2_0);
                        SIMD_TYPE acc1 = SIMD_EOR3_U8(g0_1, g1_1, g2_1);
                        SIMD_TYPE acc2 = SIMD_EOR3_U8(g0_2, g1_2, g2_2);
                        SIMD_TYPE acc3 = SIMD_EOR3_U8(g0_3, g1_3, g2_3);

                        SIMD_TYPE_X4 a9 = SIMD_LOAD_X4(d9 + offset + 0 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_XOR(acc0, SIMD_X4_VAL(a9, 0));
                        acc1 = SIMD_XOR(acc1, SIMD_X4_VAL(a9, 1));
                        acc2 = SIMD_XOR(acc2, SIMD_X4_VAL(a9, 2));
                        acc3 = SIMD_XOR(acc3, SIMD_X4_VAL(a9, 3));

                        SIMD_STORE_R0(p0 + offset + (0 + 0) * SIMD_ITERATION_SIZE, acc0);
                        SIMD_STORE_R0(p0 + offset + (0 + 1) * SIMD_ITERATION_SIZE, acc1);
                        SIMD_STORE_R0(p0 + offset + (0 + 2) * SIMD_ITERATION_SIZE, acc2);
                        SIMD_STORE_R0(p0 + offset + (0 + 3) * SIMD_ITERATION_SIZE, acc3);
                }

                {
                        SIMD_TYPE_X4 a0 = SIMD_LOAD_X4(d0 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a1 = SIMD_LOAD_X4(d1 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a2 = SIMD_LOAD_X4(d2 + offset + 4 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g0_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 0), SIMD_X4_VAL(a1, 0), SIMD_X4_VAL(a2, 0));
                        SIMD_TYPE g0_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 1), SIMD_X4_VAL(a1, 1), SIMD_X4_VAL(a2, 1));
                        SIMD_TYPE g0_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 2), SIMD_X4_VAL(a1, 2), SIMD_X4_VAL(a2, 2));
                        SIMD_TYPE g0_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 3), SIMD_X4_VAL(a1, 3), SIMD_X4_VAL(a2, 3));

                        SIMD_TYPE_X4 a3 = SIMD_LOAD_X4(d3 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a4 = SIMD_LOAD_X4(d4 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a5 = SIMD_LOAD_X4(d5 + offset + 4 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g1_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 0), SIMD_X4_VAL(a4, 0), SIMD_X4_VAL(a5, 0));
                        SIMD_TYPE g1_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 1), SIMD_X4_VAL(a4, 1), SIMD_X4_VAL(a5, 1));
                        SIMD_TYPE g1_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 2), SIMD_X4_VAL(a4, 2), SIMD_X4_VAL(a5, 2));
                        SIMD_TYPE g1_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 3), SIMD_X4_VAL(a4, 3), SIMD_X4_VAL(a5, 3));

                        SIMD_TYPE_X4 a6 = SIMD_LOAD_X4(d6 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a7 = SIMD_LOAD_X4(d7 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a8 = SIMD_LOAD_X4(d8 + offset + 4 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g2_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 0), SIMD_X4_VAL(a7, 0), SIMD_X4_VAL(a8, 0));
                        SIMD_TYPE g2_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 1), SIMD_X4_VAL(a7, 1), SIMD_X4_VAL(a8, 1));
                        SIMD_TYPE g2_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 2), SIMD_X4_VAL(a7, 2), SIMD_X4_VAL(a8, 2));
                        SIMD_TYPE g2_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 3), SIMD_X4_VAL(a7, 3), SIMD_X4_VAL(a8, 3));

                        SIMD_TYPE acc0 = SIMD_EOR3_U8(g0_0, g1_0, g2_0);
                        SIMD_TYPE acc1 = SIMD_EOR3_U8(g0_1, g1_1, g2_1);
                        SIMD_TYPE acc2 = SIMD_EOR3_U8(g0_2, g1_2, g2_2);
                        SIMD_TYPE acc3 = SIMD_EOR3_U8(g0_3, g1_3, g2_3);

                        SIMD_TYPE_X4 a9 = SIMD_LOAD_X4(d9 + offset + 4 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_XOR(acc0, SIMD_X4_VAL(a9, 0));
                        acc1 = SIMD_XOR(acc1, SIMD_X4_VAL(a9, 1));
                        acc2 = SIMD_XOR(acc2, SIMD_X4_VAL(a9, 2));
                        acc3 = SIMD_XOR(acc3, SIMD_X4_VAL(a9, 3));

                        SIMD_STORE_R0(p0 + offset + (4 + 0) * SIMD_ITERATION_SIZE, acc0);
                        SIMD_STORE_R0(p0 + offset + (4 + 1) * SIMD_ITERATION_SIZE, acc1);
                        SIMD_STORE_R0(p0 + offset + (4 + 2) * SIMD_ITERATION_SIZE, acc2);
                        SIMD_STORE_R0(p0 + offset + (4 + 3) * SIMD_ITERATION_SIZE, acc3);
                }

                {
                        SIMD_TYPE_X4 a0 = SIMD_LOAD_X4(d0 + offset + 8 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a1 = SIMD_LOAD_X4(d1 + offset + 8 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a2 = SIMD_LOAD_X4(d2 + offset + 8 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g0_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 0), SIMD_X4_VAL(a1, 0), SIMD_X4_VAL(a2, 0));
                        SIMD_TYPE g0_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 1), SIMD_X4_VAL(a1, 1), SIMD_X4_VAL(a2, 1));
                        SIMD_TYPE g0_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 2), SIMD_X4_VAL(a1, 2), SIMD_X4_VAL(a2, 2));
                        SIMD_TYPE g0_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 3), SIMD_X4_VAL(a1, 3), SIMD_X4_VAL(a2, 3));

                        SIMD_TYPE_X4 a3 = SIMD_LOAD_X4(d3 + offset + 8 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a4 = SIMD_LOAD_X4(d4 + offset + 8 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a5 = SIMD_LOAD_X4(d5 + offset + 8 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g1_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 0), SIMD_X4_VAL(a4, 0), SIMD_X4_VAL(a5, 0));
                        SIMD_TYPE g1_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 1), SIMD_X4_VAL(a4, 1), SIMD_X4_VAL(a5, 1));
                        SIMD_TYPE g1_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 2), SIMD_X4_VAL(a4, 2), SIMD_X4_VAL(a5, 2));
                        SIMD_TYPE g1_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 3), SIMD_X4_VAL(a4, 3), SIMD_X4_VAL(a5, 3));

                        SIMD_TYPE_X4 a6 = SIMD_LOAD_X4(d6 + offset + 8 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a7 = SIMD_LOAD_X4(d7 + offset + 8 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a8 = SIMD_LOAD_X4(d8 + offset + 8 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g2_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 0), SIMD_X4_VAL(a7, 0), SIMD_X4_VAL(a8, 0));
                        SIMD_TYPE g2_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 1), SIMD_X4_VAL(a7, 1), SIMD_X4_VAL(a8, 1));
                        SIMD_TYPE g2_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 2), SIMD_X4_VAL(a7, 2), SIMD_X4_VAL(a8, 2));
                        SIMD_TYPE g2_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 3), SIMD_X4_VAL(a7, 3), SIMD_X4_VAL(a8, 3));

                        SIMD_TYPE acc0 = SIMD_EOR3_U8(g0_0, g1_0, g2_0);
                        SIMD_TYPE acc1 = SIMD_EOR3_U8(g0_1, g1_1, g2_1);
                        SIMD_TYPE acc2 = SIMD_EOR3_U8(g0_2, g1_2, g2_2);
                        SIMD_TYPE acc3 = SIMD_EOR3_U8(g0_3, g1_3, g2_3);

                        SIMD_TYPE_X4 a9 = SIMD_LOAD_X4(d9 + offset + 8 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_XOR(acc0, SIMD_X4_VAL(a9, 0));
                        acc1 = SIMD_XOR(acc1, SIMD_X4_VAL(a9, 1));
                        acc2 = SIMD_XOR(acc2, SIMD_X4_VAL(a9, 2));
                        acc3 = SIMD_XOR(acc3, SIMD_X4_VAL(a9, 3));

                        SIMD_STORE_R0(p0 + offset + (8 + 0) * SIMD_ITERATION_SIZE, acc0);
                        SIMD_STORE_R0(p0 + offset + (8 + 1) * SIMD_ITERATION_SIZE, acc1);
                        SIMD_STORE_R0(p0 + offset + (8 + 2) * SIMD_ITERATION_SIZE, acc2);
                        SIMD_STORE_R0(p0 + offset + (8 + 3) * SIMD_ITERATION_SIZE, acc3);
                }

                {
                        SIMD_TYPE_X4 a0 = SIMD_LOAD_X4(d0 + offset + 12 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a1 = SIMD_LOAD_X4(d1 + offset + 12 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a2 = SIMD_LOAD_X4(d2 + offset + 12 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g0_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 0), SIMD_X4_VAL(a1, 0), SIMD_X4_VAL(a2, 0));
                        SIMD_TYPE g0_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 1), SIMD_X4_VAL(a1, 1), SIMD_X4_VAL(a2, 1));
                        SIMD_TYPE g0_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 2), SIMD_X4_VAL(a1, 2), SIMD_X4_VAL(a2, 2));
                        SIMD_TYPE g0_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a0, 3), SIMD_X4_VAL(a1, 3), SIMD_X4_VAL(a2, 3));

                        SIMD_TYPE_X4 a3 = SIMD_LOAD_X4(d3 + offset + 12 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a4 = SIMD_LOAD_X4(d4 + offset + 12 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a5 = SIMD_LOAD_X4(d5 + offset + 12 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g1_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 0), SIMD_X4_VAL(a4, 0), SIMD_X4_VAL(a5, 0));
                        SIMD_TYPE g1_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 1), SIMD_X4_VAL(a4, 1), SIMD_X4_VAL(a5, 1));
                        SIMD_TYPE g1_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 2), SIMD_X4_VAL(a4, 2), SIMD_X4_VAL(a5, 2));
                        SIMD_TYPE g1_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a3, 3), SIMD_X4_VAL(a4, 3), SIMD_X4_VAL(a5, 3));

                        SIMD_TYPE_X4 a6 = SIMD_LOAD_X4(d6 + offset + 12 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a7 = SIMD_LOAD_X4(d7 + offset + 12 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE_X4 a8 = SIMD_LOAD_X4(d8 + offset + 12 * SIMD_ITERATION_SIZE);

                        SIMD_TYPE g2_0 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 0), SIMD_X4_VAL(a7, 0), SIMD_X4_VAL(a8, 0));
                        SIMD_TYPE g2_1 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 1), SIMD_X4_VAL(a7, 1), SIMD_X4_VAL(a8, 1));
                        SIMD_TYPE g2_2 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 2), SIMD_X4_VAL(a7, 2), SIMD_X4_VAL(a8, 2));
                        SIMD_TYPE g2_3 = SIMD_EOR3_U8(SIMD_X4_VAL(a6, 3), SIMD_X4_VAL(a7, 3), SIMD_X4_VAL(a8, 3));

                        SIMD_TYPE acc0 = SIMD_EOR3_U8(g0_0, g1_0, g2_0);
                        SIMD_TYPE acc1 = SIMD_EOR3_U8(g0_1, g1_1, g2_1);
                        SIMD_TYPE acc2 = SIMD_EOR3_U8(g0_2, g1_2, g2_2);
                        SIMD_TYPE acc3 = SIMD_EOR3_U8(g0_3, g1_3, g2_3);

                        SIMD_TYPE_X4 a9 = SIMD_LOAD_X4(d9 + offset + 12 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_XOR(acc0, SIMD_X4_VAL(a9, 0));
                        acc1 = SIMD_XOR(acc1, SIMD_X4_VAL(a9, 1));
                        acc2 = SIMD_XOR(acc2, SIMD_X4_VAL(a9, 2));
                        acc3 = SIMD_XOR(acc3, SIMD_X4_VAL(a9, 3));

                        SIMD_STORE_R0(p0 + offset + (12 + 0) * SIMD_ITERATION_SIZE, acc0);
                        SIMD_STORE_R0(p0 + offset + (12 + 1) * SIMD_ITERATION_SIZE, acc1);
                        SIMD_STORE_R0(p0 + offset + (12 + 2) * SIMD_ITERATION_SIZE, acc2);
                        SIMD_STORE_R0(p0 + offset + (12 + 3) * SIMD_ITERATION_SIZE, acc3);
                }
        }

        for (; offset + tail_bytes <= length; offset += tail_bytes) {
                SIMD_TYPE acc0 = SIMD_LOAD_R0(d0 + offset + 0 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc1 = SIMD_LOAD_R0(d0 + offset + 1 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc2 = SIMD_LOAD_R0(d0 + offset + 2 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc3 = SIMD_LOAD_R0(d0 + offset + 3 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc4 = SIMD_LOAD_R0(d0 + offset + 4 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc5 = SIMD_LOAD_R0(d0 + offset + 5 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc6 = SIMD_LOAD_R0(d0 + offset + 6 * SIMD_ITERATION_SIZE);
                SIMD_TYPE acc7 = SIMD_LOAD_R0(d0 + offset + 7 * SIMD_ITERATION_SIZE);

                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d1 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d1 + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left2 = SIMD_LOAD_R0(d1 + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left3 = SIMD_LOAD_R0(d1 + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left4 = SIMD_LOAD_R0(d1 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left5 = SIMD_LOAD_R0(d1 + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left6 = SIMD_LOAD_R0(d1 + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left7 = SIMD_LOAD_R0(d1 + offset + 7 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d2 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d2 + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right2 = SIMD_LOAD_R0(d2 + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right3 = SIMD_LOAD_R0(d2 + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right4 = SIMD_LOAD_R0(d2 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right5 = SIMD_LOAD_R0(d2 + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right6 = SIMD_LOAD_R0(d2 + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right7 = SIMD_LOAD_R0(d2 + offset + 7 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                        acc2 = SIMD_EOR3_U8(acc2, left2, right2);
                        acc3 = SIMD_EOR3_U8(acc3, left3, right3);
                        acc4 = SIMD_EOR3_U8(acc4, left4, right4);
                        acc5 = SIMD_EOR3_U8(acc5, left5, right5);
                        acc6 = SIMD_EOR3_U8(acc6, left6, right6);
                        acc7 = SIMD_EOR3_U8(acc7, left7, right7);
                }

                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d3 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d3 + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left2 = SIMD_LOAD_R0(d3 + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left3 = SIMD_LOAD_R0(d3 + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left4 = SIMD_LOAD_R0(d3 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left5 = SIMD_LOAD_R0(d3 + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left6 = SIMD_LOAD_R0(d3 + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left7 = SIMD_LOAD_R0(d3 + offset + 7 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d4 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d4 + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right2 = SIMD_LOAD_R0(d4 + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right3 = SIMD_LOAD_R0(d4 + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right4 = SIMD_LOAD_R0(d4 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right5 = SIMD_LOAD_R0(d4 + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right6 = SIMD_LOAD_R0(d4 + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right7 = SIMD_LOAD_R0(d4 + offset + 7 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                        acc2 = SIMD_EOR3_U8(acc2, left2, right2);
                        acc3 = SIMD_EOR3_U8(acc3, left3, right3);
                        acc4 = SIMD_EOR3_U8(acc4, left4, right4);
                        acc5 = SIMD_EOR3_U8(acc5, left5, right5);
                        acc6 = SIMD_EOR3_U8(acc6, left6, right6);
                        acc7 = SIMD_EOR3_U8(acc7, left7, right7);
                }

                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d5 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d5 + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left2 = SIMD_LOAD_R0(d5 + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left3 = SIMD_LOAD_R0(d5 + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left4 = SIMD_LOAD_R0(d5 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left5 = SIMD_LOAD_R0(d5 + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left6 = SIMD_LOAD_R0(d5 + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left7 = SIMD_LOAD_R0(d5 + offset + 7 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d6 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d6 + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right2 = SIMD_LOAD_R0(d6 + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right3 = SIMD_LOAD_R0(d6 + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right4 = SIMD_LOAD_R0(d6 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right5 = SIMD_LOAD_R0(d6 + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right6 = SIMD_LOAD_R0(d6 + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right7 = SIMD_LOAD_R0(d6 + offset + 7 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                        acc2 = SIMD_EOR3_U8(acc2, left2, right2);
                        acc3 = SIMD_EOR3_U8(acc3, left3, right3);
                        acc4 = SIMD_EOR3_U8(acc4, left4, right4);
                        acc5 = SIMD_EOR3_U8(acc5, left5, right5);
                        acc6 = SIMD_EOR3_U8(acc6, left6, right6);
                        acc7 = SIMD_EOR3_U8(acc7, left7, right7);
                }

                {
                        SIMD_TYPE left0 = SIMD_LOAD_R0(d7 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left1 = SIMD_LOAD_R0(d7 + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left2 = SIMD_LOAD_R0(d7 + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left3 = SIMD_LOAD_R0(d7 + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left4 = SIMD_LOAD_R0(d7 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left5 = SIMD_LOAD_R0(d7 + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left6 = SIMD_LOAD_R0(d7 + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE left7 = SIMD_LOAD_R0(d7 + offset + 7 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right0 = SIMD_LOAD_R0(d8 + offset + 0 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right1 = SIMD_LOAD_R0(d8 + offset + 1 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right2 = SIMD_LOAD_R0(d8 + offset + 2 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right3 = SIMD_LOAD_R0(d8 + offset + 3 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right4 = SIMD_LOAD_R0(d8 + offset + 4 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right5 = SIMD_LOAD_R0(d8 + offset + 5 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right6 = SIMD_LOAD_R0(d8 + offset + 6 * SIMD_ITERATION_SIZE);
                        SIMD_TYPE right7 = SIMD_LOAD_R0(d8 + offset + 7 * SIMD_ITERATION_SIZE);

                        acc0 = SIMD_EOR3_U8(acc0, left0, right0);
                        acc1 = SIMD_EOR3_U8(acc1, left1, right1);
                        acc2 = SIMD_EOR3_U8(acc2, left2, right2);
                        acc3 = SIMD_EOR3_U8(acc3, left3, right3);
                        acc4 = SIMD_EOR3_U8(acc4, left4, right4);
                        acc5 = SIMD_EOR3_U8(acc5, left5, right5);
                        acc6 = SIMD_EOR3_U8(acc6, left6, right6);
                        acc7 = SIMD_EOR3_U8(acc7, left7, right7);
                }

                acc0 = SIMD_XOR(acc0, SIMD_LOAD_R0(d9 + offset + 0 * SIMD_ITERATION_SIZE));
                acc1 = SIMD_XOR(acc1, SIMD_LOAD_R0(d9 + offset + 1 * SIMD_ITERATION_SIZE));
                acc2 = SIMD_XOR(acc2, SIMD_LOAD_R0(d9 + offset + 2 * SIMD_ITERATION_SIZE));
                acc3 = SIMD_XOR(acc3, SIMD_LOAD_R0(d9 + offset + 3 * SIMD_ITERATION_SIZE));
                acc4 = SIMD_XOR(acc4, SIMD_LOAD_R0(d9 + offset + 4 * SIMD_ITERATION_SIZE));
                acc5 = SIMD_XOR(acc5, SIMD_LOAD_R0(d9 + offset + 5 * SIMD_ITERATION_SIZE));
                acc6 = SIMD_XOR(acc6, SIMD_LOAD_R0(d9 + offset + 6 * SIMD_ITERATION_SIZE));
                acc7 = SIMD_XOR(acc7, SIMD_LOAD_R0(d9 + offset + 7 * SIMD_ITERATION_SIZE));

                SIMD_STORE_R0(p0 + offset + 0 * SIMD_ITERATION_SIZE, acc0);
                SIMD_STORE_R0(p0 + offset + 1 * SIMD_ITERATION_SIZE, acc1);
                SIMD_STORE_R0(p0 + offset + 2 * SIMD_ITERATION_SIZE, acc2);
                SIMD_STORE_R0(p0 + offset + 3 * SIMD_ITERATION_SIZE, acc3);
                SIMD_STORE_R0(p0 + offset + 4 * SIMD_ITERATION_SIZE, acc4);
                SIMD_STORE_R0(p0 + offset + 5 * SIMD_ITERATION_SIZE, acc5);
                SIMD_STORE_R0(p0 + offset + 6 * SIMD_ITERATION_SIZE, acc6);
                SIMD_STORE_R0(p0 + offset + 7 * SIMD_ITERATION_SIZE, acc7);
        }

        for (; offset < length; offset += SIMD_ITERATION_SIZE) {
                SIMD_TYPE v0 = SIMD_LOAD_R0(d0 + offset);
                SIMD_TYPE v1 = SIMD_LOAD_R0(d1 + offset);
                SIMD_TYPE v2 = SIMD_LOAD_R0(d2 + offset);
                SIMD_TYPE v3 = SIMD_LOAD_R0(d3 + offset);
                SIMD_TYPE v4 = SIMD_LOAD_R0(d4 + offset);
                SIMD_TYPE v5 = SIMD_LOAD_R0(d5 + offset);
                SIMD_TYPE v6 = SIMD_LOAD_R0(d6 + offset);
                SIMD_TYPE v7 = SIMD_LOAD_R0(d7 + offset);
                SIMD_TYPE v8 = SIMD_LOAD_R0(d8 + offset);
                SIMD_TYPE v9 = SIMD_LOAD_R0(d9 + offset);

                SIMD_STORE_R0(p0 + offset, xor_ten_vectors(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9));
        }
#endif
}
