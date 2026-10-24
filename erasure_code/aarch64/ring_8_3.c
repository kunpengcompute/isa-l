#include <stddef.h>
#include <stdint.h>
#include "erasure_code.h"
#include "simd_opt.h"
#include "test.h"
#include "ring_internal.h"

/* Branch-prediction hint for user-space (no <linux/compiler.h> available). */
#ifndef likely
#define likely(x)   __builtin_expect(!!(x), 1)
#endif

/*
 * Fixed-arity XORs keep every generated four-slot inverse expression visible
 * in C while mapping directly to NEON EOR instructions.  No coefficient is
 * interpreted at run time.
 */
#define RING_XOR_1(a)                (a)
#define RING_XOR_2(a, b)             SIMD_XOR((a), (b))
#define RING_XOR_3(a, b, c)          SIMD_EOR3_U8((a), (b), (c))
#define RING_XOR_4(a, b, c, d)       SIMD_XOR(RING_XOR_3((a), (b), (c)), (d))
#define RING_XOR_5(a, b, c, d, e)    SIMD_EOR3_U8(RING_XOR_3((a), (b), (c)), (d), (e))
#define RING_XOR_6(a, b, c, d, e, f) SIMD_XOR(RING_XOR_3((a), (b), (c)), RING_XOR_3((d), (e), (f)))
#define RING_XOR_7(a, b, c, d, e, f, g)                                                            \
        SIMD_EOR3_U8(RING_XOR_3((a), (b), (c)), RING_XOR_3((d), (e), (f)), (g))
#define RING_XOR_8(a, b, c, d, e, f, g, h)                                                         \
        SIMD_XOR(RING_XOR_7((a), (b), (c), (d), (e), (f), (g)), (h))
#define RING_XOR_9(a, b, c, d, e, f, g, h, i)                                                      \
        SIMD_EOR3_U8(RING_XOR_3((a), (b), (c)), RING_XOR_3((d), (e), (f)),                         \
                     RING_XOR_3((g), (h), (i)))
#define RING_XOR_10(a, b, c, d, e, f, g, h, i, j)                                                  \
        SIMD_XOR(RING_XOR_5((a), (b), (c), (d), (e)), RING_XOR_5((f), (g), (h), (i), (j)))
#define RING_XOR_11(a, b, c, d, e, f, g, h, i, j, k)                                               \
        SIMD_EOR3_U8(RING_XOR_5((a), (b), (c), (d), (e)), RING_XOR_5((f), (g), (h), (i), (j)), (k))
#define RING_XOR_12(a, b, c, d, e, f, g, h, i, j, k, l)                                            \
        SIMD_EOR3_U8(RING_XOR_5((a), (b), (c), (d), (e)), RING_XOR_4((f), (g), (h), (i)),          \
                     RING_XOR_3((j), (k), (l)))

/*
 * Keep the four slot offsets as register values inside the hot loop.  The
 * empty compiler barrier prevents loop-strength reduction from expanding
 * every base+offset pair into a separate, long-lived pointer induction
 * variable.  It emits no machine instruction on the supported compilers.
 */
#if defined(__clang__) || defined(__GNUC__)
#define RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3)                                 \
        __asm__ volatile("" : "+r"(offset0), "+r"(offset1), "+r"(offset2), "+r"(offset3))
#else
#define RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3)                                 \
        ((void) (offset0), (void) (offset1), (void) (offset2), (void) (offset3))
#endif

enum {
        RING_8_3_SINGLE_UNROLL = 8,
        RING_8_3_DATA_SHARDS = 8,
        RING_8_3_PARITY_SHARDS = 3,
        RING_8_3_TOTAL_SHARDS = 11,
        RING_8_3_COEFFICIENT_REGIONS = 4,
};
#define RING_8_3_SIMD_BATCH_BYTES ((int) ((RING_8_3_COEFFICIENT_REGIONS) * (SIMD_ITERATION_SIZE)))

#define RING_8_3_SINGLE_BLOCK_BYTES (RING_8_3_SINGLE_UNROLL * SIMD_ITERATION_SIZE)

/*
 * Single data erasure uses only the all-one parity P0.  Because this row has
 * no cross-slot transform, process the four slot regions as one contiguous
 * node instead of revisiting four pointer arrays in every SIMD iteration.
 */
static void
recover_data_single_fused(unsigned char *output, unsigned char **shards, int missing, int slot_size)
{
        const unsigned char *p0 = shards[8];
        const unsigned char *s[8] = { shards[0], shards[1], shards[2], shards[3],
                                      shards[4], shards[5], shards[6], shards[7] };
        size_t node_size = 4 * (size_t) slot_size;
        size_t offset = 0;
        s[missing] = p0;
        /* XOR is commutative: replace the missing data pointer with P0. */

        for (; offset + RING_8_3_SINGLE_BLOCK_BYTES <= node_size;
             offset += RING_8_3_SINGLE_BLOCK_BYTES) {
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

                acc0 = SIMD_XOR(acc0, SIMD_LOAD_R0(s[7] + offset + 0 * SIMD_ITERATION_SIZE));
                acc1 = SIMD_XOR(acc1, SIMD_LOAD_R0(s[7] + offset + 1 * SIMD_ITERATION_SIZE));
                acc2 = SIMD_XOR(acc2, SIMD_LOAD_R0(s[7] + offset + 2 * SIMD_ITERATION_SIZE));
                acc3 = SIMD_XOR(acc3, SIMD_LOAD_R0(s[7] + offset + 3 * SIMD_ITERATION_SIZE));
                acc4 = SIMD_XOR(acc4, SIMD_LOAD_R0(s[7] + offset + 4 * SIMD_ITERATION_SIZE));
                acc5 = SIMD_XOR(acc5, SIMD_LOAD_R0(s[7] + offset + 5 * SIMD_ITERATION_SIZE));
                acc6 = SIMD_XOR(acc6, SIMD_LOAD_R0(s[7] + offset + 6 * SIMD_ITERATION_SIZE));
                acc7 = SIMD_XOR(acc7, SIMD_LOAD_R0(s[7] + offset + 7 * SIMD_ITERATION_SIZE));

                SIMD_STORE_R0(output + offset + 0 * SIMD_ITERATION_SIZE, acc0);
                SIMD_STORE_R0(output + offset + 1 * SIMD_ITERATION_SIZE, acc1);
                SIMD_STORE_R0(output + offset + 2 * SIMD_ITERATION_SIZE, acc2);
                SIMD_STORE_R0(output + offset + 3 * SIMD_ITERATION_SIZE, acc3);
                SIMD_STORE_R0(output + offset + 4 * SIMD_ITERATION_SIZE, acc4);
                SIMD_STORE_R0(output + offset + 5 * SIMD_ITERATION_SIZE, acc5);
                SIMD_STORE_R0(output + offset + 6 * SIMD_ITERATION_SIZE, acc6);
                SIMD_STORE_R0(output + offset + 7 * SIMD_ITERATION_SIZE, acc7);
        }

        for (; offset < node_size; offset += SIMD_ITERATION_SIZE) {
                SIMD_TYPE v0 = SIMD_LOAD_R0(s[0] + offset);
                SIMD_TYPE v1 = SIMD_LOAD_R0(s[1] + offset);
                SIMD_TYPE v2 = SIMD_LOAD_R0(s[2] + offset);
                SIMD_TYPE v3 = SIMD_LOAD_R0(s[3] + offset);
                SIMD_TYPE v4 = SIMD_LOAD_R0(s[4] + offset);
                SIMD_TYPE v5 = SIMD_LOAD_R0(s[5] + offset);
                SIMD_TYPE v6 = SIMD_LOAD_R0(s[6] + offset);
                SIMD_TYPE v7 = SIMD_LOAD_R0(s[7] + offset);

                SIMD_STORE_R0(output + offset, RING_XOR_8(v0, v1, v2, v3, v4, v5, v6, v7));
        }
}

/* Same argument layout as the original 4+2 pair-erasure kernel. */
static void
recover_data_pair_fused(unsigned char *out0, unsigned char *out1, unsigned char *const buffs[11],
                        int missing0, int missing1, int slot_size)
{
        const unsigned char *p0_base = buffs[8];
        const unsigned char *p1_base = buffs[9];
        unsigned char *dst0_base = out0;
        unsigned char *dst1_base = out1;
        uint8_t data_mask = (uint8_t) ((1U << missing0) | (1U << missing1));
        size_t node_size = 4 * (size_t) slot_size;

        /* Literal PDF RM schedule followed by 28 fixed inverses. */
        switch (data_mask) {
        case 0x03: {
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D0,D1; A^-1 = [b a; a a]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s01, s02, s03, s11, s12, s13);
                        SIMD_TYPE r01 = RING_XOR_3(s00, s10, s11);
                        SIMD_TYPE r02 = RING_XOR_3(s02, s03, s13);
                        SIMD_TYPE r03 = RING_XOR_7(s00, s01, s02, s10, s11, s12, s13);

                        SIMD_TYPE r10 = RING_XOR_6(s01, s02, s03, s11, s12, s13);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s01, s10, s11);
                        SIMD_TYPE r12 = RING_XOR_2(s03, s13);
                        SIMD_TYPE r13 = RING_XOR_8(s00, s01, s02, s03, s10, s11, s12, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x05: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D0,D2; A^-1 = [e f; f f]. */
                        SIMD_TYPE r00 = RING_XOR_3(s01, s10, s11);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s01, s02, s10, s12);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s02, s03, s10, s13);
                        SIMD_TYPE r03 = RING_XOR_3(s00, s03, s10);

                        SIMD_TYPE r10 = RING_XOR_4(s00, s01, s10, s11);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s02, s10, s12);
                        SIMD_TYPE r12 = RING_XOR_4(s00, s03, s10, s13);
                        SIMD_TYPE r13 = RING_XOR_2(s00, s10);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x09: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D0,D3; A^-1 = [7 6; 6 6]. */
                        SIMD_TYPE r00 = RING_XOR_3(s00, s02, s12);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s01, s02, s03, s10, s12, s13);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s01, s03, s10, s11, s12, s13);
                        SIMD_TYPE r03 = RING_XOR_3(s01, s11, s13);

                        SIMD_TYPE r10 = RING_XOR_2(s02, s12);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s02, s03, s10, s12, s13);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s02, s03, s10, s11, s12, s13);
                        SIMD_TYPE r13 = RING_XOR_4(s01, s03, s11, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x11: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D0,D4; A^-1 = [9 8; 8 8]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s01, s02, s11, s12);
                        SIMD_TYPE r01 = RING_XOR_3(s03, s11, s13);
                        SIMD_TYPE r02 = RING_XOR_3(s01, s02, s11);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s01, s03, s10, s11);

                        SIMD_TYPE r10 = RING_XOR_4(s01, s02, s11, s12);
                        SIMD_TYPE r11 = RING_XOR_4(s01, s03, s11, s13);
                        SIMD_TYPE r12 = RING_XOR_2(s01, s11);
                        SIMD_TYPE r13 = RING_XOR_4(s00, s01, s10, s11);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x21: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D0,D5; A^-1 = [8 9; 9 9]. */
                        SIMD_TYPE r00 = RING_XOR_5(s01, s02, s10, s11, s12);
                        SIMD_TYPE r01 = RING_XOR_3(s01, s03, s13);
                        SIMD_TYPE r02 = RING_XOR_3(s01, s11, s12);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s01, s10, s11, s13);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s01, s02, s10, s11, s12);
                        SIMD_TYPE r11 = RING_XOR_2(s03, s13);
                        SIMD_TYPE r12 = RING_XOR_4(s01, s02, s11, s12);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s01, s03, s10, s11, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x41: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D0,D6; A^-1 = [4 5; 5 5]. */
                        SIMD_TYPE r00 = RING_XOR_5(s02, s03, s10, s12, s13);
                        SIMD_TYPE r01 = RING_XOR_3(s02, s11, s12);
                        SIMD_TYPE r02 = RING_XOR_3(s00, s02, s10);
                        SIMD_TYPE r03 = RING_XOR_5(s01, s02, s11, s12, s13);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s02, s03, s10, s12, s13);
                        SIMD_TYPE r11 = RING_XOR_4(s01, s02, s11, s12);
                        SIMD_TYPE r12 = RING_XOR_2(s00, s10);
                        SIMD_TYPE r13 = RING_XOR_6(s01, s02, s03, s11, s12, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x81: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D0,D7; A^-1 = [6 7; 7 7]. */
                        SIMD_TYPE r00 = RING_XOR_3(s02, s10, s12);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s02, s03, s10, s11, s12, s13);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s01, s02, s03, s10, s11, s13);
                        SIMD_TYPE r03 = RING_XOR_3(s01, s03, s11);

                        SIMD_TYPE r10 = RING_XOR_4(s00, s02, s10, s12);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s02, s03, s10, s11, s12, s13);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s01, s03, s10, s11, s13);
                        SIMD_TYPE r13 = RING_XOR_2(s01, s11);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x06: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D1,D2; A^-1 = [3 1; 2 1]. */
                        SIMD_TYPE r00 = RING_XOR_3(s00, s03, s10);
                        SIMD_TYPE r01 = RING_XOR_4(s00, s01, s03, s11);
                        SIMD_TYPE r02 = RING_XOR_4(s01, s02, s03, s12);
                        SIMD_TYPE r03 = RING_XOR_2(s02, s13);

                        SIMD_TYPE r10 = RING_XOR_2(s03, s10);
                        SIMD_TYPE r11 = RING_XOR_3(s00, s03, s11);
                        SIMD_TYPE r12 = RING_XOR_3(s01, s03, s12);
                        SIMD_TYPE r13 = RING_XOR_3(s02, s03, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x0a: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D1,D3; A^-1 = [b 5; a 5]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s01, s02, s03, s10, s12, s13);
                        SIMD_TYPE r01 = RING_XOR_3(s00, s11, s12);
                        SIMD_TYPE r02 = RING_XOR_3(s02, s03, s10);
                        SIMD_TYPE r03 = RING_XOR_6(s00, s01, s02, s11, s12, s13);

                        SIMD_TYPE r10 = RING_XOR_6(s01, s02, s03, s10, s12, s13);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s01, s11, s12);
                        SIMD_TYPE r12 = RING_XOR_2(s03, s10);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s01, s02, s03, s11, s12, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x12: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D1,D4; A^-1 = [c 9; d 9]. */
                        SIMD_TYPE r00 = RING_XOR_5(s01, s03, s10, s11, s12);
                        SIMD_TYPE r01 = RING_XOR_4(s01, s02, s03, s13);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s01, s02, s11, s12);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s02, s10, s11, s13);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s01, s03, s10, s11, s12);
                        SIMD_TYPE r11 = RING_XOR_3(s02, s03, s13);
                        SIMD_TYPE r12 = RING_XOR_4(s00, s01, s11, s12);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s02, s03, s10, s11, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x22: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D1,D5; A^-1 = [e 8; f 8]. */
                        SIMD_TYPE r00 = RING_XOR_3(s01, s11, s12);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s01, s02, s11, s13);
                        SIMD_TYPE r02 = RING_XOR_4(s00, s02, s03, s11);
                        SIMD_TYPE r03 = RING_XOR_4(s00, s03, s10, s11);

                        SIMD_TYPE r10 = RING_XOR_4(s00, s01, s11, s12);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s02, s11, s13);
                        SIMD_TYPE r12 = RING_XOR_3(s00, s03, s11);
                        SIMD_TYPE r13 = RING_XOR_3(s00, s10, s11);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x42: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D1,D6; A^-1 = [d 6; c 6]. */
                        SIMD_TYPE r00 = RING_XOR_4(s00, s01, s03, s12);
                        SIMD_TYPE r01 = RING_XOR_5(s02, s03, s10, s12, s13);
                        SIMD_TYPE r02 = RING_XOR_6(s00, s01, s10, s11, s12, s13);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s02, s03, s11, s13);

                        SIMD_TYPE r10 = RING_XOR_3(s01, s03, s12);
                        SIMD_TYPE r11 = RING_XOR_6(s01, s02, s03, s10, s12, s13);
                        SIMD_TYPE r12 = RING_XOR_7(s00, s01, s02, s10, s11, s12, s13);
                        SIMD_TYPE r13 = RING_XOR_4(s00, s02, s11, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x82: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D1,D7; A^-1 = [7 3; 6 3]. */
                        SIMD_TYPE r00 = RING_XOR_4(s00, s02, s10, s13);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s01, s02, s03, s10, s11, s13);
                        SIMD_TYPE r02 = RING_XOR_6(s00, s01, s03, s11, s12, s13);
                        SIMD_TYPE r03 = RING_XOR_2(s01, s12);

                        SIMD_TYPE r10 = RING_XOR_3(s02, s10, s13);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s02, s03, s10, s11, s13);
                        SIMD_TYPE r12 = RING_XOR_7(s00, s01, s02, s03, s11, s12, s13);
                        SIMD_TYPE r13 = RING_XOR_3(s01, s03, s12);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x0c: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D2,D3; A^-1 = [5 9; 4 9]. */
                        SIMD_TYPE r00 = RING_XOR_6(s00, s02, s03, s10, s11, s12);
                        SIMD_TYPE r01 = RING_XOR_3(s01, s02, s13);
                        SIMD_TYPE r02 = RING_XOR_3(s00, s11, s12);
                        SIMD_TYPE r03 = RING_XOR_6(s01, s02, s03, s10, s11, s13);

                        SIMD_TYPE r10 = RING_XOR_5(s02, s03, s10, s11, s12);
                        SIMD_TYPE r11 = RING_XOR_2(s02, s13);
                        SIMD_TYPE r12 = RING_XOR_4(s00, s02, s11, s12);
                        SIMD_TYPE r13 = RING_XOR_5(s01, s02, s10, s11, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x14: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D2,D4; A^-1 = [e 5; f 5]. */
                        SIMD_TYPE r00 = RING_XOR_4(s01, s10, s12, s13);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s01, s02, s11, s12);
                        SIMD_TYPE r02 = RING_XOR_4(s00, s02, s03, s10);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s03, s11, s12, s13);

                        SIMD_TYPE r10 = RING_XOR_5(s00, s01, s10, s12, s13);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s02, s11, s12);
                        SIMD_TYPE r12 = RING_XOR_3(s00, s03, s10);
                        SIMD_TYPE r13 = RING_XOR_4(s00, s11, s12, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x24: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D2,D5; A^-1 = [b 6; a 6]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s01, s02, s03, s12);
                        SIMD_TYPE r01 = RING_XOR_4(s00, s10, s12, s13);
                        SIMD_TYPE r02 = RING_XOR_6(s02, s03, s10, s11, s12, s13);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s01, s02, s11, s13);

                        SIMD_TYPE r10 = RING_XOR_4(s01, s02, s03, s12);
                        SIMD_TYPE r11 = RING_XOR_5(s00, s01, s10, s12, s13);
                        SIMD_TYPE r12 = RING_XOR_5(s03, s10, s11, s12, s13);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s01, s02, s03, s11, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x44: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D2,D6; A^-1 = [6 8; 7 8]. */
                        SIMD_TYPE r00 = RING_XOR_3(s02, s11, s12);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s02, s03, s11, s13);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s01, s02, s03, s11);
                        SIMD_TYPE r03 = RING_XOR_4(s01, s03, s10, s11);

                        SIMD_TYPE r10 = RING_XOR_4(s00, s02, s11, s12);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s01, s02, s03, s11, s13);
                        SIMD_TYPE r12 = RING_XOR_4(s00, s01, s03, s11);
                        SIMD_TYPE r13 = RING_XOR_3(s01, s10, s11);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x84: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D2,D7; A^-1 = [c e; d e]. */
                        SIMD_TYPE r00 = RING_XOR_3(s01, s03, s11);
                        SIMD_TYPE r01 = RING_XOR_6(s01, s02, s03, s10, s11, s12);
                        SIMD_TYPE r02 = RING_XOR_6(s00, s01, s02, s10, s12, s13);
                        SIMD_TYPE r03 = RING_XOR_4(s00, s02, s10, s13);

                        SIMD_TYPE r10 = RING_XOR_4(s00, s01, s03, s11);
                        SIMD_TYPE r11 = RING_XOR_5(s02, s03, s10, s11, s12);
                        SIMD_TYPE r12 = RING_XOR_5(s00, s01, s10, s12, s13);
                        SIMD_TYPE r13 = RING_XOR_5(s00, s02, s03, s10, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x18: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D3,D4; A^-1 = [5 1; 4 1]. */
                        SIMD_TYPE r00 = RING_XOR_4(s00, s02, s03, s10);
                        SIMD_TYPE r01 = RING_XOR_3(s01, s02, s11);
                        SIMD_TYPE r02 = RING_XOR_2(s00, s12);
                        SIMD_TYPE r03 = RING_XOR_4(s01, s02, s03, s13);

                        SIMD_TYPE r10 = RING_XOR_3(s02, s03, s10);
                        SIMD_TYPE r11 = RING_XOR_2(s02, s11);
                        SIMD_TYPE r12 = RING_XOR_3(s00, s02, s12);
                        SIMD_TYPE r13 = RING_XOR_3(s01, s02, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x28: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D3,D5; A^-1 = [3 f; 2 f]. */
                        SIMD_TYPE r00 = RING_XOR_4(s00, s03, s10, s11);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s01, s03, s10, s12);
                        SIMD_TYPE r02 = RING_XOR_5(s01, s02, s03, s10, s13);
                        SIMD_TYPE r03 = RING_XOR_2(s02, s10);

                        SIMD_TYPE r10 = RING_XOR_3(s03, s10, s11);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s03, s10, s12);
                        SIMD_TYPE r12 = RING_XOR_4(s01, s03, s10, s13);
                        SIMD_TYPE r13 = RING_XOR_3(s02, s03, s10);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x48: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D3,D6; A^-1 = [8 a; 9 a]. */
                        SIMD_TYPE r00 = RING_XOR_5(s01, s02, s11, s12, s13);
                        SIMD_TYPE r01 = RING_XOR_4(s01, s03, s10, s11);
                        SIMD_TYPE r02 = RING_XOR_2(s01, s13);
                        SIMD_TYPE r03 = RING_XOR_6(s00, s01, s10, s11, s12, s13);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s01, s02, s11, s12, s13);
                        SIMD_TYPE r11 = RING_XOR_3(s03, s10, s11);
                        SIMD_TYPE r12 = RING_XOR_3(s01, s02, s13);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s01, s03, s10, s11, s12, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x88: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D3,D7; A^-1 = [b d; a d]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s01, s02, s03, s10, s11, s13);
                        SIMD_TYPE r01 = RING_XOR_3(s00, s12, s13);
                        SIMD_TYPE r02 = RING_XOR_4(s02, s03, s10, s11);
                        SIMD_TYPE r03 = RING_XOR_6(s00, s01, s02, s10, s12, s13);

                        SIMD_TYPE r10 = RING_XOR_6(s01, s02, s03, s10, s11, s13);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s01, s12, s13);
                        SIMD_TYPE r12 = RING_XOR_3(s03, s10, s11);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s01, s02, s03, s10, s12, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x30: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D4,D5; A^-1 = [2 a; 3 a]. */
                        SIMD_TYPE r00 = RING_XOR_4(s03, s11, s12, s13);
                        SIMD_TYPE r01 = RING_XOR_4(s00, s03, s10, s11);
                        SIMD_TYPE r02 = RING_XOR_3(s01, s03, s13);
                        SIMD_TYPE r03 = RING_XOR_6(s02, s03, s10, s11, s12, s13);

                        SIMD_TYPE r10 = RING_XOR_5(s00, s03, s11, s12, s13);
                        SIMD_TYPE r11 = RING_XOR_5(s00, s01, s03, s10, s11);
                        SIMD_TYPE r12 = RING_XOR_4(s01, s02, s03, s13);
                        SIMD_TYPE r13 = RING_XOR_5(s02, s10, s11, s12, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x50: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D4,D6; A^-1 = [c f; d f]. */
                        SIMD_TYPE r00 = RING_XOR_4(s01, s03, s10, s11);
                        SIMD_TYPE r01 = RING_XOR_5(s01, s02, s03, s10, s12);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s01, s02, s10, s13);
                        SIMD_TYPE r03 = RING_XOR_3(s00, s02, s10);

                        SIMD_TYPE r10 = RING_XOR_5(s00, s01, s03, s10, s11);
                        SIMD_TYPE r11 = RING_XOR_4(s02, s03, s10, s12);
                        SIMD_TYPE r12 = RING_XOR_4(s00, s01, s10, s13);
                        SIMD_TYPE r13 = RING_XOR_4(s00, s02, s03, s10);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x90: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D4,D7; A^-1 = [3 c; 2 c]. */
                        SIMD_TYPE r00 = RING_XOR_4(s00, s03, s11, s13);
                        SIMD_TYPE r01 = RING_XOR_6(s00, s01, s03, s11, s12, s13);
                        SIMD_TYPE r02 = RING_XOR_6(s01, s02, s03, s10, s11, s12);
                        SIMD_TYPE r03 = RING_XOR_3(s02, s10, s12);

                        SIMD_TYPE r10 = RING_XOR_3(s03, s11, s13);
                        SIMD_TYPE r11 = RING_XOR_5(s00, s03, s11, s12, s13);
                        SIMD_TYPE r12 = RING_XOR_5(s01, s03, s10, s11, s12);
                        SIMD_TYPE r13 = RING_XOR_4(s02, s03, s10, s12);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0x60: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D5,D6; A^-1 = [7 1; 6 1]. */
                        SIMD_TYPE r00 = RING_XOR_3(s00, s02, s10);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s01, s02, s03, s11);
                        SIMD_TYPE r02 = RING_XOR_4(s00, s01, s03, s12);
                        SIMD_TYPE r03 = RING_XOR_2(s01, s13);

                        SIMD_TYPE r10 = RING_XOR_2(s02, s10);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s02, s03, s11);
                        SIMD_TYPE r12 = RING_XOR_5(s00, s01, s02, s03, s12);
                        SIMD_TYPE r13 = RING_XOR_3(s01, s03, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0xa0: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D5,D7; A^-1 = [5 b; 4 b]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s02, s03, s10, s11, s12, s13);
                        SIMD_TYPE r01 = RING_XOR_3(s01, s02, s10);
                        SIMD_TYPE r02 = RING_XOR_3(s00, s12, s13);
                        SIMD_TYPE r03 = RING_XOR_6(s01, s02, s03, s10, s11, s12);

                        SIMD_TYPE r10 = RING_XOR_6(s02, s03, s10, s11, s12, s13);
                        SIMD_TYPE r11 = RING_XOR_2(s02, s10);
                        SIMD_TYPE r12 = RING_XOR_4(s00, s02, s12, s13);
                        SIMD_TYPE r13 = RING_XOR_5(s01, s02, s10, s11, s12);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }

        case 0xc0: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        /* Missing D6,D7; A^-1 = [f 2; e 2]. */
                        SIMD_TYPE r00 = RING_XOR_3(s00, s01, s13);
                        SIMD_TYPE r01 = RING_XOR_4(s00, s02, s10, s13);
                        SIMD_TYPE r02 = RING_XOR_4(s00, s03, s11, s13);
                        SIMD_TYPE r03 = RING_XOR_3(s00, s12, s13);

                        SIMD_TYPE r10 = RING_XOR_2(s01, s13);
                        SIMD_TYPE r11 = RING_XOR_5(s00, s01, s02, s10, s13);
                        SIMD_TYPE r12 = RING_XOR_5(s00, s02, s03, s11, s13);
                        SIMD_TYPE r13 = RING_XOR_4(s00, s03, s12, s13);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                }
                return;
        }
        }
}

/* Three data erasures use all three PDF syndrome equations. */
static void
recover_data_triple_fused(unsigned char *out0, unsigned char *out1, unsigned char *out2,
                          unsigned char *const buffs[11], int missing0, int missing1, int missing2,
                          int slot_size)
{
        const unsigned char *p0_base = buffs[8];
        const unsigned char *p1_base = buffs[9];
        const unsigned char *p2_base = buffs[10];
        unsigned char *dst0_base = out0;
        unsigned char *dst1_base = out1;
        unsigned char *dst2_base = out2;
        uint8_t data_mask = (uint8_t) ((1U << missing0) | (1U << missing1) | (1U << missing2));
        size_t node_size = 4 * (size_t) slot_size;

        /* Hot path: D2,D4,D6 missing. Hoisted out of the 56-case switch so the
         * compiler keeps it in a small, register-tight function and the i-cache
         * footprint stays low. The key optimization here is common-subexpression
         * elimination in the inverse matrix application: three EOR3 triples are
         * reused across 4-6 of the 12 r outputs, cutting the EOR3 instruction
         * count from 32 to 25. The EOR3 execution port (1/cycle throughput,
         * 3-cycle latency on Neoverse V2) is the actual bottleneck for this
         * kernel, not load latency, so the CSE directly attacks the hot path. */
        if (likely(data_mask == 0x54)) {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i + RING_8_3_SIMD_BATCH_BYTES <= node_size;
                     i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D4,D6; A^-1 = [5 d 9; 6 a d; 2 7 4].
                         * CSE: three EOR3 triples reused across 4-6 r outputs,
                         * cutting EOR3 count 32 -> 25. */
                        SIMD_TYPE t1 = SIMD_EOR3_U8(s03, s10, s22);
                        SIMD_TYPE t2 = SIMD_EOR3_U8(s01, s13, s20);
                        SIMD_TYPE t3 = SIMD_EOR3_U8(s02, s03, s21);

                        SIMD_TYPE r00 = RING_XOR_7(t1, s00, s02, s11, s13, s20, s21);
                        SIMD_TYPE r01 = RING_XOR_5(s01, s02, s12, s13, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s10, s11, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_5(t2, t3, s10, s12, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_5(t1, s00, s02, s11, s23);
                        SIMD_TYPE r12 = SIMD_XOR(SIMD_XOR(t2, t3), s00);
                        SIMD_TYPE r13 = RING_XOR_5(t1, t2, s11, s12, s23);

                        SIMD_TYPE r20 = SIMD_XOR(SIMD_XOR(t1, s12), s23);
                        SIMD_TYPE r21 = RING_XOR_5(t1, s00, s11, s12, s13);
                        SIMD_TYPE r22 = SIMD_XOR(SIMD_XOR(t1, t2), s11);
                        SIMD_TYPE r23 = SIMD_XOR(SIMD_XOR(t3, s11), s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        /* Literal PDF RM schedule followed by 56 fixed inverses. */
        switch (data_mask) {
        case 0x07: {
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D1,D2; A^-1 = [1 5 5; 1 b a; 1 e f]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_5(s01, s11, s12, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_3(s02, s10, s20);
                        SIMD_TYPE r03 = RING_XOR_7(s03, s11, s12, s13, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_8(s00, s10, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_4(s01, s10, s20, s21);
                        SIMD_TYPE r12 = RING_XOR_4(s02, s12, s13, s23);
                        SIMD_TYPE r13 = RING_XOR_8(s03, s10, s11, s12, s20, s21, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_4(s00, s11, s20, s21);
                        SIMD_TYPE r21 = RING_XOR_6(s01, s10, s11, s12, s20, s22);
                        SIMD_TYPE r22 = RING_XOR_6(s02, s10, s12, s13, s20, s23);
                        SIMD_TYPE r23 = RING_XOR_4(s03, s10, s13, s20);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x0b: {
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D1,D3; A^-1 = [f c 2; c f 3; 2 3 1]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s01, s11, s13, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s02, s11, s12, s13, s20, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s03, s10, s11, s12, s21, s23);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s10, s12, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_6(s01, s03, s10, s11, s20, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s01, s02, s03, s10, s12, s20, s21, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s02, s10, s13, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_4(s00, s02, s10, s22);

                        SIMD_TYPE r20 = RING_XOR_4(s03, s10, s13, s20);
                        SIMD_TYPE r21 = RING_XOR_6(s00, s03, s10, s11, s13, s21);
                        SIMD_TYPE r22 = RING_XOR_6(s01, s03, s11, s12, s13, s22);
                        SIMD_TYPE r23 = RING_XOR_4(s02, s03, s12, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x13: {
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D1,D4; A^-1 = [e 2 d; 4 3 7; b 1 a]. */
                        SIMD_TYPE r00 = RING_XOR_5(s01, s13, s20, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s01, s02, s10, s13, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s02, s03, s11, s13, s20, s21);
                        SIMD_TYPE r03 = RING_XOR_7(s00, s03, s12, s13, s20, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_6(s02, s03, s10, s13, s20, s22);
                        SIMD_TYPE r11 = RING_XOR_8(s02, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_4(s01, s02, s12, s21);

                        SIMD_TYPE r20 = RING_XOR_8(s00, s01, s02, s03, s10, s21, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_4(s00, s11, s20, s21);
                        SIMD_TYPE r22 = RING_XOR_4(s02, s03, s12, s23);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s01, s02, s13, s20, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x23: {
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D1,D5; A^-1 = [5 3 7; f 2 d; b 1 a]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s02, s03, s10, s13, s20, s22);
                        SIMD_TYPE r01 = RING_XOR_9(s01, s02, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r03 = RING_XOR_5(s01, s02, s03, s12, s21);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s01, s13, s20, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s02, s10, s13, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s03, s11, s13, s20, s21);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s12, s13, s20, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_8(s00, s01, s02, s03, s10, s21, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_4(s00, s11, s20, s21);
                        SIMD_TYPE r22 = RING_XOR_4(s02, s03, s12, s23);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s01, s02, s13, s20, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x43: {
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D1,D6; A^-1 = [d f 3; e c 2; 2 3 1]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s01, s03, s10, s11, s20, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s02, s03, s10, s12, s20, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s01, s10, s13, s21, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s02, s03, s10, s22);

                        SIMD_TYPE r10 = RING_XOR_4(s01, s11, s13, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s02, s11, s12, s13, s20, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s02, s03, s10, s11, s12, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s03, s10, s12, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_4(s03, s10, s13, s20);
                        SIMD_TYPE r21 = RING_XOR_6(s00, s03, s10, s11, s13, s21);
                        SIMD_TYPE r22 = RING_XOR_6(s01, s03, s11, s12, s13, s22);
                        SIMD_TYPE r23 = RING_XOR_4(s02, s03, s12, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x83: {
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D1,D7; A^-1 = [4 d 8; 8 9 1; d 4 9]. */
                        SIMD_TYPE r00 = RING_XOR_7(s02, s03, s10, s11, s13, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_5(s02, s12, s13, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s02, s10, s11, s21);
                        SIMD_TYPE r03 = RING_XOR_7(s01, s02, s10, s12, s13, s20, s21);

                        SIMD_TYPE r10 = RING_XOR_6(s01, s02, s10, s11, s12, s20);
                        SIMD_TYPE r11 = RING_XOR_4(s01, s03, s13, s21);
                        SIMD_TYPE r12 = RING_XOR_4(s01, s11, s12, s22);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s01, s10, s11, s13, s23);

                        SIMD_TYPE r20 = RING_XOR_8(s00, s01, s03, s12, s13, s20, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_4(s02, s03, s12, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s01, s10, s12, s21, s22);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s02, s03, s11, s12, s20, s21, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x0d: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D2,D3; A^-1 = [b 9 3; d 6 b; 7 f 8]. */
                        SIMD_TYPE r00 = RING_XOR_9(s00, s01, s02, s03, s10, s11, s12, s20, s23);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s13, s20, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s02, s03, s11, s12, s21, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_7(s00, s01, s02, s10, s11, s13, s22);

                        SIMD_TYPE r10 = RING_XOR_8(s00, s01, s03, s12, s20, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s02, s03, s10, s12, s13, s20);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s10, s11, s12, s13, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_8(s00, s02, s03, s11, s13, s20, s21, s22);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s02, s10, s11, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_8(s00, s01, s02, s03, s10, s12, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s01, s03, s10, s13, s21);
                        SIMD_TYPE r23 = RING_XOR_4(s01, s10, s20, s21);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x15: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D2,D4; A^-1 = [2 7 4; 7 a d; 4 d 9]. */
                        SIMD_TYPE r00 = RING_XOR_5(s03, s10, s12, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s03, s10, s11, s12, s13, s22);
                        SIMD_TYPE r02 = RING_XOR_7(s01, s03, s10, s11, s13, s20, s22);
                        SIMD_TYPE r03 = RING_XOR_5(s02, s03, s11, s21, s22);

                        SIMD_TYPE r10 = RING_XOR_8(s00, s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s02, s03, s10, s11, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s01, s03, s13, s20, s21);
                        SIMD_TYPE r13 = RING_XOR_8(s01, s10, s11, s12, s13, s20, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_8(s02, s03, s10, s11, s13, s20, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_4(s02, s12, s13, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s02, s10, s11, s21, s22);
                        SIMD_TYPE r23 = RING_XOR_8(s01, s02, s10, s12, s13, s20, s21, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x25: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D2,D5; A^-1 = [c 6 b; a 9 3; 7 f 8]. */
                        SIMD_TYPE r00 = RING_XOR_7(s01, s03, s12, s20, s21, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s01, s02, s03, s10, s12, s13, s20);
                        SIMD_TYPE r02 = RING_XOR_9(s00, s01, s02, s10, s11, s12, s13, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_7(s00, s02, s11, s13, s20, s21, s22);

                        SIMD_TYPE r10 = RING_XOR_8(s01, s02, s03, s10, s11, s12, s20, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s01, s13, s20, s21, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s03, s11, s12, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_8(s00, s01, s02, s03, s10, s11, s13, s22);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s02, s10, s11, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_8(s00, s01, s02, s03, s10, s12, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s01, s03, s10, s13, s21);
                        SIMD_TYPE r23 = RING_XOR_4(s01, s10, s20, s21);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x45: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D2,D6; A^-1 = [6 a d; 3 7 4; 4 d 9]. */
                        SIMD_TYPE r00 = RING_XOR_7(s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s02, s03, s10, s11, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s01, s02, s03, s13, s20, s21);
                        SIMD_TYPE r03 = RING_XOR_9(s01, s03, s10, s11, s12, s13, s20, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s03, s10, s12, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s03, s10, s11, s12, s13, s22);
                        SIMD_TYPE r12 = RING_XOR_8(s01, s02, s03, s10, s11, s13, s20, s22);
                        SIMD_TYPE r13 = RING_XOR_4(s02, s11, s21, s22);

                        SIMD_TYPE r20 = RING_XOR_8(s02, s03, s10, s11, s13, s20, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_4(s02, s12, s13, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s02, s10, s11, s21, s22);
                        SIMD_TYPE r23 = RING_XOR_8(s01, s02, s10, s12, s13, s20, s21, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x85: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D2,D7; A^-1 = [5 8 c; 6 1 7; 2 9 b]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s02, s03, s11, s12, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s01, s02, s11, s13, s21, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s11, s20, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_7(s01, s02, s03, s10, s11, s20, s22);

                        SIMD_TYPE r10 = RING_XOR_4(s02, s10, s20, s22);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s02, s03, s11, s20, s21, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s02, s03, s12, s20, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_4(s01, s03, s13, s21);

                        SIMD_TYPE r20 = RING_XOR_8(s03, s10, s11, s12, s20, s21, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_4(s00, s03, s13, s20);
                        SIMD_TYPE r22 = RING_XOR_6(s01, s03, s11, s12, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_8(s02, s03, s10, s11, s13, s20, s21, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x19: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D3,D4; A^-1 = [1 e e; 1 7 6; 1 9 8]. */
                        SIMD_TYPE r00 = RING_XOR_3(s00, s11, s21);
                        SIMD_TYPE r01 = RING_XOR_7(s01, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_7(s02, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_5(s03, s10, s13, s20, s23);

                        SIMD_TYPE r10 = RING_XOR_4(s00, s10, s12, s22);
                        SIMD_TYPE r11 = RING_XOR_8(s01, s10, s11, s12, s13, s20, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s02, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_4(s03, s11, s21, s23);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s10, s11, s12, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_4(s01, s13, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_4(s02, s11, s12, s21);
                        SIMD_TYPE r23 = RING_XOR_6(s03, s10, s11, s13, s20, s21);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x29: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D3,D5; A^-1 = [6 f 8; a 9 3; d 6 b]. */
                        SIMD_TYPE r00 = RING_XOR_5(s02, s10, s11, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s02, s03, s10, s12, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s01, s02, s03, s10, s13, s21);
                        SIMD_TYPE r03 = RING_XOR_5(s01, s03, s10, s20, s21);

                        SIMD_TYPE r10 = RING_XOR_8(s01, s02, s03, s10, s11, s12, s20, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s01, s13, s20, s21, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s03, s11, s12, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_8(s00, s01, s02, s03, s10, s11, s13, s22);

                        SIMD_TYPE r20 = RING_XOR_8(s00, s01, s03, s12, s20, s21, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s02, s03, s10, s12, s13, s20);
                        SIMD_TYPE r22 = RING_XOR_8(s00, s01, s10, s11, s12, s13, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s02, s03, s11, s13, s20, s21, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x49: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D3,D6; A^-1 = [3 3 1; e c 2; c f 3]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s03, s10, s13, s20);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s01, s03, s10, s11, s13, s21);
                        SIMD_TYPE r02 = RING_XOR_7(s01, s02, s03, s11, s12, s13, s22);
                        SIMD_TYPE r03 = RING_XOR_3(s02, s12, s23);

                        SIMD_TYPE r10 = RING_XOR_4(s01, s11, s13, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s02, s11, s12, s13, s20, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s02, s03, s10, s11, s12, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s03, s10, s12, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_6(s01, s03, s10, s11, s20, s23);
                        SIMD_TYPE r21 = RING_XOR_8(s01, s02, s03, s10, s12, s20, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_8(s00, s01, s02, s10, s13, s21, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_4(s00, s02, s10, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x89: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D3,D7; A^-1 = [d 1 d; 4 b f; 8 a 2]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s01, s03, s10, s20, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_5(s02, s03, s11, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s01, s12, s20, s21);
                        SIMD_TYPE r03 = RING_XOR_7(s00, s02, s03, s13, s20, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_8(s02, s03, s10, s11, s12, s13, s20, s21);
                        SIMD_TYPE r11 = RING_XOR_4(s02, s10, s20, s22);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s02, s12, s13, s20, s23);
                        SIMD_TYPE r13 = RING_XOR_6(s01, s02, s10, s11, s12, s20);

                        SIMD_TYPE r20 = RING_XOR_6(s01, s02, s11, s12, s13, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s01, s03, s10, s11, s20, s23);
                        SIMD_TYPE r22 = RING_XOR_4(s01, s13, s21, s23);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s01, s10, s11, s12, s13, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x31: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D4,D5; A^-1 = [a 1 a; f 2 d; 4 3 7]. */
                        SIMD_TYPE r00 = RING_XOR_7(s01, s02, s03, s10, s21, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s01, s11, s20, s21);
                        SIMD_TYPE r02 = RING_XOR_3(s03, s12, s23);
                        SIMD_TYPE r03 = RING_XOR_9(s00, s01, s02, s03, s13, s20, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s01, s13, s20, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s02, s10, s13, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s03, s11, s13, s20, s21);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s12, s13, s20, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_6(s02, s03, s10, s13, s20, s22);
                        SIMD_TYPE r21 = RING_XOR_8(s02, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_8(s00, s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r23 = RING_XOR_4(s01, s02, s12, s21);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x51: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D4,D6; A^-1 = [5 d 9; 3 7 4; 7 a d]. */
                        SIMD_TYPE r00 = RING_XOR_9(s00, s02, s03, s10, s11, s13, s20, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_5(s01, s02, s12, s13, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s10, s11, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_9(s01, s02, s03, s10, s12, s13, s20, s21, s23);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s03, s10, s12, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s03, s10, s11, s12, s13, s22);
                        SIMD_TYPE r12 = RING_XOR_8(s01, s02, s03, s10, s11, s13, s20, s22);
                        SIMD_TYPE r13 = RING_XOR_4(s02, s11, s21, s22);

                        SIMD_TYPE r20 = RING_XOR_8(s00, s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r21 = RING_XOR_8(s00, s01, s02, s03, s10, s11, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s01, s03, s13, s20, s21);
                        SIMD_TYPE r23 = RING_XOR_8(s01, s10, s11, s12, s13, s20, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x91: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D4,D7; A^-1 = [8 f 6; 7 4 3; e b 5]. */
                        SIMD_TYPE r00 = RING_XOR_5(s01, s02, s10, s11, s22);
                        SIMD_TYPE r01 = RING_XOR_7(s01, s03, s10, s12, s20, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s01, s10, s13, s20, s21, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s01, s10, s21, s23);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s02, s12, s13, s20, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s02, s03, s12, s20, s21, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s03, s10, s12, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_4(s01, s11, s12, s22);

                        SIMD_TYPE r20 = RING_XOR_8(s01, s10, s11, s12, s13, s20, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s00, s01, s02, s10, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s02, s03, s12, s13, s20);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s03, s10, s11, s12, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x61: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D5,D6; A^-1 = [1 c c; 1 8 9; 1 4 5]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s11, s13, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s01, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s02, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_5(s03, s10, s12, s20, s22);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s11, s12, s20, s21, s22);
                        SIMD_TYPE r11 = RING_XOR_4(s01, s11, s13, s23);
                        SIMD_TYPE r12 = RING_XOR_4(s02, s11, s21, s22);
                        SIMD_TYPE r13 = RING_XOR_6(s03, s10, s11, s20, s21, s23);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s12, s13, s20, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_4(s01, s12, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_4(s02, s10, s12, s20);
                        SIMD_TYPE r23 = RING_XOR_6(s03, s11, s12, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xa1: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D5,D7; A^-1 = [e e 1; c 2 e; 3 c f]. */
                        SIMD_TYPE r00 = RING_XOR_3(s01, s11, s20);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s01, s02, s10, s11, s12, s21);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s02, s03, s10, s12, s13, s22);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s03, s10, s13, s23);

                        SIMD_TYPE r10 = RING_XOR_4(s01, s03, s13, s21);
                        SIMD_TYPE r11 = RING_XOR_8(s01, s02, s03, s10, s13, s20, s21, s22);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s02, s11, s13, s20, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s02, s12, s13, s20, s23);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s03, s11, s13, s20, s21);
                        SIMD_TYPE r21 = RING_XOR_8(s00, s01, s03, s11, s12, s13, s20, s22);
                        SIMD_TYPE r22 = RING_XOR_8(s01, s02, s03, s10, s11, s12, s20, s23);
                        SIMD_TYPE r23 = RING_XOR_4(s02, s10, s12, s20);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xc1: {
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = zero;
                        SIMD_TYPE c11 = zero;
                        SIMD_TYPE c12 = zero;
                        SIMD_TYPE c13 = zero;

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D0,D6,D7; A^-1 = [7 2 4; d 7 a; b 5 e]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s02, s13, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s01, s02, s03, s10, s13, s22);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s01, s03, s11, s13, s20, s22);
                        SIMD_TYPE r03 = RING_XOR_5(s01, s12, s13, s21, s22);

                        SIMD_TYPE r10 = RING_XOR_8(s00, s01, s03, s10, s12, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s02, s03, s10, s11, s12, s13, s20, s21);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s01, s10, s11, s13, s23);
                        SIMD_TYPE r13 = RING_XOR_8(s00, s02, s03, s11, s20, s21, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_8(s00, s01, s02, s03, s10, s12, s13, s21);
                        SIMD_TYPE r21 = RING_XOR_6(s00, s11, s12, s20, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_6(s02, s03, s10, s20, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s01, s02, s11, s12, s13, s20, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x0e: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D2,D3; A^-1 = [2 4 5; a 8 9; 9 c c]. */
                        SIMD_TYPE r00 = RING_XOR_6(s03, s12, s13, s20, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s03, s12, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_5(s01, s03, s10, s12, s20);
                        SIMD_TYPE r03 = RING_XOR_7(s02, s03, s11, s12, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_8(s01, s02, s03, s11, s12, s20, s21, s22);
                        SIMD_TYPE r11 = RING_XOR_5(s00, s01, s11, s13, s23);
                        SIMD_TYPE r12 = RING_XOR_4(s03, s11, s21, s22);
                        SIMD_TYPE r13 = RING_XOR_9(s00, s01, s02, s03, s10, s11, s20, s21, s23);

                        SIMD_TYPE r20 = RING_XOR_7(s00, s01, s02, s11, s13, s21, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s03, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_8(s01, s02, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r23 = RING_XOR_7(s00, s01, s03, s10, s12, s20, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x16: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D2,D4; A^-1 = [b 8 9; 3 4 5; 9 c c]. */
                        SIMD_TYPE r00 = RING_XOR_9(s00, s01, s02, s03, s11, s12, s20, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_4(s00, s11, s13, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s02, s03, s11, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_8(s00, s01, s02, s10, s11, s20, s21, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s00, s03, s12, s13, s20, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s01, s03, s12, s21, s22);
                        SIMD_TYPE r12 = RING_XOR_6(s01, s02, s03, s10, s12, s20);
                        SIMD_TYPE r13 = RING_XOR_6(s02, s11, s12, s21, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_7(s00, s01, s02, s11, s13, s21, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s03, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_8(s01, s02, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r23 = RING_XOR_7(s00, s01, s03, s10, s12, s20, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x26: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D2,D5; A^-1 = [d 9 8; 9 7 6; 5 e e]. */
                        SIMD_TYPE r00 = RING_XOR_8(s00, s01, s03, s10, s11, s12, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_5(s02, s03, s13, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s01, s11, s12, s21);
                        SIMD_TYPE r03 = RING_XOR_8(s00, s02, s03, s10, s11, s13, s20, s21);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s01, s02, s10, s12, s22);
                        SIMD_TYPE r11 = RING_XOR_8(s03, s10, s11, s12, s13, s20, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_9(s01, s02, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s01, s03, s11, s21, s23);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s02, s03, s11, s21);
                        SIMD_TYPE r21 = RING_XOR_8(s01, s02, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_7(s00, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_7(s01, s02, s03, s10, s13, s20, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x46: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D2,D6; A^-1 = [8 7 6; c 9 8; 5 e e]. */
                        SIMD_TYPE r00 = RING_XOR_5(s01, s02, s10, s12, s22);
                        SIMD_TYPE r01 = RING_XOR_9(s01, s03, s10, s11, s12, s13, s20, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_8(s01, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_5(s00, s01, s11, s21, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s01, s03, s10, s11, s12, s21, s22);
                        SIMD_TYPE r11 = RING_XOR_6(s01, s02, s03, s13, s21, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s01, s02, s11, s12, s21);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s02, s10, s11, s13, s20, s21);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s02, s03, s11, s21);
                        SIMD_TYPE r21 = RING_XOR_8(s01, s02, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_7(s00, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_7(s01, s02, s03, s10, s13, s20, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x86: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D2,D7; A^-1 = [9 2 3; 7 f e; f d d]. */
                        SIMD_TYPE r00 = RING_XOR_6(s00, s01, s02, s13, s20, s23);
                        SIMD_TYPE r01 = RING_XOR_6(s03, s10, s13, s20, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s01, s02, s11, s13, s21, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_6(s00, s01, s03, s12, s13, s22);

                        SIMD_TYPE r10 = RING_XOR_5(s00, s02, s10, s11, s21);
                        SIMD_TYPE r11 = RING_XOR_9(s00, s01, s02, s03, s10, s12, s20, s21, s22);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s03, s10, s13, s20, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_4(s01, s10, s20, s23);

                        SIMD_TYPE r20 = RING_XOR_8(s00, s01, s10, s11, s13, s20, s21, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s00, s02, s12, s13, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s03, s10, s11, s20, s21);
                        SIMD_TYPE r23 = RING_XOR_7(s00, s10, s12, s13, s20, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x1a: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D3,D4; A^-1 = [8 c c; 3 4 5; a 8 9]. */
                        SIMD_TYPE r00 = RING_XOR_6(s01, s02, s11, s13, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_8(s01, s03, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s01, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_6(s00, s01, s10, s12, s20, s22);

                        SIMD_TYPE r10 = RING_XOR_7(s00, s03, s12, s13, s20, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s01, s03, s12, s21, s22);
                        SIMD_TYPE r12 = RING_XOR_6(s01, s02, s03, s10, s12, s20);
                        SIMD_TYPE r13 = RING_XOR_6(s02, s11, s12, s21, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_8(s01, s02, s03, s11, s12, s20, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_5(s00, s01, s11, s13, s23);
                        SIMD_TYPE r22 = RING_XOR_4(s03, s11, s21, s22);
                        SIMD_TYPE r23 = RING_XOR_9(s00, s01, s02, s03, s10, s11, s20, s21, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x2a: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D3,D5; A^-1 = [1 d 9; 1 a d; 1 7 4]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s10, s11, s13, s20, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_4(s01, s12, s13, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s02, s10, s11, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_7(s03, s10, s12, s13, s20, s21, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s00, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_5(s01, s10, s11, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_4(s02, s13, s20, s21);
                        SIMD_TYPE r13 = RING_XOR_8(s03, s10, s11, s12, s13, s20, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s10, s12, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s01, s10, s11, s12, s13, s22);
                        SIMD_TYPE r22 = RING_XOR_6(s02, s10, s11, s13, s20, s22);
                        SIMD_TYPE r23 = RING_XOR_4(s03, s11, s21, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x4a: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D3,D6; A^-1 = [3 3 1; d f 3; f c 2]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s03, s10, s13, s20);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s01, s03, s10, s11, s13, s21);
                        SIMD_TYPE r02 = RING_XOR_7(s01, s02, s03, s11, s12, s13, s22);
                        SIMD_TYPE r03 = RING_XOR_3(s02, s12, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s00, s01, s03, s10, s11, s20, s23);
                        SIMD_TYPE r11 = RING_XOR_7(s02, s03, s10, s12, s20, s21, s23);
                        SIMD_TYPE r12 = RING_XOR_7(s00, s01, s10, s13, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_5(s00, s02, s03, s10, s22);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s01, s11, s13, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s00, s02, s11, s12, s13, s20, s23);
                        SIMD_TYPE r22 = RING_XOR_7(s00, s03, s10, s11, s12, s21, s23);
                        SIMD_TYPE r23 = RING_XOR_5(s00, s10, s12, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x8a: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D3,D7; A^-1 = [f 6 f; c 8 7; 2 e 8]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s01, s12, s20, s21);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s02, s10, s12, s13, s20, s22);
                        SIMD_TYPE r02 = RING_XOR_8(s00, s03, s10, s11, s12, s13, s20, s23);
                        SIMD_TYPE r03 = RING_XOR_4(s00, s11, s13, s20);

                        SIMD_TYPE r10 = RING_XOR_6(s01, s03, s11, s12, s20, s22);
                        SIMD_TYPE r11 = RING_XOR_9(s01, s02, s03, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_7(s00, s01, s02, s11, s20, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_5(s00, s02, s10, s11, s21);

                        SIMD_TYPE r20 = RING_XOR_4(s03, s11, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_7(s00, s03, s10, s11, s12, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s01, s03, s10, s12, s13, s21);
                        SIMD_TYPE r23 = RING_XOR_6(s02, s03, s10, s13, s20, s21);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x32: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D4,D5; A^-1 = [a 1 a; 5 3 7; e 2 d]. */
                        SIMD_TYPE r00 = RING_XOR_7(s01, s02, s03, s10, s21, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s01, s11, s20, s21);
                        SIMD_TYPE r02 = RING_XOR_3(s03, s12, s23);
                        SIMD_TYPE r03 = RING_XOR_9(s00, s01, s02, s03, s13, s20, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s00, s02, s03, s10, s13, s20, s22);
                        SIMD_TYPE r11 = RING_XOR_9(s01, s02, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_7(s00, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_5(s01, s02, s03, s12, s21);

                        SIMD_TYPE r20 = RING_XOR_5(s01, s13, s20, s21, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s00, s01, s02, s10, s13, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_7(s00, s02, s03, s11, s13, s20, s21);
                        SIMD_TYPE r23 = RING_XOR_7(s00, s03, s12, s13, s20, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x52: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D4,D6; A^-1 = [1 f 8; 1 6 b; 1 9 3]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s10, s11, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_5(s01, s10, s12, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_4(s02, s10, s13, s21);
                        SIMD_TYPE r03 = RING_XOR_4(s03, s10, s20, s21);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s12, s20, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_5(s01, s10, s12, s13, s20);
                        SIMD_TYPE r12 = RING_XOR_7(s02, s10, s11, s12, s13, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_6(s03, s11, s13, s20, s21, s22);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s10, s11, s12, s20, s23);
                        SIMD_TYPE r21 = RING_XOR_5(s01, s13, s20, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s02, s11, s12, s21, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_5(s03, s10, s11, s13, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x92: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D4,D7; A^-1 = [5 a 4; 8 5 f; c f b]. */
                        SIMD_TYPE r00 = RING_XOR_8(s00, s02, s03, s11, s12, s13, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_5(s01, s02, s10, s11, s22);
                        SIMD_TYPE r02 = RING_XOR_4(s00, s13, s20, s22);
                        SIMD_TYPE r03 = RING_XOR_9(s01, s02, s03, s10, s11, s12, s13, s21, s22);

                        SIMD_TYPE r10 = RING_XOR_7(s01, s02, s10, s12, s13, s20, s21);
                        SIMD_TYPE r11 = RING_XOR_6(s01, s03, s11, s12, s20, s22);
                        SIMD_TYPE r12 = RING_XOR_4(s01, s10, s20, s23);
                        SIMD_TYPE r13 = RING_XOR_6(s00, s01, s11, s12, s13, s20);

                        SIMD_TYPE r20 = RING_XOR_8(s01, s03, s10, s11, s20, s21, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s01, s02, s03, s10, s12, s20);
                        SIMD_TYPE r22 = RING_XOR_7(s00, s01, s02, s10, s13, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_6(s00, s02, s10, s20, s21, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x62: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D5,D6; A^-1 = [4 e e; c 9 8; 9 7 6]. */
                        SIMD_TYPE r00 = RING_XOR_4(s02, s03, s11, s21);
                        SIMD_TYPE r01 = RING_XOR_7(s02, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_8(s00, s02, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_6(s01, s02, s10, s13, s20, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s01, s03, s10, s11, s12, s21, s22);
                        SIMD_TYPE r11 = RING_XOR_6(s01, s02, s03, s13, s21, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s01, s02, s11, s12, s21);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s02, s10, s11, s13, s20, s21);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s01, s02, s10, s12, s22);
                        SIMD_TYPE r21 = RING_XOR_8(s03, s10, s11, s12, s13, s20, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_9(s01, s02, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_6(s00, s01, s03, s11, s21, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xa2: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D5,D7; A^-1 = [b b 7; d 3 5; 7 8 2]. */
                        SIMD_TYPE r00 =
                                RING_XOR_10(s00, s01, s02, s03, s10, s11, s12, s13, s20, s22);
                        SIMD_TYPE r01 = RING_XOR_6(s00, s10, s20, s21, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s02, s03, s12, s13, s20, s21, s23);
                        SIMD_TYPE r03 = RING_XOR_7(s00, s01, s02, s10, s11, s12, s21);

                        SIMD_TYPE r10 = RING_XOR_8(s00, s01, s03, s10, s13, s20, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_7(s02, s03, s10, s11, s13, s21, s22);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s01, s11, s12, s13, s20);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s02, s03, s12, s21, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s02, s11, s12, s23);
                        SIMD_TYPE r21 = RING_XOR_8(s00, s01, s02, s03, s11, s13, s20, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s01, s03, s11, s21, s23);
                        SIMD_TYPE r23 = RING_XOR_5(s01, s10, s11, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xc2: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = zero;
                        SIMD_TYPE c21 = zero;
                        SIMD_TYPE c22 = zero;
                        SIMD_TYPE c23 = zero;

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D1,D6,D7; A^-1 = [2 5 a; 6 4 c; 5 1 6]. */
                        SIMD_TYPE r00 = RING_XOR_7(s03, s10, s12, s13, s21, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_6(s00, s03, s11, s12, s20, s21);
                        SIMD_TYPE r02 = RING_XOR_4(s01, s03, s10, s23);
                        SIMD_TYPE r03 = RING_XOR_9(s02, s03, s11, s12, s13, s20, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_5(s02, s12, s13, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_7(s00, s02, s03, s12, s21, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_9(s00, s01, s02, s03, s10, s12, s20, s21, s22);
                        SIMD_TYPE r13 = RING_XOR_6(s01, s03, s11, s12, s20, s22);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s02, s03, s10, s22);
                        SIMD_TYPE r21 = RING_XOR_6(s01, s02, s11, s20, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s12, s20, s21, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_6(s01, s02, s03, s13, s21, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x1c: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D3,D4; A^-1 = [8 c c; b 8 9; 2 4 5]. */
                        SIMD_TYPE r00 = RING_XOR_6(s01, s02, s11, s13, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_8(s01, s03, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s01, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_6(s00, s01, s10, s12, s20, s22);

                        SIMD_TYPE r10 = RING_XOR_9(s00, s01, s02, s03, s11, s12, s20, s21, s22);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s11, s13, s23);
                        SIMD_TYPE r12 = RING_XOR_5(s02, s03, s11, s21, s22);
                        SIMD_TYPE r13 = RING_XOR_8(s00, s01, s02, s10, s11, s20, s21, s23);

                        SIMD_TYPE r20 = RING_XOR_6(s03, s12, s13, s20, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_5(s00, s03, s12, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_5(s01, s03, s10, s12, s20);
                        SIMD_TYPE r23 = RING_XOR_7(s02, s03, s11, s12, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x2c: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D3,D5; A^-1 = [6 f 8; c 6 b; b 9 3]. */
                        SIMD_TYPE r00 = RING_XOR_5(s02, s10, s11, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s02, s03, s10, s12, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s01, s02, s03, s10, s13, s21);
                        SIMD_TYPE r03 = RING_XOR_5(s01, s03, s10, s20, s21);

                        SIMD_TYPE r10 = RING_XOR_7(s01, s03, s12, s20, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_7(s01, s02, s03, s10, s12, s13, s20);
                        SIMD_TYPE r12 = RING_XOR_9(s00, s01, s02, s10, s11, s12, s13, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s02, s11, s13, s20, s21, s22);

                        SIMD_TYPE r20 = RING_XOR_9(s00, s01, s02, s03, s10, s11, s12, s20, s23);
                        SIMD_TYPE r21 = RING_XOR_5(s00, s13, s20, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_7(s02, s03, s11, s12, s21, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_7(s00, s01, s02, s10, s11, s13, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x4c: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D3,D6; A^-1 = [1 1 a; 1 3 7; 1 2 d]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s10, s21, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_4(s01, s11, s20, s21);
                        SIMD_TYPE r02 = RING_XOR_3(s02, s12, s23);
                        SIMD_TYPE r03 = RING_XOR_6(s03, s13, s20, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_5(s00, s10, s13, s20, s22);
                        SIMD_TYPE r11 = RING_XOR_8(s01, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_7(s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_3(s03, s12, s21);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s13, s20, s21, s23);
                        SIMD_TYPE r21 = RING_XOR_5(s01, s10, s13, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_5(s02, s11, s13, s20, s21);
                        SIMD_TYPE r23 = RING_XOR_6(s03, s12, s13, s20, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x8c: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D3,D7; A^-1 = [2 7 2; d 4 6; e 3 4]. */
                        SIMD_TYPE r00 = RING_XOR_4(s03, s10, s12, s23);
                        SIMD_TYPE r01 = RING_XOR_8(s00, s03, s10, s11, s12, s13, s20, s23);
                        SIMD_TYPE r02 = RING_XOR_7(s01, s03, s10, s11, s13, s21, s23);
                        SIMD_TYPE r03 = RING_XOR_5(s02, s03, s11, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_6(s00, s01, s03, s12, s13, s22);
                        SIMD_TYPE r11 = RING_XOR_6(s02, s03, s12, s20, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s10, s12, s20, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s02, s03, s11, s12, s21, s23);

                        SIMD_TYPE r20 = RING_XOR_5(s01, s10, s13, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s00, s01, s02, s10, s11, s13, s22);
                        SIMD_TYPE r22 = RING_XOR_8(s00, s02, s03, s11, s12, s13, s20, s22);
                        SIMD_TYPE r23 = RING_XOR_5(s00, s03, s12, s21, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x34: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D4,D5; A^-1 = [1 3 1; 1 f 3; 1 c 2]. */
                        SIMD_TYPE r00 = RING_XOR_4(s00, s10, s13, s20);
                        SIMD_TYPE r01 = RING_XOR_5(s01, s10, s11, s13, s21);
                        SIMD_TYPE r02 = RING_XOR_5(s02, s11, s12, s13, s22);
                        SIMD_TYPE r03 = RING_XOR_3(s03, s12, s23);

                        SIMD_TYPE r10 = RING_XOR_5(s00, s10, s11, s20, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s01, s10, s12, s20, s21, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s02, s10, s13, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_3(s03, s10, s22);

                        SIMD_TYPE r20 = RING_XOR_4(s00, s11, s13, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s01, s11, s12, s13, s20, s23);
                        SIMD_TYPE r22 = RING_XOR_6(s02, s10, s11, s12, s21, s23);
                        SIMD_TYPE r23 = RING_XOR_5(s03, s10, s12, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x54: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D4,D6; A^-1 = [5 d 9; 6 a d; 2 7 4]. */
                        SIMD_TYPE r00 = RING_XOR_9(s00, s02, s03, s10, s11, s13, s20, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_5(s01, s02, s12, s13, s23);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s10, s11, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_9(s01, s02, s03, s10, s12, s13, s20, s21, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_7(s00, s02, s03, s10, s11, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_7(s00, s01, s02, s03, s13, s20, s21);
                        SIMD_TYPE r13 = RING_XOR_9(s01, s03, s10, s11, s12, s13, s20, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_5(s03, s10, s12, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s00, s03, s10, s11, s12, s13, s22);
                        SIMD_TYPE r22 = RING_XOR_7(s01, s03, s10, s11, s13, s20, s22);
                        SIMD_TYPE r23 = RING_XOR_5(s02, s03, s11, s21, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x94: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D4,D7; A^-1 = [a b 8; e 9 2; 5 2 a]. */
                        SIMD_TYPE r00 = RING_XOR_9(s01, s02, s03, s10, s11, s12, s13, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s01, s10, s21, s23);
                        SIMD_TYPE r02 = RING_XOR_4(s03, s12, s13, s21);
                        SIMD_TYPE r03 = RING_XOR_9(s00, s01, s02, s03, s10, s11, s12, s20, s21);

                        SIMD_TYPE r10 = RING_XOR_5(s01, s10, s11, s12, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s01, s02, s13, s20, s23);
                        SIMD_TYPE r12 = RING_XOR_7(s00, s02, s03, s11, s12, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s03, s10, s11, s13, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_7(s00, s02, s03, s13, s21, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s01, s02, s10, s13, s20, s21);
                        SIMD_TYPE r22 = RING_XOR_4(s00, s11, s13, s23);
                        SIMD_TYPE r23 = RING_XOR_9(s01, s02, s03, s12, s13, s20, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x64: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D5,D6; A^-1 = [4 e e; 8 7 6; d 9 8]. */
                        SIMD_TYPE r00 = RING_XOR_4(s02, s03, s11, s21);
                        SIMD_TYPE r01 = RING_XOR_7(s02, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_8(s00, s02, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_6(s01, s02, s10, s13, s20, s23);

                        SIMD_TYPE r10 = RING_XOR_5(s01, s02, s10, s12, s22);
                        SIMD_TYPE r11 = RING_XOR_9(s01, s03, s10, s11, s12, s13, s20, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s01, s10, s11, s13, s20, s21, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_5(s00, s01, s11, s21, s23);

                        SIMD_TYPE r20 = RING_XOR_8(s00, s01, s03, s10, s11, s12, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_5(s02, s03, s13, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_5(s00, s01, s11, s12, s21);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s02, s03, s10, s11, s13, s20, s21);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xa4: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D5,D7; A^-1 = [8 8 5; 3 d 4; a 5 1]. */
                        SIMD_TYPE r00 = RING_XOR_7(s01, s02, s11, s12, s20, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_6(s01, s03, s11, s13, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_3(s01, s11, s20);
                        SIMD_TYPE r03 = RING_XOR_7(s00, s01, s10, s11, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s00, s03, s10, s11, s13, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s01, s03, s12, s13, s22);
                        SIMD_TYPE r12 = RING_XOR_7(s01, s02, s03, s10, s11, s20, s22);
                        SIMD_TYPE r13 = RING_XOR_6(s02, s10, s12, s13, s21, s22);

                        SIMD_TYPE r20 = RING_XOR_7(s01, s02, s03, s10, s12, s13, s20);
                        SIMD_TYPE r21 = RING_XOR_5(s00, s01, s11, s12, s21);
                        SIMD_TYPE r22 = RING_XOR_3(s03, s10, s22);
                        SIMD_TYPE r23 = RING_XOR_8(s00, s01, s02, s03, s11, s12, s13, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xc4: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = zero;
                        SIMD_TYPE c31 = zero;
                        SIMD_TYPE c32 = zero;
                        SIMD_TYPE c33 = zero;

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D2,D6,D7; A^-1 = [9 6 c; c a f; 4 c 3]. */
                        SIMD_TYPE r00 = RING_XOR_6(s00, s01, s02, s12, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s03, s10, s12, s13, s21, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_9(s01, s02, s10, s11, s12, s13, s20, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_7(s00, s01, s03, s11, s13, s20, s22);

                        SIMD_TYPE r10 = RING_XOR_7(s01, s03, s11, s12, s13, s20, s21);
                        SIMD_TYPE r11 = RING_XOR_7(s01, s02, s03, s10, s11, s20, s22);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s01, s02, s13, s20, s23);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s02, s10, s11, s12, s13, s20);

                        SIMD_TYPE r20 = RING_XOR_6(s02, s03, s11, s13, s20, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s02, s11, s12, s13, s20, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_8(s00, s02, s10, s11, s12, s21, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_5(s01, s02, s10, s12, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x38: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c7_base = buffs[6];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D3,D4,D5; A^-1 = [f e f; 8 b a; 6 5 5]. */
                        SIMD_TYPE r00 = RING_XOR_5(s00, s01, s11, s20, s21);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s02, s10, s11, s12, s20, s22);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s03, s10, s12, s13, s20, s23);
                        SIMD_TYPE r03 = RING_XOR_4(s00, s10, s13, s20);

                        SIMD_TYPE r10 = RING_XOR_9(s01, s02, s10, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_5(s01, s03, s10, s20, s21);
                        SIMD_TYPE r12 = RING_XOR_4(s01, s12, s13, s23);
                        SIMD_TYPE r13 = RING_XOR_9(s00, s01, s10, s11, s12, s20, s21, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_7(s02, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s00, s02, s03, s11, s12, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s01, s02, s03, s10, s20);
                        SIMD_TYPE r23 = RING_XOR_8(s01, s03, s11, s12, s13, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x58: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D3,D4,D6; A^-1 = [9 b a; e e f; 6 5 5]. */
                        SIMD_TYPE r00 =
                                RING_XOR_10(s00, s01, s02, s10, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_4(s03, s10, s20, s21);
                        SIMD_TYPE r02 = RING_XOR_5(s01, s02, s12, s13, s23);
                        SIMD_TYPE r03 =
                                RING_XOR_10(s00, s01, s03, s10, s11, s12, s20, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_4(s01, s11, s20, s21);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s02, s10, s11, s12, s20, s22);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s02, s03, s10, s12, s13, s20, s23);
                        SIMD_TYPE r13 = RING_XOR_5(s00, s03, s10, s13, s20);

                        SIMD_TYPE r20 = RING_XOR_7(s02, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s00, s02, s03, s11, s12, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_6(s00, s01, s02, s03, s10, s20);
                        SIMD_TYPE r23 = RING_XOR_8(s01, s03, s11, s12, s13, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x98: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c6_base = buffs[5];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D3,D4,D7; A^-1 = [6 c d; c d c; b 1 1]. */
                        SIMD_TYPE r00 = RING_XOR_6(s02, s11, s13, s20, s21, s23);
                        SIMD_TYPE r01 = RING_XOR_8(s00, s02, s03, s11, s12, s13, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_9(s00, s01, s02, s03, s10, s11, s12, s20, s21);
                        SIMD_TYPE r03 = RING_XOR_7(s01, s03, s10, s12, s20, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s01, s03, s10, s11, s13, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s01, s02, s03, s12, s13, s21, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s01, s02, s10, s11, s20, s21, s22);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s02, s10, s12, s13, s20, s22);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s01, s02, s03, s10, s20);
                        SIMD_TYPE r21 = RING_XOR_3(s00, s11, s21);
                        SIMD_TYPE r22 = RING_XOR_4(s02, s03, s12, s22);
                        SIMD_TYPE r23 = RING_XOR_5(s00, s01, s02, s13, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x68: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D3,D5,D6; A^-1 = [7 5 5; e e f; 8 b a]. */
                        SIMD_TYPE r00 = RING_XOR_8(s00, s02, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_8(s00, s01, s02, s03, s11, s12, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s01, s03, s10, s20);
                        SIMD_TYPE r03 = RING_XOR_7(s01, s11, s12, s13, s21, s22, s23);

                        SIMD_TYPE r10 = RING_XOR_4(s01, s11, s20, s21);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s02, s10, s11, s12, s20, s22);
                        SIMD_TYPE r12 = RING_XOR_8(s00, s02, s03, s10, s12, s13, s20, s23);
                        SIMD_TYPE r13 = RING_XOR_5(s00, s03, s10, s13, s20);

                        SIMD_TYPE r20 = RING_XOR_9(s01, s02, s10, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_5(s01, s03, s10, s20, s21);
                        SIMD_TYPE r22 = RING_XOR_4(s01, s12, s13, s23);
                        SIMD_TYPE r23 = RING_XOR_9(s00, s01, s10, s11, s12, s20, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xa8: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D3,D5,D7; A^-1 = [2 2 9; a 4 a; 9 6 3]. */
                        SIMD_TYPE r00 = RING_XOR_5(s03, s13, s20, s21, s22);
                        SIMD_TYPE r01 = RING_XOR_5(s00, s03, s10, s13, s23);
                        SIMD_TYPE r02 = RING_XOR_6(s01, s03, s11, s13, s21, s22);
                        SIMD_TYPE r03 = RING_XOR_7(s02, s03, s12, s13, s20, s21, s23);

                        SIMD_TYPE r10 = RING_XOR_8(s01, s02, s03, s12, s13, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_5(s00, s01, s12, s20, s21);
                        SIMD_TYPE r12 = RING_XOR_4(s03, s10, s12, s23);
                        SIMD_TYPE r13 =
                                RING_XOR_10(s00, s01, s02, s03, s11, s12, s20, s21, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_6(s00, s01, s02, s12, s20, s23);
                        SIMD_TYPE r21 = RING_XOR_7(s03, s10, s12, s13, s20, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_9(s01, s02, s10, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r23 = RING_XOR_6(s00, s01, s03, s11, s13, s22);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xc8: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c5_base = buffs[4];
                const unsigned char *c6_base = buffs[5];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = zero;
                        SIMD_TYPE c41 = zero;
                        SIMD_TYPE c42 = zero;
                        SIMD_TYPE c43 = zero;

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D3,D6,D7; A^-1 = [5 7 e; b 8 b; f f 5]. */
                        SIMD_TYPE r00 = RING_XOR_6(s00, s02, s03, s10, s12, s21);
                        SIMD_TYPE r01 = RING_XOR_9(s01, s02, s10, s11, s12, s13, s20, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_7(s00, s10, s11, s13, s20, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_6(s01, s02, s03, s11, s20, s23);

                        SIMD_TYPE r10 =
                                RING_XOR_10(s00, s01, s02, s03, s11, s12, s20, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_4(s00, s11, s13, s20);
                        SIMD_TYPE r12 = RING_XOR_5(s02, s03, s11, s22, s23);
                        SIMD_TYPE r13 = RING_XOR_8(s00, s01, s02, s10, s11, s20, s21, s22);

                        SIMD_TYPE r20 = RING_XOR_7(s00, s01, s10, s11, s20, s22, s23);
                        SIMD_TYPE r21 = RING_XOR_6(s00, s02, s10, s12, s21, s22);
                        SIMD_TYPE r22 = RING_XOR_5(s00, s03, s10, s13, s20);
                        SIMD_TYPE r23 = RING_XOR_5(s00, s10, s21, s22, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0x70: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c8_base = buffs[7];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = SIMD_LOAD_R0(c8_base + offset0);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(c8_base + offset1);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(c8_base + offset2);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(c8_base + offset3);

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D4,D5,D6; A^-1 = [7 5 5; 9 b a; f e f]. */
                        SIMD_TYPE r00 = RING_XOR_8(s00, s02, s10, s12, s13, s20, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_8(s00, s01, s02, s03, s11, s12, s21, s22);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s01, s03, s10, s20);
                        SIMD_TYPE r03 = RING_XOR_7(s01, s11, s12, s13, s21, s22, s23);

                        SIMD_TYPE r10 =
                                RING_XOR_10(s00, s01, s02, s10, s11, s12, s13, s21, s22, s23);
                        SIMD_TYPE r11 = RING_XOR_4(s03, s10, s20, s21);
                        SIMD_TYPE r12 = RING_XOR_5(s01, s02, s12, s13, s23);
                        SIMD_TYPE r13 =
                                RING_XOR_10(s00, s01, s03, s10, s11, s12, s20, s21, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s01, s11, s20, s21);
                        SIMD_TYPE r21 = RING_XOR_7(s00, s02, s10, s11, s12, s20, s22);
                        SIMD_TYPE r22 = RING_XOR_7(s00, s03, s10, s12, s13, s20, s23);
                        SIMD_TYPE r23 = RING_XOR_4(s00, s10, s13, s20);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xb0: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c7_base = buffs[6];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = SIMD_LOAD_R0(c7_base + offset0);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(c7_base + offset1);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(c7_base + offset2);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(c7_base + offset3);

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D4,D5,D7; A^-1 = [6 6 4; f 1 c; 8 7 8]. */
                        SIMD_TYPE r00 = RING_XOR_4(s02, s12, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s02, s03, s10, s12, s13, s22);
                        SIMD_TYPE r02 =
                                RING_XOR_10(s00, s01, s02, s03, s10, s11, s12, s13, s20, s22);
                        SIMD_TYPE r03 = RING_XOR_6(s01, s03, s11, s13, s21, s22);

                        SIMD_TYPE r10 = RING_XOR_5(s00, s01, s10, s21, s23);
                        SIMD_TYPE r11 = RING_XOR_6(s00, s02, s11, s21, s22, s23);
                        SIMD_TYPE r12 = RING_XOR_6(s00, s03, s12, s20, s21, s22);
                        SIMD_TYPE r13 = RING_XOR_4(s00, s13, s20, s22);

                        SIMD_TYPE r20 = RING_XOR_6(s01, s02, s10, s12, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_8(s01, s03, s10, s11, s12, s13, s21, s23);
                        SIMD_TYPE r22 = RING_XOR_5(s01, s10, s11, s13, s21);
                        SIMD_TYPE r23 = RING_XOR_5(s00, s01, s11, s20, s21);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xd0: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c6_base = buffs[5];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = zero;
                        SIMD_TYPE c51 = zero;
                        SIMD_TYPE c52 = zero;
                        SIMD_TYPE c53 = zero;

                        SIMD_TYPE c60 = SIMD_LOAD_R0(c6_base + offset0);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(c6_base + offset1);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(c6_base + offset2);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(c6_base + offset3);

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D4,D6,D7; A^-1 = [b 3 6; 9 d 1; 3 e 7]. */
                        SIMD_TYPE r00 = RING_XOR_7(s00, s01, s02, s03, s10, s13, s22);
                        SIMD_TYPE r01 = RING_XOR_7(s00, s10, s11, s13, s20, s22, s23);
                        SIMD_TYPE r02 = RING_XOR_9(s02, s03, s11, s12, s13, s20, s21, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_6(s00, s01, s02, s12, s21, s23);

                        SIMD_TYPE r10 = RING_XOR_7(s00, s01, s02, s10, s11, s13, s20);
                        SIMD_TYPE r11 = RING_XOR_4(s03, s12, s13, s21);
                        SIMD_TYPE r12 = RING_XOR_5(s01, s02, s10, s11, s22);
                        SIMD_TYPE r13 = RING_XOR_7(s00, s01, s03, s10, s12, s13, s23);

                        SIMD_TYPE r20 = RING_XOR_5(s00, s03, s11, s20, s22);
                        SIMD_TYPE r21 =
                                RING_XOR_10(s00, s01, s03, s10, s11, s12, s20, s21, s22, s23);
                        SIMD_TYPE r22 = RING_XOR_9(s01, s02, s03, s10, s12, s13, s20, s21, s23);
                        SIMD_TYPE r23 = RING_XOR_4(s02, s10, s13, s21);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }

        case 0xe0: {
                const unsigned char *c1_base = buffs[0];
                const unsigned char *c2_base = buffs[1];
                const unsigned char *c3_base = buffs[2];
                const unsigned char *c4_base = buffs[3];
                const unsigned char *c5_base = buffs[4];

                for (size_t i = 0; i < node_size; i += RING_8_3_SIMD_BATCH_BYTES) {
                        size_t offset0 = i;
                        size_t offset1 = i + SIMD_ITERATION_SIZE;
                        size_t offset2 = i + 2 * SIMD_ITERATION_SIZE;
                        size_t offset3 = i + 3 * SIMD_ITERATION_SIZE;
                        SIMD_TYPE zero = SIMD_SET_ZERO;

                        RING_KEEP_SLOT_OFFSETS(offset0, offset1, offset2, offset3);
                        SIMD_TYPE c10 = SIMD_LOAD_R0(c1_base + offset0);
                        SIMD_TYPE c11 = SIMD_LOAD_R0(c1_base + offset1);
                        SIMD_TYPE c12 = SIMD_LOAD_R0(c1_base + offset2);
                        SIMD_TYPE c13 = SIMD_LOAD_R0(c1_base + offset3);

                        SIMD_TYPE c20 = SIMD_LOAD_R0(c2_base + offset0);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(c2_base + offset1);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(c2_base + offset2);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(c2_base + offset3);

                        SIMD_TYPE c30 = SIMD_LOAD_R0(c3_base + offset0);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(c3_base + offset1);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(c3_base + offset2);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(c3_base + offset3);

                        SIMD_TYPE c40 = SIMD_LOAD_R0(c4_base + offset0);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(c4_base + offset1);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(c4_base + offset2);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(c4_base + offset3);

                        SIMD_TYPE c50 = SIMD_LOAD_R0(c5_base + offset0);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(c5_base + offset1);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(c5_base + offset2);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(c5_base + offset3);

                        SIMD_TYPE c60 = zero;
                        SIMD_TYPE c61 = zero;
                        SIMD_TYPE c62 = zero;
                        SIMD_TYPE c63 = zero;

                        SIMD_TYPE c70 = zero;
                        SIMD_TYPE c71 = zero;
                        SIMD_TYPE c72 = zero;
                        SIMD_TYPE c73 = zero;

                        SIMD_TYPE c80 = zero;
                        SIMD_TYPE c81 = zero;
                        SIMD_TYPE c82 = zero;
                        SIMD_TYPE c83 = zero;

                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_TYPE q00 = SIMD_XOR(y00, c80);
                        SIMD_TYPE q10 = RING_XOR_4(y10, a_y20, a2_y40, a3_c80);
                        SIMD_TYPE q20 = RING_XOR_4(y10, a2_y20, a4_y40, a_c80);

                        SIMD_TYPE q01 = SIMD_XOR(y01, c81);
                        SIMD_TYPE q11 = RING_XOR_4(y11, a_y21, a2_y41, a3_c81);
                        SIMD_TYPE q21 = RING_XOR_4(y11, a2_y21, a4_y41, a_c81);

                        SIMD_TYPE q02 = SIMD_XOR(y02, c82);
                        SIMD_TYPE q12 = RING_XOR_4(y12, a_y22, a2_y42, a3_c82);
                        SIMD_TYPE q22 = RING_XOR_4(y12, a2_y22, a4_y42, a_c82);

                        SIMD_TYPE q03 = SIMD_XOR(y03, c83);
                        SIMD_TYPE q13 = RING_XOR_4(y13, a_y23, a2_y43, a3_c83);
                        SIMD_TYPE q23 = RING_XOR_4(y13, a2_y23, a4_y43, a_c83);

                        SIMD_TYPE s00 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset0), q00);
                        SIMD_TYPE s01 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset1), q01);
                        SIMD_TYPE s02 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset2), q02);
                        SIMD_TYPE s03 = SIMD_XOR(SIMD_LOAD_R0(p0_base + offset3), q03);

                        SIMD_TYPE s10 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset0), q10);
                        SIMD_TYPE s11 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset1), q11);
                        SIMD_TYPE s12 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset2), q12);
                        SIMD_TYPE s13 = SIMD_XOR(SIMD_LOAD_R0(p1_base + offset3), q13);

                        SIMD_TYPE s20 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset0), q20);
                        SIMD_TYPE s21 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset1), q21);
                        SIMD_TYPE s22 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset2), q22);
                        SIMD_TYPE s23 = SIMD_XOR(SIMD_LOAD_R0(p2_base + offset3), q23);

                        /* Missing D5,D6,D7; A^-1 = [4 a b; 3 3 2; 6 9 9]. */
                        SIMD_TYPE r00 = RING_XOR_9(s02, s03, s11, s12, s13, s20, s21, s22, s23);
                        SIMD_TYPE r01 = RING_XOR_4(s02, s10, s11, s20);
                        SIMD_TYPE r02 = RING_XOR_5(s00, s02, s13, s22, s23);
                        SIMD_TYPE r03 = RING_XOR_9(s01, s02, s10, s11, s12, s13, s20, s21, s22);

                        SIMD_TYPE r10 = RING_XOR_5(s00, s03, s10, s13, s23);
                        SIMD_TYPE r11 = RING_XOR_8(s00, s01, s03, s10, s11, s13, s20, s23);
                        SIMD_TYPE r12 = RING_XOR_8(s01, s02, s03, s11, s12, s13, s21, s23);
                        SIMD_TYPE r13 = RING_XOR_4(s02, s12, s22, s23);

                        SIMD_TYPE r20 = RING_XOR_7(s02, s10, s11, s12, s20, s21, s22);
                        SIMD_TYPE r21 = RING_XOR_5(s00, s02, s03, s13, s23);
                        SIMD_TYPE r22 = RING_XOR_8(s00, s01, s02, s03, s11, s12, s21, s22);
                        SIMD_TYPE r23 = RING_XOR_8(s01, s03, s10, s11, s13, s20, s21, s23);

                        SIMD_STORE_R0(dst0_base + offset0, r00);
                        SIMD_STORE_R0(dst0_base + offset1, r01);
                        SIMD_STORE_R0(dst0_base + offset2, r02);
                        SIMD_STORE_R0(dst0_base + offset3, r03);
                        SIMD_STORE_R0(dst1_base + offset0, r10);
                        SIMD_STORE_R0(dst1_base + offset1, r11);
                        SIMD_STORE_R0(dst1_base + offset2, r12);
                        SIMD_STORE_R0(dst1_base + offset3, r13);
                        SIMD_STORE_R0(dst2_base + offset0, r20);
                        SIMD_STORE_R0(dst2_base + offset1, r21);
                        SIMD_STORE_R0(dst2_base + offset2, r22);
                        SIMD_STORE_R0(dst2_base + offset3, r23);
                }
                return;
        }
        }
}

void
ring_8_3_decode_data(int len, int m, int k, int node_size, unsigned char **buffs,
                     unsigned char *src_err_list, int nerrs, unsigned char **temp_buffs)
{
        int a[3] = { 0, 1, 2 };
        int i, j;
        int slot_size;

        (void) len;
        (void) m;
        (void) k;

        for (i = 0; i < nerrs; i++) {
                for (j = i + 1; j < nerrs; j++) {
                        if (src_err_list[i] > src_err_list[j]) {
                                unsigned char error = src_err_list[i];
                                int output = a[i];

                                src_err_list[i] = src_err_list[j];
                                src_err_list[j] = error;
                                a[i] = a[j];
                                a[j] = output;
                        }
                }
        }

        slot_size = node_size / 4;
        if (nerrs == 1) {
                recover_data_single_fused(temp_buffs[0], buffs, src_err_list[0], slot_size);
        } else if (nerrs == 2) {
                recover_data_pair_fused(temp_buffs[a[0]], temp_buffs[a[1]], buffs, src_err_list[0],
                                        src_err_list[1], slot_size);
        } else if (nerrs == 3) {
                recover_data_triple_fused(temp_buffs[a[0]], temp_buffs[a[1]], temp_buffs[a[2]],
                                          buffs, src_err_list[0], src_err_list[1], src_err_list[2],
                                          slot_size);
        }
}

void
ring_8_3_encode_data(int len, int k, int rows, int node_size, unsigned char **data,
                     unsigned char **coding)
{
        const unsigned char *d0 = data[0];
        const unsigned char *d1 = data[1];
        const unsigned char *d2 = data[2];
        const unsigned char *d3 = data[3];
        const unsigned char *d4 = data[4];
        const unsigned char *d5 = data[5];
        const unsigned char *d6 = data[6];
        const unsigned char *d7 = data[7];
        unsigned char *p0_dst = coding[0];
        unsigned char *p1_dst = coding[1];
        unsigned char *p2_dst = coding[2];
        size_t offset = 0;

        while (offset + RING_8_3_SIMD_BATCH_BYTES <= (size_t) node_size) {
                size_t offset0 = offset;
                size_t offset1 = offset + SIMD_ITERATION_SIZE;
                size_t offset2 = offset + 2 * SIMD_ITERATION_SIZE;
                size_t offset3 = offset + 3 * SIMD_ITERATION_SIZE;

                {
                        SIMD_TYPE c10 = SIMD_LOAD_R0(d0 + offset0);
                        SIMD_TYPE c20 = SIMD_LOAD_R0(d1 + offset0);
                        SIMD_TYPE c30 = SIMD_LOAD_R0(d2 + offset0);
                        SIMD_TYPE c40 = SIMD_LOAD_R0(d3 + offset0);
                        SIMD_TYPE c50 = SIMD_LOAD_R0(d4 + offset0);
                        SIMD_TYPE c60 = SIMD_LOAD_R0(d5 + offset0);
                        SIMD_TYPE c70 = SIMD_LOAD_R0(d6 + offset0);
                        SIMD_TYPE c80 = SIMD_LOAD_R0(d7 + offset0);
                        SIMD_TYPE x010 = c40;
                        SIMD_TYPE x110 = SIMD_XOR(c10, c50);
                        SIMD_TYPE x210 = SIMD_XOR(c20, c60);
                        SIMD_TYPE x310 = SIMD_XOR(c30, c70);
                        SIMD_TYPE x410 = SIMD_XOR(c40, c60);
                        SIMD_TYPE x510 = SIMD_XOR(c50, c70);
                        SIMD_TYPE x020 = SIMD_XOR(x010, x210);
                        SIMD_TYPE y10 = SIMD_XOR(x110, x310);
                        SIMD_TYPE y20 = SIMD_XOR(x210, x310);
                        SIMD_TYPE y40 = SIMD_XOR(x410, x510);
                        SIMD_TYPE y00 = SIMD_XOR(x020, y10);

                        SIMD_TYPE c11 = SIMD_LOAD_R0(d0 + offset1);
                        SIMD_TYPE c21 = SIMD_LOAD_R0(d1 + offset1);
                        SIMD_TYPE c31 = SIMD_LOAD_R0(d2 + offset1);
                        SIMD_TYPE c41 = SIMD_LOAD_R0(d3 + offset1);
                        SIMD_TYPE c51 = SIMD_LOAD_R0(d4 + offset1);
                        SIMD_TYPE c61 = SIMD_LOAD_R0(d5 + offset1);
                        SIMD_TYPE c71 = SIMD_LOAD_R0(d6 + offset1);
                        SIMD_TYPE c81 = SIMD_LOAD_R0(d7 + offset1);
                        SIMD_TYPE x011 = c41;
                        SIMD_TYPE x111 = SIMD_XOR(c11, c51);
                        SIMD_TYPE x211 = SIMD_XOR(c21, c61);
                        SIMD_TYPE x311 = SIMD_XOR(c31, c71);
                        SIMD_TYPE x411 = SIMD_XOR(c41, c61);
                        SIMD_TYPE x511 = SIMD_XOR(c51, c71);
                        SIMD_TYPE x021 = SIMD_XOR(x011, x211);
                        SIMD_TYPE y11 = SIMD_XOR(x111, x311);
                        SIMD_TYPE y21 = SIMD_XOR(x211, x311);
                        SIMD_TYPE y41 = SIMD_XOR(x411, x511);
                        SIMD_TYPE y01 = SIMD_XOR(x021, y11);

                        SIMD_TYPE c12 = SIMD_LOAD_R0(d0 + offset2);
                        SIMD_TYPE c22 = SIMD_LOAD_R0(d1 + offset2);
                        SIMD_TYPE c32 = SIMD_LOAD_R0(d2 + offset2);
                        SIMD_TYPE c42 = SIMD_LOAD_R0(d3 + offset2);
                        SIMD_TYPE c52 = SIMD_LOAD_R0(d4 + offset2);
                        SIMD_TYPE c62 = SIMD_LOAD_R0(d5 + offset2);
                        SIMD_TYPE c72 = SIMD_LOAD_R0(d6 + offset2);
                        SIMD_TYPE c82 = SIMD_LOAD_R0(d7 + offset2);
                        SIMD_TYPE x012 = c42;
                        SIMD_TYPE x112 = SIMD_XOR(c12, c52);
                        SIMD_TYPE x212 = SIMD_XOR(c22, c62);
                        SIMD_TYPE x312 = SIMD_XOR(c32, c72);
                        SIMD_TYPE x412 = SIMD_XOR(c42, c62);
                        SIMD_TYPE x512 = SIMD_XOR(c52, c72);
                        SIMD_TYPE x022 = SIMD_XOR(x012, x212);
                        SIMD_TYPE y12 = SIMD_XOR(x112, x312);
                        SIMD_TYPE y22 = SIMD_XOR(x212, x312);
                        SIMD_TYPE y42 = SIMD_XOR(x412, x512);
                        SIMD_TYPE y02 = SIMD_XOR(x022, y12);

                        SIMD_TYPE c13 = SIMD_LOAD_R0(d0 + offset3);
                        SIMD_TYPE c23 = SIMD_LOAD_R0(d1 + offset3);
                        SIMD_TYPE c33 = SIMD_LOAD_R0(d2 + offset3);
                        SIMD_TYPE c43 = SIMD_LOAD_R0(d3 + offset3);
                        SIMD_TYPE c53 = SIMD_LOAD_R0(d4 + offset3);
                        SIMD_TYPE c63 = SIMD_LOAD_R0(d5 + offset3);
                        SIMD_TYPE c73 = SIMD_LOAD_R0(d6 + offset3);
                        SIMD_TYPE c83 = SIMD_LOAD_R0(d7 + offset3);
                        SIMD_TYPE x013 = c43;
                        SIMD_TYPE x113 = SIMD_XOR(c13, c53);
                        SIMD_TYPE x213 = SIMD_XOR(c23, c63);
                        SIMD_TYPE x313 = SIMD_XOR(c33, c73);
                        SIMD_TYPE x413 = SIMD_XOR(c43, c63);
                        SIMD_TYPE x513 = SIMD_XOR(c53, c73);
                        SIMD_TYPE x023 = SIMD_XOR(x013, x213);
                        SIMD_TYPE y13 = SIMD_XOR(x113, x313);
                        SIMD_TYPE y23 = SIMD_XOR(x213, x313);
                        SIMD_TYPE y43 = SIMD_XOR(x413, x513);
                        SIMD_TYPE y03 = SIMD_XOR(x023, y13);

                        /* a*y2. */
                        SIMD_TYPE a_y20 = y23;
                        SIMD_TYPE a_y21 = SIMD_XOR(y20, y23);
                        SIMD_TYPE a_y22 = SIMD_XOR(y21, y23);
                        SIMD_TYPE a_y23 = SIMD_XOR(y22, y23);

                        /* a^2*y4. */
                        SIMD_TYPE a2_y40 = SIMD_XOR(y42, y43);
                        SIMD_TYPE a2_y41 = y42;
                        SIMD_TYPE a2_y42 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a2_y43 = SIMD_XOR(y41, y42);

                        /* a^3*c8. */
                        SIMD_TYPE a3_c80 = SIMD_XOR(c81, c82);
                        SIMD_TYPE a3_c81 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a3_c82 = c81;
                        SIMD_TYPE a3_c83 = SIMD_XOR(c80, c81);

                        /* a^2*y2. */
                        SIMD_TYPE a2_y20 = SIMD_XOR(y22, y23);
                        SIMD_TYPE a2_y21 = y22;
                        SIMD_TYPE a2_y22 = SIMD_XOR(y20, y22);
                        SIMD_TYPE a2_y23 = SIMD_XOR(y21, y22);

                        /* a^4*y4. */
                        SIMD_TYPE a4_y40 = SIMD_XOR(y40, y41);
                        SIMD_TYPE a4_y41 = SIMD_XOR(y40, y42);
                        SIMD_TYPE a4_y42 = SIMD_XOR(y40, y43);
                        SIMD_TYPE a4_y43 = y40;

                        /* a*c8. */
                        SIMD_TYPE a_c80 = c83;
                        SIMD_TYPE a_c81 = SIMD_XOR(c80, c83);
                        SIMD_TYPE a_c82 = SIMD_XOR(c81, c83);
                        SIMD_TYPE a_c83 = SIMD_XOR(c82, c83);

                        SIMD_STORE_R0(p0_dst + offset0, SIMD_XOR(y00, c80));
                        SIMD_STORE_R0(p0_dst + offset1, SIMD_XOR(y01, c81));
                        SIMD_STORE_R0(p0_dst + offset2, SIMD_XOR(y02, c82));

                        SIMD_STORE_R0(p0_dst + offset3, SIMD_XOR(y03, c83));

                        SIMD_STORE_R0(p1_dst + offset0, RING_XOR_4(y10, a_y20, a2_y40, a3_c80));
                        SIMD_STORE_R0(p1_dst + offset1, RING_XOR_4(y11, a_y21, a2_y41, a3_c81));
                        SIMD_STORE_R0(p1_dst + offset2, RING_XOR_4(y12, a_y22, a2_y42, a3_c82));
                        SIMD_STORE_R0(p1_dst + offset3, RING_XOR_4(y13, a_y23, a2_y43, a3_c83));

                        SIMD_STORE_R0(p2_dst + offset0, RING_XOR_4(y10, a2_y20, a4_y40, a_c80));
                        SIMD_STORE_R0(p2_dst + offset1, RING_XOR_4(y11, a2_y21, a4_y41, a_c81));
                        SIMD_STORE_R0(p2_dst + offset2, RING_XOR_4(y12, a2_y22, a4_y42, a_c82));
                        SIMD_STORE_R0(p2_dst + offset3, RING_XOR_4(y13, a2_y23, a4_y43, a_c83));
                }

                offset += RING_8_3_SIMD_BATCH_BYTES;
        }

        /* One batch holds four vectors of the target SIMD load width. */
}

#undef RING_XOR_12
#undef RING_XOR_11
#undef RING_XOR_10
#undef RING_XOR_9
#undef RING_XOR_8
#undef RING_XOR_7
#undef RING_XOR_6
#undef RING_XOR_5
#undef RING_XOR_4
#undef RING_XOR_3
#undef RING_XOR_2
#undef RING_XOR_1
