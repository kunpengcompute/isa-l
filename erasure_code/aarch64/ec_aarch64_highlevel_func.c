/**************************************************************
  Copyright (c) 2019 Huawei Technologies Co., Ltd.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Huawei Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************/
#include <stdlib.h>
#include "erasure_code.h"
#include "ring_internal.h"
#include <assert.h>
#ifndef unlikely
#define unlikely(x)  __builtin_expect(!!(x), 0)
#endif

/*external function*/
extern void
gf_vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char *dest);
extern void
gf_2vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_3vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_4vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_5vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char *dest);
extern void
gf_2vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_3vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_4vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_5vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_6vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);

void
ec_encode_data_neon(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                    unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        while (rows > 5) {
                gf_5vect_dot_prod_neon(len, k, g_tbls, data, coding);
                g_tbls += 5 * k * 32;
                coding += 5;
                rows -= 5;
        }
        switch (rows) {
        case 5:
                gf_5vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_neon(len, k, g_tbls, data, *coding);
                break;
        case 0:
                break;
        default:
                break;
        }
}

void
ec_encode_data_update_neon(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                           unsigned char *data, unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_update_base(len, k, rows, vec_i, g_tbls, data, coding);
                return;
        }
        while (rows > 6) {
                gf_6vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 6:
                gf_6vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_neon(len, k, vec_i, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

/* SVE */
extern void
gf_vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                     unsigned char *dest);
extern void
gf_vect_dot_prod_eor_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                         unsigned char *dest);
extern void
gf_vect_dot_prod_sve2(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char *dest);
extern void
gf_vect_dot_prod_eor_sve2(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                          unsigned char *dest);
extern void
gf_2vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_2vect_dot_prod_eor_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                          unsigned char **dest);
extern void
gf_2vect_dot_prod_sve2(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_2vect_dot_prod_eor_sve2(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                           unsigned char **dest);
extern void
gf_3vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_3vect_dot_prod_eor_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                          unsigned char **dest);
extern void
gf_3vect_dot_prod_sve2(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_3vect_dot_prod_eor_sve2(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                           unsigned char **dest);
extern void
gf_4vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_5vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_6vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_7vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_8vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                unsigned char *dest);
extern void
gf_2vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);
extern void
gf_3vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);
extern void
gf_4vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);
extern void
gf_5vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);
extern void
gf_6vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);

int
is_all_one(unsigned char *g_tbls, int k)
{
        for (int i = 0; i < k; i++) {
                if (g_tbls[i * 32 + 1] != 0x1) {
                        return 0;
                }
        }
        return 1;
}

void
ec_encode_data_sve2(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                    unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        if (rows == 1 && is_all_one(g_tbls, k)) {
                gf_vect_dot_prod_eor_sve2(len, k, g_tbls, data, *coding);
                return;
        } else if (rows == 2 && is_all_one(g_tbls, k)) {
                gf_2vect_dot_prod_eor_sve2(len, k, g_tbls, data, coding);
                return;
        } else if (rows == 3 && is_all_one(g_tbls, k)) {
                gf_3vect_dot_prod_eor_sve2(len, k, g_tbls, data, coding);
                return;
        }

        while (rows > 11) {
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }

        switch (rows) {
        case 11:
                /* 7 + 4 */
                gf_7vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 7 * k * 32;
                coding += 7;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 10:
                /* 6 + 4 */
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 9:
                /* 5 + 4 */
                gf_5vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 5 * k * 32;
                coding += 5;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 8:
                /* 4 + 4 */
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 4 * k * 32;
                coding += 4;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 7:
                gf_7vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 6:
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_sve2(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_sve2(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_sve2(len, k, g_tbls, data, *coding);
                break;
        default:
                break;
        }
}

void
ec_encode_data_sve(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                   unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        if (rows == 1 && is_all_one(g_tbls, k)) {
                gf_vect_dot_prod_eor_sve(len, k, g_tbls, data, *coding);
                return;
        } else if (rows == 2 && is_all_one(g_tbls, k)) {
                gf_2vect_dot_prod_eor_sve(len, k, g_tbls, data, coding);
                return;
        } else if (rows == 3 && is_all_one(g_tbls, k)) {
                gf_3vect_dot_prod_eor_sve(len, k, g_tbls, data, coding);
                return;
        }

        while (rows > 11) {
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }

        switch (rows) {
        case 11:
                /* 7 + 4 */
                gf_7vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 7 * k * 32;
                coding += 7;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 10:
                /* 6 + 4 */
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 9:
                /* 5 + 4 */
                gf_5vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 5 * k * 32;
                coding += 5;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 8:
                /* 4 + 4 */
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 4 * k * 32;
                coding += 4;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 7:
                gf_7vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 6:
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_sve(len, k, g_tbls, data, *coding);
                break;
        default:
                break;
        }
}

void
ec_encode_data_update_sve(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                          unsigned char *data, unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_update_base(len, k, rows, vec_i, g_tbls, data, coding);
                return;
        }
        while (rows > 6) {
                gf_6vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 6:
                gf_6vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_sve(len, k, vec_i, g_tbls, data, *coding);
                break;
        default:
                break;
        }
}

/* k/config and nerrs are validated by the caller before entering here. */
static inline int
ring_copy_errors_validated_k(int k, const unsigned char *errors, int nerrs,
                             unsigned char *local_errors, unsigned int *missing_mask)
{
        unsigned int mask = 0;
        for (int i = 0; i < nerrs; ++i) {
                unsigned int id = errors[i];
                if (id >= (unsigned int) k)
                        return -1;
                unsigned int bit = 1U << id;
                if (mask & bit)
                        return -1;
                mask |= bit;
                local_errors[i] = (unsigned char) id;
        }
        *missing_mask = mask;
        return 0;
}

/*
 * recov contains the first k survivors in increasing shard-id order.
 * All missing ids refer to data shards; the first nerrs parity shards survive.
 * Initialize each data position, then every parity position. A full-array
 * zeroing pass and a second missing[] construction are unnecessary.
 */
static inline void
ring_build_shards(int k, int parity, unsigned char **recov,
                            const unsigned char *errors, int nerrs,
                            unsigned int missing_mask,
                            unsigned char **outputs, unsigned char **shards)
{
        int input = 0;
        for (int id = 0; id < k; ++id) {
                if (missing_mask & (1U << id))
                        shards[id] = NULL;
                else
                        shards[id] = recov[input++];
        }
        for (int i = 0; i < parity; ++i)
                shards[k + i] = i < nerrs ? recov[input++] : NULL;
        for (int i = 0; i < nerrs; ++i)
                shards[errors[i]] = outputs[i];
}

int
ring_algorithm_encode(int len, int k, int parity, unsigned char **data,
                      unsigned char **coding)
{
        switch (k) {
        case 4:
                ring_4_2_encode_data(k, len, data, coding);
                return 0;
        case 8:
                ring_8_3_encode_data(len, k, parity, len, data, coding);
                return 0;
        case 10:
                ring_10_1_encode_data(len, k, parity, len, data, coding);
                return 0;
        default:
                return -1;
        }
}

int
ring_algorithm_decode(int len, int k, int parity, unsigned char **shards,
                      unsigned char *errors, int nerrs, unsigned char **coding)
{
        switch (k) {
        case 4:
                ring_4_2_decode_data(len, shards, errors, nerrs, coding);
                return 0;
        case 8:
                ring_8_3_decode_data(len, k + parity, k, len, shards, errors, nerrs, coding);
                return 0;
        case 10:
                ring_10_1_decode_data(len, k + parity, k, len, shards, errors, nerrs, coding);
                return 0;
        default:
                return -1;
        }
}

static int
ring_try_execute(int len, int k, int rows, unsigned char *g_tbls,
                 unsigned char **data, unsigned char **coding)
{
        if (k <= 0 || rows <= 0)
                return RING_NOT_HANDLED;
        if (g_tbls == NULL)
                return RING_INVALID;
        if (g_tbls[0] == 0)
                return RING_NOT_HANDLED;
        if (g_tbls[0] != 0x52 || g_tbls[1] != 0x54 ||
            g_tbls[2] != 1 || g_tbls[7] != 0)
                return RING_INVALID;

        const int op = g_tbls[3];
        const int parity = g_tbls[5];
        const int nerrs = g_tbls[6];
        if (g_tbls[4] != k || ring_config_id(k, parity) < 0 || len < 0)
                return RING_INVALID;
        if (k == 8 && len % 4 != 0)
                return RING_INVALID;

        if (op == RING_OP_ENCODE) {
                if (rows != parity || nerrs != 0)
                        return RING_INVALID;
                if (len == 0)
                        return RING_HANDLED;
                if (data == NULL || coding == NULL)
                        return RING_INVALID;
                for (int i = 0; i < k; ++i)
                        if (data[i] == NULL)
                                return RING_INVALID;
                for (int i = 0; i < parity; ++i)
                        if (coding[i] == NULL)
                                return RING_INVALID;
                return ring_algorithm_encode(len, k, parity, data, coding) == 0
                               ? RING_HANDLED : RING_INVALID;
        }

        if (op != RING_OP_DECODE)
                return RING_INVALID;
        if (nerrs < 1 || nerrs > parity || rows != nerrs)
                return RING_INVALID;

        unsigned char errors[RING_MAX_PARITY];
        unsigned int missing_mask;
        if (ring_copy_errors_validated_k(k, g_tbls + 8, nerrs, errors, &missing_mask) != 0)
                return RING_INVALID;
        if (len == 0)
                return RING_HANDLED;
        if (data == NULL || coding == NULL)
                return RING_INVALID;
        for (int i = 0; i < k; ++i)
                if (data[i] == NULL)
                        return RING_INVALID;
        for (int i = 0; i < nerrs; ++i)
                if (coding[i] == NULL)
                        return RING_INVALID;

        unsigned char *shards[RING_MAX_SHARDS];
        ring_build_shards(k, parity, data, errors, nerrs, missing_mask,
                                    coding, shards);
        return ring_algorithm_decode(len, k, parity, shards, errors, nerrs,
                                               coding) == 0
                       ? RING_HANDLED : RING_INVALID;
}

void
ec_encode_data_ring(int len, int k, int rows, unsigned char *g_tbls,
                    unsigned char **data, unsigned char **coding)
{
        const int status = ring_try_execute(len, k, rows, g_tbls, data, coding);
        if (status == RING_HANDLED)
                return;
        if (status == RING_NOT_HANDLED) {
                ec_encode_data_native(len, k, rows, g_tbls, data, coding);
                return;
        }
        abort();
}
