/**********************************************************************
  Copyright(c) 2011-2015 Intel Corporation All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
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

#include <limits.h>
#include <string.h> // for memset
#include <stdint.h>
#include <stdlib.h>

#include "erasure_code.h"
#include "ec_base.h" // for GF tables
#include "ring_internal.h"

static int
ring_try_init_tables(int k, int rows, const unsigned char *a, unsigned char *g_tbls)
{
        int config, recovery, index;
        unsigned char errors[RING_MAX_PARITY] = { 0 };
        unsigned char seen[RING_MAX_K] = { 0 };

        if (rows <= 0)
                return RING_NOT_HANDLED;

        if (k != 4 && k != 8 && k != 10)
                return RING_NOT_HANDLED;

        if (!ring_read_row(a, k, &config, &recovery, &index))
                return RING_NOT_HANDLED;

        int parity = ring_config_parity(config, k);

        if (recovery == 0) {
                /* 当前方案要求一次生成全部校验块 */
                if (rows != parity)
                        return RING_INVALID;
        } else {
                if (rows > parity)
                        return RING_INVALID;
        }

        for (int i = 0; i < rows; i++) {
                int c, r, id;

                if (!ring_read_row(a + (size_t) i * k, k, &c, &r, &id))
                        return RING_INVALID;

                if (c != config || r != recovery)
                        return RING_INVALID;

                if (recovery == 0) {
                        /* 编码输入应为第 k ... k+parity-1 行 */
                        if (id != k + i)
                                return RING_INVALID;
                } else {
                        /* ring_read_row 已验证恢复编号小于 k */
                        if (seen[id])
                                return RING_INVALID;

                        seen[id] = 1;
                        errors[i] = (unsigned char) id;
                }
        }

        memset(g_tbls, 0, RING_META_SIZE);

        g_tbls[0] = 0x52;
        g_tbls[1] = 0x54;
        g_tbls[2] = 1;
        g_tbls[3] = recovery ? RING_OP_DECODE : RING_OP_ENCODE;
        g_tbls[4] = (unsigned char) k;
        g_tbls[5] = (unsigned char) parity;
        g_tbls[6] = recovery ? (unsigned char) rows : 0;

        if (recovery)
                memcpy(g_tbls + 8, errors, (size_t) rows);

        return RING_HANDLED;
}

void
ec_init_tables_base(int k, int rows, unsigned char *a, unsigned char *g_tbls)
{
#ifdef ENABLE_RING_PROTOCOL
        int status = ring_try_init_tables(k, rows, a, g_tbls);

        if (status == RING_HANDLED)
                return;

        if (status == RING_INVALID)
                abort();
#endif
        int i, j;

        for (i = 0; i < rows; i++) {
                for (j = 0; j < k; j++) {
                        gf_vect_mul_init(*a++, g_tbls);
                        g_tbls += 32;
                }
        }
}

unsigned char
gf_mul(unsigned char a, unsigned char b)
{
#ifndef GF_LARGE_TABLES
        int i;

        if ((a == 0) || (b == 0))
                return 0;

        return gff_base[(i = gflog_base[a] + gflog_base[b]) > 254 ? i - 255 : i];
#else
        return gf_mul_table_base[b * 256 + a];
#endif
}

unsigned char
gf_inv(unsigned char a)
{
#ifndef GF_LARGE_TABLES
        if (a == 0)
                return 0;

        return gff_base[255 - gflog_base[a]];
#else
        return gf_inv_table_base[a];
#endif
}

void
gf_gen_rs_matrix(unsigned char *a, int m, int k)
{
        int i, j;
        unsigned char p, gen = 1;

#ifdef ENABLE_RING_PROTOCOL
        int config = ring_config_id(k, m - k);

        if (config >= 0) {
                /*
                 * 只在 4+2、8+3、10+1 下生成 Ring 描述矩阵。
                 * 容量仍然是原来的 m * k 字节。
                 */
                memset(a, 0, (size_t) k * m);

                for (i = 0; i < m; i++)
                        a[(size_t) k * i] = ring_row_code(config, 0, i);

                return;
        }
#endif

        /* 其他配置：保留原来的 RS 矩阵生成逻辑 */
        memset(a, 0, (size_t) k * m);

        for (i = 0; i < k; i++)
                a[k * i + i] = 1;

        for (i = k; i < m; i++) {
                p = 1;
                for (j = 0; j < k; j++) {
                        a[k * i + j] = p;
                        p = gf_mul(p, gen);
                }
                gen = gf_mul(gen, 2);
        }
}

void
gf_gen_cauchy1_matrix(unsigned char *a, int m, int k)
{
        int i, j;
        unsigned char *p;

        // Identity matrix in high position
        memset(a, 0, k * m);
        for (i = 0; i < k; i++)
                a[k * i + i] = 1;

        // For the rest choose 1/(i + j) | i != j
        p = &a[k * k];
        for (i = k; i < m; i++)
                for (j = 0; j < k; j++)
                        *p++ = gf_inv(i ^ j);
}

static int
ring_try_invert(const unsigned char *in, unsigned char *out, int k)
{
        int config, recovery, index;
        unsigned char seen[RING_MAX_SHARDS] = { 0 };
        int previous = -1;
        int parity_used = 0;

        if (k != 4 && k != 8 && k != 10)
                return RING_NOT_HANDLED;

        if (!ring_read_row(in, k, &config, &recovery, &index) || recovery != 0)
                return RING_NOT_HANDLED;

        int parity = ring_config_parity(config, k);

        for (int i = 0; i < k; i++) {
                int c, r, id;

                if (!ring_read_row(in + (size_t) i * k, k, &c, &r, &id))
                        return RING_INVALID;

                if (c != config || r != 0)
                        return RING_INVALID;

                if (id <= previous || seen[id])
                        return RING_INVALID;

                seen[id] = 1;
                previous = id;

                if (id >= k) {
                        if (id != k + parity_used)
                                return RING_INVALID;

                        parity_used++;
                }
        }

        if (parity_used > parity)
                return RING_INVALID;

        memset(out, 0, (size_t) k * k);

        for (int i = 0; i < k; i++)
                out[(size_t) i * k] = ring_row_code(config, 1, i);

        return RING_HANDLED;
}

int
gf_invert_matrix(unsigned char *in_mat, unsigned char *out_mat, const int n)
{
#ifdef ENABLE_RING_PROTOCOL
        int status = ring_try_invert(in_mat, out_mat, n);

        if (status == RING_HANDLED)
                return 0;

        if (status == RING_INVALID)
                return -1;
#endif
        int i, j, k;
        unsigned char temp;

        // Set out_mat[] to the identity matrix
        for (i = 0; i < n * n; i++) // memset(out_mat, 0, n*n)
                out_mat[i] = 0;

        for (i = 0; i < n; i++)
                out_mat[i * n + i] = 1;

        // Inverse
        for (i = 0; i < n; i++) {
                // Check for 0 in pivot element
                if (in_mat[i * n + i] == 0) {
                        // Find a row with non-zero in current column and swap
                        for (j = i + 1; j < n; j++)
                                if (in_mat[j * n + i])
                                        break;

                        if (j == n) // Couldn't find means it's singular
                                return -1;

                        for (k = 0; k < n; k++) { // Swap rows i,j
                                temp = in_mat[i * n + k];
                                in_mat[i * n + k] = in_mat[j * n + k];
                                in_mat[j * n + k] = temp;

                                temp = out_mat[i * n + k];
                                out_mat[i * n + k] = out_mat[j * n + k];
                                out_mat[j * n + k] = temp;
                        }
                }

                temp = gf_inv(in_mat[i * n + i]); // 1/pivot
                for (j = 0; j < n; j++) {         // Scale row i by 1/pivot
                        in_mat[i * n + j] = gf_mul(in_mat[i * n + j], temp);
                        out_mat[i * n + j] = gf_mul(out_mat[i * n + j], temp);
                }

                for (j = 0; j < n; j++) {
                        if (j == i)
                                continue;

                        temp = in_mat[j * n + i];
                        for (k = 0; k < n; k++) {
                                out_mat[j * n + k] ^= gf_mul(temp, out_mat[i * n + k]);
                                in_mat[j * n + k] ^= gf_mul(temp, in_mat[i * n + k]);
                        }
                }
        }
        return 0;
}

// Calculates const table gftbl in GF(2^8) from single input A
// gftbl(A) = {A{00}, A{01}, A{02}, ... , A{0f} }, {A{00}, A{10}, A{20}, ... , A{f0} }

void
gf_vect_mul_init(unsigned char c, unsigned char *tbl)
{
        unsigned char c2 = (c << 1) ^ ((c & 0x80) ? 0x1d : 0);   // Mult by GF{2}
        unsigned char c4 = (c2 << 1) ^ ((c2 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        unsigned char c8 = (c4 << 1) ^ ((c4 & 0x80) ? 0x1d : 0); // Mult by GF{2}

#if (__WORDSIZE == 64 || _WIN64 || __x86_64__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        unsigned long long v1, v2, v4, v8, *t;
        unsigned long long v10, v20, v40, v80;
        unsigned char c17, c18, c20, c24;

        t = (unsigned long long *) tbl;

        v1 = c * 0x0100010001000100ull;
        v2 = c2 * 0x0101000001010000ull;
        v4 = c4 * 0x0101010100000000ull;
        v8 = c8 * 0x0101010101010101ull;

        v4 = v1 ^ v2 ^ v4;
        t[0] = v4;
        t[1] = v8 ^ v4;

        c17 = (c8 << 1) ^ ((c8 & 0x80) ? 0x1d : 0);   // Mult by GF{2}
        c18 = (c17 << 1) ^ ((c17 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c20 = (c18 << 1) ^ ((c18 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c24 = (c20 << 1) ^ ((c20 & 0x80) ? 0x1d : 0); // Mult by GF{2}

        v10 = c17 * 0x0100010001000100ull;
        v20 = c18 * 0x0101000001010000ull;
        v40 = c20 * 0x0101010100000000ull;
        v80 = c24 * 0x0101010101010101ull;

        v40 = v10 ^ v20 ^ v40;
        t[2] = v40;
        t[3] = v80 ^ v40;

#else // 32-bit or other
        unsigned char c3, c5, c6, c7, c9, c10, c11, c12, c13, c14, c15;
        unsigned char c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31;

        c3 = c2 ^ c;
        c5 = c4 ^ c;
        c6 = c4 ^ c2;
        c7 = c4 ^ c3;

        c9 = c8 ^ c;
        c10 = c8 ^ c2;
        c11 = c8 ^ c3;
        c12 = c8 ^ c4;
        c13 = c8 ^ c5;
        c14 = c8 ^ c6;
        c15 = c8 ^ c7;

        tbl[0] = 0;
        tbl[1] = c;
        tbl[2] = c2;
        tbl[3] = c3;
        tbl[4] = c4;
        tbl[5] = c5;
        tbl[6] = c6;
        tbl[7] = c7;
        tbl[8] = c8;
        tbl[9] = c9;
        tbl[10] = c10;
        tbl[11] = c11;
        tbl[12] = c12;
        tbl[13] = c13;
        tbl[14] = c14;
        tbl[15] = c15;

        c17 = (c8 << 1) ^ ((c8 & 0x80) ? 0x1d : 0);   // Mult by GF{2}
        c18 = (c17 << 1) ^ ((c17 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c19 = c18 ^ c17;
        c20 = (c18 << 1) ^ ((c18 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c21 = c20 ^ c17;
        c22 = c20 ^ c18;
        c23 = c20 ^ c19;
        c24 = (c20 << 1) ^ ((c20 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c25 = c24 ^ c17;
        c26 = c24 ^ c18;
        c27 = c24 ^ c19;
        c28 = c24 ^ c20;
        c29 = c24 ^ c21;
        c30 = c24 ^ c22;
        c31 = c24 ^ c23;

        tbl[16] = 0;
        tbl[17] = c17;
        tbl[18] = c18;
        tbl[19] = c19;
        tbl[20] = c20;
        tbl[21] = c21;
        tbl[22] = c22;
        tbl[23] = c23;
        tbl[24] = c24;
        tbl[25] = c25;
        tbl[26] = c26;
        tbl[27] = c27;
        tbl[28] = c28;
        tbl[29] = c29;
        tbl[30] = c30;
        tbl[31] = c31;

#endif //__WORDSIZE == 64 || _WIN64 || __x86_64__
}

void
gf_vect_dot_prod_base(int len, int vlen, unsigned char *v, unsigned char **src, unsigned char *dest)
{
        int i, j;
        unsigned char s;
        for (i = 0; i < len; i++) {
                s = 0;
                for (j = 0; j < vlen; j++)
                        s ^= gf_mul(src[j][i], v[j * 32 + 1]);

                dest[i] = s;
        }
}

void
gf_vect_mad_base(int len, int vec, int vec_i, unsigned char *v, unsigned char *src,
                 unsigned char *dest)
{
        int i;
        unsigned char s;
        for (i = 0; i < len; i++) {
                s = dest[i];
                s ^= gf_mul(src[i], v[vec_i * 32 + 1]);
                dest[i] = s;
        }
}

void
ec_encode_data_base(int len, int srcs, int dests, unsigned char *v, unsigned char **src,
                    unsigned char **dest)
{
        int i, j, l;
        unsigned char s;

        for (l = 0; l < dests; l++) {
                for (i = 0; i < len; i++) {
                        s = 0;
                        for (j = 0; j < srcs; j++)
                                s ^= gf_mul(src[j][i], v[j * 32 + l * srcs * 32 + 1]);

                        dest[l][i] = s;
                }
        }
}

void
ring_encode_data_base(int len, int srcs, int dests, int node_size, unsigned char **src,
                      unsigned char **dest)
{
        /// 待改写
        // int i, j, l;
        // unsigned char s;
        return;
}

void
ring_decode_data_base(int len, int m, int k, int node_size, unsigned char **buffs,
                      unsigned char *src_err_list, int nerrs, unsigned char **temp_buffs)
{
        /// 待改写
        // int i, j, l;
        // unsigned char s;
        return;
}

void
ec_encode_data_update_base(int len, int k, int rows, int vec_i, unsigned char *v,
                           unsigned char *data, unsigned char **dest)
{
        int i, l;
        unsigned char s;

        for (l = 0; l < rows; l++) {
                for (i = 0; i < len; i++) {
                        s = dest[l][i];
                        s ^= gf_mul(data[i], v[vec_i * 32 + l * k * 32 + 1]);

                        dest[l][i] = s;
                }
        }
}

int
gf_vect_mul_base(int len, unsigned char *a, unsigned char *src, unsigned char *dest)
{
        // 2nd element of table array is ref value used to fill it in
        unsigned char c = a[1];

        // Len must be aligned to 32B
        if ((len % 32) != 0) {
                return -1;
        }

        while (len-- > 0)
                *dest++ = gf_mul(c, *src++);
        return 0;
}
