/*************************************************************
  Copyright (c) 2021 Linaro Ltd.

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
.text
.align		6
.arch		armv8-a+sve2

#include "../include/aarch64_label.h"

.global cdecl(gf_2vect_dot_prod_sve)
#ifndef __APPLE__
.type gf_2vect_dot_prod_sve, %function
#endif
/* void gf_2vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls,
				   unsigned char **src, unsigned char **dest);
 */

/* arguments */
x_len		.req	x0	/* vector length */
x_vec		.req	x1	/* number of source vectors (ie. data blocks) */
x_tbl		.req	x2
x_src		.req	x3
x_dest		.req	x4

/* returns */
w_ret		.req	w0

/* local variables */
x_vec_i		.req	x5
x_ptr		.req	x6
x_pos		.req	x7

x_tbl1		.req	x8
x_tbl2		.req	x9
x_dest1		.req	x10
x_dest2		.req	x_dest	/* reused */

x_pos2      .req    x11
x_temp      .req    x12

/* r16,r17,r18,r29,r30: special role registers, avoided */
/* r19..r29 and SP must be preserved */

/* vectors */
z_mask0f	.req	z0

z_src		.req	z1
z_src_lo	.req	z2
z_src_hi	.req	z_src

z_dest1		.req	z3
z_dest1_2   .req    z8

z_src2      .req    z9
z_src2_lo   .req    z10
z_src2_hi   .req    z_src2

z_gft1_lo	.req	z4
z_gft1_hi	.req	z5
q_gft1_lo	.req	q4
q_gft1_hi	.req	q5

z_gft1_lo_r .req    z6
z_gft1_hi_r .req    z7

z_gft1_lo_r_2  .req  z11
z_gft1_hi_r_2  .req  z12

/* bottom 64-bit of v8..v15 must be preserved if used */
z_gft2_lo	.req	z17
z_gft2_hi	.req	z18
q_gft2_lo	.req	q17
q_gft2_hi	.req	q18

z_gft2_lo_r .req    z19
z_gft2_hi_r .req    z20

z_gft2_lo_r_2  .req  z19
z_gft2_hi_r_2  .req  z20

z_dest2		.req	z27
z_dest2_2   .req    z28

cdecl(gf_2vect_dot_prod_sve):
	/* less than 16 bytes, return_fail */
	cmp	x_len, #16
	blt	.return_fail

	mov	z_mask0f.b, #0x0f		/* z_mask0f = 0x0F0F...0F */
	mov	x_pos, #0
    mov x_pos2, #32

	lsl	x_vec, x_vec, #3
	ldp	x_dest1, x_dest2, [x_dest, #8*0]

/* Loop 1: x_len, vector length */
.Lloopsve_vl:
	whilelo	p0.b, x_pos, x_len
	b.none	.return_pass
	whilelo	p1.b, x_pos2, x_len

	mov	x_vec_i, #0			/* clear x_vec_i */
	ldr	x_ptr, [x_src, x_vec_i]		/* x_ptr: src base addr. */

	mov	z_dest1.b, #0			/* clear z_dest1 */
	mov z_dest1_2.b, #0

	mov	z_dest2.b, #0			/* clear z_dest2 */
	mov z_dest2_2.b, #0

	/* gf_tbl base = (x_tbl + dest_idx * x_vec * 32) */
	mov	x_tbl1, x_tbl			/* reset x_tbl1 */
	add	x_tbl2, x_tbl1, x_vec, LSL #2	/* reset x_tbl2 */

/* Loop 2: x_vec, number of source vectors (ie. data blocks) */
.Lloopsve_vl_vects:
	/* load src data, governed by p0 */
	ld1b	z_src.b,  p0/z, [x_ptr, x_pos]	/* load from: src base + pos offset */
	ld1b    z_src2.b, p1/z, [x_ptr, x_pos2]

    add x_temp, x_ptr, x_pos
	prfm   pldl1strm, [x_temp, #128]

	add x_vec_i, x_vec_i, #8
	ldr x_ptr, [x_src, x_vec_i]

	add x_temp, x_ptr, x_pos
	prfm   pldl1strm, [x_temp]

	/* split 4-bit lo; 4-bit hi */
	and	z_src_lo.d, z_src.d, z_mask0f.d
	lsr	z_src_hi.b, z_src.b, #4

	and z_src2_lo.d, z_src2.d, z_mask0f.d
	lsr z_src2_hi.b, z_src2.b, #4


	/* gf_tbl addr: (x_tbl + dest_idx * x_vec * 32) + src_vec_idx * 32 */
	/* load gf_table's */
	ldp	q_gft1_lo, q_gft1_hi, [x_tbl1], #32	/* x_tbl1 is post-added by #32 for each src vect */
	ldp	q_gft2_lo, q_gft2_hi, [x_tbl2], #32

	/* prefetch */
	prfm	pldl1keep, [x_tbl1, #64]
	prfm	pldl1keep, [x_tbl2, #64]


	/* dest 1 */
	/* table indexing, ie. gf(2^8) multiplication */
	tbl	z_gft1_lo_r.b, {z_gft1_lo.b}, z_src_lo.b
	tbl	z_gft1_hi_r.b, {z_gft1_hi.b}, z_src_hi.b
	eor3 z_dest1.d, z_dest1.d, z_gft1_hi_r.d, z_gft1_lo_r.d

	tbl	z_gft1_lo_r.b, {z_gft1_lo.b}, z_src2_lo.b
	tbl	z_gft1_hi_r.b, {z_gft1_hi.b}, z_src2_hi.b
	eor3 z_dest1_2.d, z_dest1_2.d, z_gft1_hi_r.d, z_gft1_lo_r.d

	/* dest 2 */
	tbl	z_gft2_lo_r.b, {z_gft2_lo.b}, z_src_lo.b
	tbl	z_gft2_hi_r.b, {z_gft2_hi.b}, z_src_hi.b
	eor3 z_dest2.d, z_dest2.d, z_gft2_hi_r.d, z_gft2_lo_r.d

    tbl	z_gft2_lo_r.b, {z_gft2_lo.b}, z_src2_lo.b
	tbl	z_gft2_hi_r.b, {z_gft2_hi.b}, z_src2_hi.b
	eor3 z_dest2_2.d, z_dest2_2.d, z_gft2_hi_r.d, z_gft2_lo_r.d

	cmp	x_vec_i, x_vec
	blt	.Lloopsve_vl_vects
/* end of Loop 2 */

	/* store dest data, governed by p0 */
	st1b	z_dest1.b, p0, [x_dest1, x_pos]
    st1b	z_dest1_2.b, p1, [x_dest1, x_pos2]

	st1b	z_dest2.b, p0, [x_dest2, x_pos]
	st1b	z_dest2_2.b, p1, [x_dest2, x_pos2]

	/* increment two vector length */
	incb	x_pos
	incb    x_pos

	incb    x_pos2
	incb    x_pos2
	b	.Lloopsve_vl
/* end of Loop 1 */

.return_pass:
	mov	w_ret, #0
	ret

.return_fail:
	mov	w_ret, #1
	ret
