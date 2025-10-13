/**************************************************************
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
.arch		armv8-a+sve

#include "../include/aarch64_label.h"

.global cdecl(gf_vect_dot_prod_eor_sve)
#ifndef __APPLE__
.type gf_vect_dot_prod_eor_sve, %function
#endif
/* void gf_vect_dot_prod_eor_sve(int len, int vlen, unsigned char *gftbls,
				unsigned char **src, unsigned char *dest);
 */

/* arguments */
x_len		.req	x0	/* vector length */
x_vec		.req	x1	/* number of source vectors (ie. data blocks) */
x_tbl		.req	x2
x_src		.req	x3
x_dest1		.req	x4
x_src2      .req    x9 

/* returns */
w_ret		.req	w0

/* local variables */
x_vec_i		.req	x5
x_ptr		.req	x6
x_pos		.req	x7
x_pos2		.req	x8

/* vectors */
z_src		.req	z0
z_src2      .req    z1

z_dest		.req	z2
z_dest2     .req    z3

cdecl(gf_vect_dot_prod_eor_sve):
	/* less than 16 bytes, return_fail */
	cmp	x_len, #16
	blt	.return_fail

	mov	x_pos, #0
    mov x_pos2, #32
	lsl	x_vec, x_vec, #3

/* Loop 1: x_len, vector length */
.Lloopsve_vl:
	whilelo	p0.b, x_pos, x_len
	b.none	.return_pass

    whilelo	p1.b, x_pos2, x_len

	mov	z_dest.b, #0			/* clear z_dest */
	mov	z_dest2.b, #0

    mov x_vec_i, #0

/* Loop 2: x_vec, number of source vectors (ie. data blocks) */
.Lloopsve_vl_vects:
	ldr	x_ptr, [x_src, x_vec_i]		/* x_ptr: src base addr. */
	/* load src data, governed by p0 */
	ld1b	z_src.b,  p0/z, [x_ptr, x_pos]	/* load from: src base + pos offset */
    ld1b	z_src2.b, p1/z, [x_ptr, x_pos2]

	add	x_vec_i, x_vec_i, #8		/* move x_vec_i to next */

	eor z_dest.d, z_dest.d, z_src.d
    eor z_dest2.d, z_dest2.d, z_src2.d

	cmp	x_vec_i, x_vec
	blt	.Lloopsve_vl_vects

	/* end of Loop 2 */
	/* store dest data, governed by p0 */
	st1b	z_dest.b, p0, [x_dest1, x_pos]
    st1b	z_dest2.b, p1, [x_dest1, x_pos2]
	/* increment two vector length */
	incb	x_pos
    incb    x_pos
    incb    x_pos2
    incb    x_pos2

	b	.Lloopsve_vl

.return_pass:
	mov	w_ret, #0
	ret

.return_fail:
	mov	w_ret, #1
	ret
