.arch armv8.5-a+crypto+crc+sha3
	.text
	.align	2
	.p2align 4,,11
	.global	crc32_iscsi_crc_mix
	.type	crc32_iscsi_crc_mix, %function
crc32_iscsi_crc_mix:
.LFB4311:
	.cfi_startproc
	cmp	w1, 255
	ble	.L83
	asr	w11, w1, 8
	stp	x29, x30, [sp, -96]!
	.cfi_def_cfa_offset 96
	.cfi_offset 29, -96
	.cfi_offset 30, -88
	mov	w8, 96
	sbfiz	x14, x11, 5, 32
	sbfiz	x3, x11, 7, 32
	mov	x10, x0
	mov	x29, sp
	mov	w7, w2
	add	x2, x0, x3
	add	x13, x14, 4
	smull	x8, w11, w8
	adrp	x17, .LANCHOR0
	stp	x19, x20, [sp, 16]
	lsl	x13, x13, 1
	.cfi_offset 19, -80
	.cfi_offset 20, -72
	sub	x19, x14, #64
	stp	x21, x22, [sp, 32]
	add	x1, x0, x1, sxtw
	adds	x18, x0, x19
	stp	x23, x24, [sp, 48]
	add	x16, x13, 8
	add	x15, x13, 16
	stp	x25, x26, [sp, 64]
	add	x20, x2, 128
	.cfi_offset 21, -64
	.cfi_offset 22, -56
	.cfi_offset 23, -48
	.cfi_offset 24, -40
	.cfi_offset 25, -32
	.cfi_offset 26, -24
	add	x26, x14, 8
	ldr	q0, [x17, #:lo12:.LANCHOR0]
	add	x24, x14, 16
	add	x22, x14, 24
	add	x25, x8, 8
	add	x23, x8, 16
	add	x21, x8, 24
	sxtw	x0, w11
	sbfiz	x30, x11, 6, 32
	ldr	q1, [x10, x3]
	ldp	q26, q3, [x2, 16]
	ldp	q25, q22, [x2, 48]
	ldp	q24, q2, [x2, 80]
	ldr	q4, [x2, 112]
	bcs	.L25
	add	x8, x10, x8
	add	x9, x10, x14
	mov	x2, x20
	mov	x6, x10
	mov	w5, 0
	mov	w4, 0
	mov	w3, 0
	stp	x27, x28, [sp, 80]
	.cfi_offset 28, -8
	.cfi_offset 27, -16
	.p2align 3,,7
.L4:
	pmull2 v7.1q, v22.2d, v0.2d

	ldr	x28, [x6]
	pmull v18.1q, v22.1d, v0.1d

	ldr	q22, [x2]
	pmull v28.1q, v1.1d, v0.1d

	crc32cx	w28, w7, x28
	add	x2, x2, 128
	ldr	x7, [x6, 8]
	pmull2 v1.1q, v1.2d, v0.2d

	eor3	v1.16b, v1.16b, v28.16b, v22.16b
	ldr	q22, [x2, -112]
	pmull2 v19.1q, v26.2d, v0.2d

	crc32cx	w7, w28, x7
	ldp	x12, x28, [x8]
	pmull v23.1q, v26.1d, v0.1d

	eor3	v26.16b, v19.16b, v23.16b, v22.16b
	ldr	q19, [x2, -96]
	pmull v21.1q, v3.1d, v0.1d

	pmull2 v17.1q, v25.2d, v0.2d

	pmull2 v5.1q, v24.2d, v0.2d

	crc32cx	w5, w5, x12
	add	x8, x8, 32
	ldp	x27, x12, [x9]
	crc32cx	w5, w5, x28
	ldr	x28, [x9, 16]
	pmull2 v3.1q, v3.2d, v0.2d

	pmull v20.1q, v25.1d, v0.1d

	pmull v16.1q, v24.1d, v0.1d

	crc32cx	w27, w3, x27
	eor3	v3.16b, v3.16b, v21.16b, v19.16b
	ldr	x3, [x6, x30]
	pmull v6.1q, v2.1d, v0.1d

	ldr	q19, [x2, -80]
	eor3	v25.16b, v17.16b, v20.16b, v19.16b
	pmull v27.1q, v4.1d, v0.1d

	crc32cx	w4, w4, x3
	add	x9, x9, 32
	ldr	x3, [x6, x13]
	pmull2 v2.1q, v2.2d, v0.2d

	ldr	q17, [x2, -64]
	eor3	v22.16b, v7.16b, v18.16b, v17.16b
	pmull2 v4.1q, v4.2d, v0.2d

	crc32cx	w4, w4, x3
	crc32cx	w3, w27, x12
	ldr	x27, [x6, 16]
	crc32cx	w3, w3, x28
	ldr	x28, [x8, -16]
	crc32cx	w7, w7, x27
	ldr	x12, [x6, x16]
	crc32cx	w5, w5, x28
	ldr	x27, [x6, 24]
	crc32cx	w4, w4, x12
	ldr	x28, [x8, -8]
	crc32cx	w7, w7, x27
	ldr	x27, [x9, -8]
	ldr	x12, [x6, x15]
	add	x6, x6, 32
	ldr	q7, [x2, -48]
	eor3	v24.16b, v5.16b, v16.16b, v7.16b
	crc32cx	w5, w5, x28
	ldr	q5, [x2, -32]
	eor3	v2.16b, v2.16b, v6.16b, v5.16b
	crc32cx	w4, w4, x12
	crc32cx	w3, w3, x27
	ldr	q5, [x2, -16]
	eor3	v4.16b, v4.16b, v27.16b, v5.16b
	cmp	x18, x6
	bcs	.L4
	lsr	x19, x19, 5
	ldp	x27, x28, [sp, 80]
	.cfi_restore 28
	.cfi_restore 27
	add	x19, x19, 1
	add	x10, x10, x19, lsl 5
	add	x20, x20, x19, lsl 7
.L3:
	add	x2, x10, x14
	add	x17, x17, :lo12:.LANCHOR0
	add	x9, x2, x14
	ldr	x6, [x10, x14]
	ldr	x8, [x2, x14]
	ldr	x9, [x9, x14]
	crc32cx	w6, w3, x6
	crc32cx	w4, w4, x8
	ldr	x8, [x10, x26]
	crc32cx	w3, w5, x9
	ldr	x5, [x10, x25]
	crc32cx	w6, w6, x8
	ldp	x2, x8, [x10]
	crc32cx	w3, w3, x5
	ldr	x9, [x10, x13]
	ldr	x5, [x10, x16]
	crc32cx	w2, w7, x2
	crc32cx	w4, w4, x9
	ldr	x7, [x10, x24]
	crc32cx	w4, w4, x5
	ldr	x5, [x10, x22]
	crc32cx	w2, w2, x8
	crc32cx	w6, w6, x7
	ldr	x8, [x10, x23]
	crc32cx	w6, w6, x5
	ldp	x7, x5, [x10, 16]
	crc32cx	w3, w3, x8
	ldr	x8, [x10, x15]
	ldr	q5, [x17, 16]
	crc32cx	w2, w2, x7
	ldr	x7, [x10, x21]
	pmull v7.1q, v3.1d, v5.1d

	pmull2 v3.1q, v3.2d, v5.2d

	eor3	v3.16b, v3.16b, v7.16b, v25.16b
	crc32cx	w4, w4, x8
	pmull v6.1q, v1.1d, v5.1d

	pmull2 v0.1q, v1.2d, v5.2d

	pmull v7.1q, v22.1d, v5.1d

	eor3	v0.16b, v0.16b, v6.16b, v26.16b
	crc32cx	w3, w3, x7
	crc32cx	w2, w2, x5
	ldr	q6, [x17, 32]
	pmull v1.1q, v0.1d, v6.1d

	pmull2 v0.1q, v0.2d, v6.2d

	eor3	v0.16b, v0.16b, v1.16b, v3.16b
	ldr	q3, [x17, 48]
	pmull v1.1q, v2.1d, v5.1d

	pmull2 v2.1q, v2.2d, v5.2d

	eor3	v2.16b, v2.16b, v1.16b, v4.16b
	pmull v4.1q, v0.1d, v3.1d

	pmull2 v1.1q, v22.2d, v5.2d

	pmull2 v0.1q, v0.2d, v3.2d

	eor3	v1.16b, v1.16b, v7.16b, v24.16b
	pmull v3.1q, v1.1d, v6.1d

	pmull2 v1.1q, v1.2d, v6.2d

	eor3	v1.16b, v1.16b, v3.16b, v2.16b
	eor3	v1.16b, v0.16b, v4.16b, v1.16b
	cmp	w11, 128
	bgt	.L5
	add	x8, x17, 64
	add	x7, x17, 592
	add	x5, x17, 1120
	add	x17, x17, 1648
	fmov	s0, w6
	fmov	s5, w2
	ldr	s16, [x8, x0, lsl 2]
	fmov	s4, w4
	ldr	s2, [x7, x0, lsl 2]
	fmov	s3, w3
	ldr	s7, [x5, x0, lsl 2]
	dup	v0.2d, v0.d[0]
	ldr	s6, [x17, x0, lsl 2]
	dup	v16.2d, v16.d[0]
	dup	v5.2d, v5.d[0]
	dup	v2.2d, v2.d[0]
	dup	v7.2d, v7.d[0]
	pmull v2.1q, v0.1d, v2.1d

	dup	v4.2d, v4.d[0]
	dup	v0.2d, v6.d[0]
	dup	v3.2d, v3.d[0]
	pmull v5.1q, v5.1d, v16.1d

	pmull v4.1q, v4.1d, v7.1d

	pmull v3.1q, v3.1d, v0.1d

.L6:
	eor	v0.16b, v2.16b, v4.16b
	mov	w0, 0
	dup	d2, v1.d[1]
	sub	x1, x1, x20
	umov	x2, v1.d[0]
	eor3	v0.16b, v5.16b, v0.16b, v3.16b
	eor	v0.8b, v2.8b, v0.8b
	crc32cx	w0, w0, x2
	fmov	x2, d0
	crc32cx	w0, w0, x2
	cmp	w1, 0
	bgt	.L85
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x23, x24, [sp, 48]
	ldp	x25, x26, [sp, 64]
	ldp	x29, x30, [sp], 96
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L85:
	.cfi_restore_state
	mov	w2, w0
	mov	x0, x20
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x23, x24, [sp, 48]
	ldp	x25, x26, [sp, 64]
	ldp	x29, x30, [sp], 96
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
.L83:
	b	crc32_iscsi_crc_ext
	.p2align 2,,3
.L5:
	.cfi_def_cfa_offset 96
	.cfi_offset 19, -80
	.cfi_offset 20, -72
	.cfi_offset 21, -64
	.cfi_offset 22, -56
	.cfi_offset 23, -48
	.cfi_offset 24, -40
	.cfi_offset 25, -32
	.cfi_offset 26, -24
	.cfi_offset 29, -96
	.cfi_offset 30, -88
	mov	w7, 1792
	mov	x0, -33
	mov	x5, -2
	smaddl	x7, w11, w7, x0
	.p2align 3,,7
.L7:
	lsr	x8, x7, 1
	and	x0, x7, 1
	sub	x7, x8, #16
	add	x5, x0, x5, lsl 1
	cmp	x7, 191
	bhi	.L7
	mov	w0, -2147483648
	mov	w8, 0
	lsr	w0, w0, w7
	lsr	x7, x7, 5
	crc32cw	w0, w0, w8
	mvn	x5, x5
	crc32cw	w0, w0, w8
	cmp	x7, 2
	beq	.L8
	crc32cw	w0, w0, w8
	subs	x7, x7, #3
	beq	.L8
	crc32cw	w0, w0, w8
	cmp	x7, 1
	crc32cw	w8, w0, w8
	csel	w0, w8, w0, ne
.L8:
	and	w7, w5, 1
	lsr	x5, x5, 1
	cbz	x5, .L9
	mov	w8, 0
	.p2align 3,,7
.L10:
	fmov	s0, w0
	pmull v0.8h, v0.8b, v0.8b
	umov	x0, v0.d[0]
	lsl	x0, x0, x7
	and	w7, w5, 1
	crc32cx	w0, w8, x0
	lsr	x5, x5, 1
	cbnz	x5, .L10
.L9:
	mov	w5, 1536
	mov	x7, -33
	fmov	s5, w2
	fmov	s0, w0
	smaddl	x5, w11, w5, x7
	mov	x2, -2
	dup	v5.2d, v5.d[0]
	dup	v0.2d, v0.d[0]
	pmull v5.1q, v5.1d, v0.1d

	.p2align 3,,7
.L11:
	lsr	x7, x5, 1
	and	x0, x5, 1
	sub	x5, x7, #16
	add	x2, x0, x2, lsl 1
	cmp	x5, 191
	bhi	.L11
	mov	w0, -2147483648
	mov	w7, 0
	lsr	w0, w0, w5
	lsr	x5, x5, 5
	crc32cw	w0, w0, w7
	mvn	x2, x2
	crc32cw	w0, w0, w7
	cmp	x5, 2
	beq	.L12
	crc32cw	w0, w0, w7
	subs	x5, x5, #3
	beq	.L12
	crc32cw	w0, w0, w7
	cmp	x5, 1
	crc32cw	w7, w0, w7
	csel	w0, w7, w0, ne
.L12:
	and	w5, w2, 1
	lsr	x2, x2, 1
	cbz	x2, .L13
	mov	w7, 0
	.p2align 3,,7
.L14:
	fmov	s0, w0
	pmull v0.8h, v0.8b, v0.8b
	umov	x0, v0.d[0]
	lsl	x0, x0, x5
	and	w5, w2, 1
	crc32cx	w0, w7, x0
	lsr	x2, x2, 1
	cbnz	x2, .L14
.L13:
	mov	x2, -33
	mov	w5, 1280
	fmov	s2, w6
	fmov	s0, w0
	smaddl	x5, w11, w5, x2
	mov	x2, -2
	dup	v2.2d, v2.d[0]
	dup	v0.2d, v0.d[0]
	pmull v2.1q, v2.1d, v0.1d

	.p2align 3,,7
.L15:
	lsr	x6, x5, 1
	and	x0, x5, 1
	sub	x5, x6, #16
	add	x2, x0, x2, lsl 1
	cmp	x5, 191
	bhi	.L15
	mov	w0, -2147483648
	mov	w6, 0
	lsr	w0, w0, w5
	lsr	x5, x5, 5
	crc32cw	w0, w0, w6
	mvn	x2, x2
	crc32cw	w0, w0, w6
	cmp	x5, 2
	beq	.L16
	crc32cw	w0, w0, w6
	subs	x5, x5, #3
	beq	.L16
	crc32cw	w0, w0, w6
	cmp	x5, 1
	crc32cw	w6, w0, w6
	csel	w0, w6, w0, ne
.L16:
	and	w5, w2, 1
	lsr	x2, x2, 1
	cbz	x2, .L17
	mov	w6, 0
	.p2align 3,,7
.L18:
	fmov	s0, w0
	pmull v0.8h, v0.8b, v0.8b
	umov	x0, v0.d[0]
	lsl	x0, x0, x5
	and	w5, w2, 1
	crc32cx	w0, w6, x0
	lsr	x2, x2, 1
	cbnz	x2, .L18
.L17:
	mov	x2, -33
	mov	w5, 1024
	fmov	s4, w4
	fmov	s0, w0
	smaddl	x11, w11, w5, x2
	mov	x2, -2
	dup	v4.2d, v4.d[0]
	dup	v0.2d, v0.d[0]
	pmull v4.1q, v4.1d, v0.1d

	.p2align 3,,7
.L19:
	lsr	x4, x11, 1
	and	x0, x11, 1
	sub	x11, x4, #16
	add	x2, x0, x2, lsl 1
	cmp	x11, 191
	bhi	.L19
	mov	w0, -2147483648
	mov	w4, 0
	lsr	w0, w0, w11
	lsr	x11, x11, 5
	crc32cw	w0, w0, w4
	mvn	x2, x2
	crc32cw	w0, w0, w4
	cmp	x11, 2
	beq	.L20
	crc32cw	w0, w0, w4
	subs	x11, x11, #3
	beq	.L20
	crc32cw	w0, w0, w4
	cmp	x11, 1
	crc32cw	w4, w0, w4
	csel	w0, w4, w0, ne
.L20:
	and	w4, w2, 1
	lsr	x2, x2, 1
	cbz	x2, .L21
	mov	w5, 0
	.p2align 3,,7
.L22:
	fmov	s0, w0
	pmull v0.8h, v0.8b, v0.8b
	umov	x0, v0.d[0]
	lsl	x0, x0, x4
	and	w4, w2, 1
	crc32cx	w0, w5, x0
	lsr	x2, x2, 1
	cbnz	x2, .L22
.L21:
	fmov	s3, w3
	fmov	s0, w0
	dup	v3.2d, v3.d[0]
	dup	v0.2d, v0.d[0]
	pmull v3.1q, v3.1d, v0.1d

	b	.L6
	.p2align 2,,3
.L25:
	mov	w5, 0
	mov	w4, 0
	mov	w3, 0
	b	.L3
	.cfi_endproc
.LFE4311:
	.size	crc32_iscsi_crc_mix, .-crc32_iscsi_crc_mix
	.section	.rodata
	.align	4
	.set	.LANCHOR0,. + 0
	.type	k_.3, %object
	.size	k_.3, 16
k_.3:
	.xword	1771228834
	.xword	221995154
	.type	k_.2, %object
	.size	k_.2, 16
k_.2:
	.xword	4060876286
	.xword	1228700967
	.type	k_.1, %object
	.size	k_.1, 16
k_.1:
	.xword	1034342603
	.xword	3125789326
	.type	k_.0, %object
	.size	k_.0, 16
k_.0:
	.xword	1947135746
	.xword	2655706616
	.type	table1792, %object
	.size	table1792, 516
table1792:
	.word	0
	.word	-2093709262
	.word	453196159
	.word	-826055122
	.word	1757210746
	.word	-126687924
	.word	-909592702
	.word	-232317344
	.word	-1434682667
	.word	1171119950
	.word	-553933796
	.word	-1796528786
	.word	-2003340063
	.word	125654240
	.word	-2032915194
	.word	-1363941854
	.word	-1767771749
	.word	1717885377
	.word	619118223
	.word	596715026
	.word	1306659452
	.word	141671728
	.word	-509318960
	.word	-344612720
	.word	234956466
	.word	1909230056
	.word	-2107375126
	.word	1356931645
	.word	-1346345388
	.word	1528164917
	.word	-65665996
	.word	223508235
	.word	-1457312213
	.word	2141283417
	.word	1863679064
	.word	1542155509
	.word	-751172756
	.word	1467109944
	.word	-26099209
	.word	-312225747
	.word	-500515454
	.word	1428064498
	.word	1790014773
	.word	1143518834
	.word	-632416807
	.word	1728477135
	.word	-1908613724
	.word	-449165331
	.word	1474832662
	.word	1050334367
	.word	1281487342
	.word	-1401507353
	.word	-1614279034
	.word	1023435735
	.word	1584425186
	.word	1266092229
	.word	-1049838034
	.word	893010400
	.word	51479204
	.word	-103214040
	.word	446184297
	.word	-1615495272
	.word	-986736366
	.word	-637213383
	.word	429193274
	.word	-2130852001
	.word	883170304
	.word	600363283
	.word	-1401368093
	.word	-1322432084
	.word	603468486
	.word	-790319931
	.word	-277522028
	.word	1137270588
	.word	1658989944
	.word	-38123535
	.word	-319816801
	.word	1001281297
	.word	-14893500
	.word	1619791735
	.word	66639991
	.word	19419889
	.word	1639237408
	.word	787422611
	.word	1057332687
	.word	-143881691
	.word	-138747846
	.word	692796179
	.word	-1854992607
	.word	-1801704045
	.word	-43269093
	.word	-919054249
	.word	1275411545
	.word	-405324228
	.word	-680593819
	.word	1982412250
	.word	-830577733
	.word	-1609351010
	.word	-1383826329
	.word	-1224720987
	.word	-1866118564
	.word	170474100
	.word	807823454
	.word	-636209334
	.word	-1851121116
	.word	-124580460
	.word	-1231347959
	.word	-1987562031
	.word	606234419
	.word	804387599
	.word	-1812116484
	.word	1305246995
	.word	1732908996
	.word	845631520
	.word	-778889379
	.word	-534088371
	.word	-72788053
	.word	1061111855
	.word	-1409721409
	.word	1659693879
	.word	1801860704
	.word	1888917318
	.word	708745821
	.word	-80481361
	.word	-1605109253
	.word	-446262045
	.word	308477651
	.word	-2087318962
	.word	222519078
	.zero	12
	.type	table1536, %object
	.size	table1536, 516
table1536:
	.word	0
	.word	-1418002646
	.word	-764350046
	.word	656250948
	.word	-677084580
	.word	930212239
	.word	-2032605998
	.word	-909592702
	.word	-1628009163
	.word	12383734
	.word	-192527639
	.word	35309039
	.word	-1092850783
	.word	1273494928
	.word	-2003340063
	.word	1180323566
	.word	899052791
	.word	-1778395172
	.word	2135377469
	.word	1290321893
	.word	601221093
	.word	619118223
	.word	425445421
	.word	1251820696
	.word	6011236
	.word	-152643938
	.word	-2016380192
	.word	1208909389
	.word	234956466
	.word	-171702887
	.word	1808700229
	.word	-1263493695
	.word	-1177121463
	.word	288809158
	.word	-88583428
	.word	-65665996
	.word	1619425923
	.word	897621037
	.word	-1598212397
	.word	-275514297
	.word	-139686113
	.word	1788375104
	.word	-751172756
	.word	-1694698410
	.word	35773646
	.word	-582268314
	.word	469403423
	.word	1504797168
	.word	1809749723
	.word	1790014773
	.word	-243232812
	.word	1617820577
	.word	-1717854491
	.word	1980112636
	.word	-1238141144
	.word	-398615066
	.word	1474832662
	.word	877402229
	.word	2024025559
	.word	-218831715
	.word	-129459460
	.word	-832178982
	.word	907922345
	.word	1584425186
	.word	-1041129759
	.word	2096986845
	.word	1403088406
	.word	1426567796
	.word	850030342
	.word	-1505119745
	.word	446184297
	.word	235148916
	.word	2079749124
	.word	736146991
	.word	1829890452
	.word	106084992
	.word	-1442873225
	.word	883170304
	.word	287684578
	.word	-893543784
	.word	663240941
	.word	1449403918
	.word	1693905025
	.word	-242495270
	.word	-277522028
	.word	-1515445432
	.word	1024437090
	.word	223826164
	.word	452779341
	.word	271122020
	.word	-1570834715
	.word	-14893500
	.word	943984245
	.word	-186511217
	.word	-2073581915
	.word	-245005516
	.word	-538298116
	.word	487512067
	.word	1057332687
	.word	-1508379453
	.word	504779732
	.word	-438587969
	.word	62042090
	.word	1273972669
	.word	-843381767
	.word	-43269093
	.word	926099117
	.word	1180025802
	.word	-312285522
	.word	-36518119
	.word	95375316
	.word	1118621166
	.word	-830577733
	.word	-865617075
	.word	-1833753178
	.word	-1482085410
	.word	-912123617
	.word	-1679983244
	.word	312919289
	.word	807823454
	.word	1175426427
	.word	900204385
	.word	284769419
	.word	95699922
	.word	429992475
	.word	1936519835
	.word	606234419
	.word	1354007796
	.word	1779908513
	.zero	12
	.type	table1280, %object
	.size	table1280, 516
table1280:
	.word	0
	.word	-2020961625
	.word	-1161626245
	.word	-1276957656
	.word	1802805170
	.word	23581458
	.word	930212239
	.word	-126687924
	.word	-580465733
	.word	591527966
	.word	644152715
	.word	401766040
	.word	-192527639
	.word	-2131589013
	.word	-553933796
	.word	-691820537
	.word	583235999
	.word	1520563151
	.word	1180323566
	.word	1370419046
	.word	-129726912
	.word	-1363941854
	.word	1254565611
	.word	-777436548
	.word	601221093
	.word	-825242301
	.word	-993075918
	.word	-210926366
	.word	1306659452
	.word	1060001713
	.word	-152643938
	.word	-1827103464
	.word	-904853864
	.word	2035940906
	.word	-99669040
	.word	1909230056
	.word	1808700229
	.word	1817404608
	.word	-957935319
	.word	-946461356
	.word	-604613827
	.word	-11494704
	.word	-65665996
	.word	-517147407
	.word	-90832354
	.word	-384366945
	.word	1840093094
	.word	-1548830827
	.word	-139686113
	.word	1542155509
	.word	-517076459
	.word	-48337128
	.word	-1386019263
	.word	-1273478023
	.word	-582268314
	.word	1370077571
	.word	-500515454
	.word	-897891241
	.word	-609791894
	.word	628657770
	.word	-243232812
	.word	-1921609857
	.word	-1130084167
	.word	1728477135
	.word	-487480828
	.word	-1255101627
	.word	-398615066
	.word	-1495739398
	.word	210203815
	.word	1812465518
	.word	1281487342
	.word	1073163278
	.word	-129459460
	.word	-1210056828
	.word	756351435
	.word	1942791863
	.word	1737313984
	.word	1266092229
	.word	2096986845
	.word	-1715414374
	.word	-282056401
	.word	500280135
	.word	294868895
	.word	-1183012504
	.word	446184297
	.word	-717860185
	.word	-1692853050
	.word	555860600
	.word	1085710378
	.word	-1242383780
	.word	106084992
	.word	-2130852001
	.word	708689667
	.word	-15743300
	.word	379584545
	.word	1649346193
	.word	663240941
	.word	1442711093
	.word	603468486
	.word	1926022416
	.word	1000053327
	.word	-1583247894
	.word	-1515445432
	.word	277774007
	.word	-1173035198
	.word	-38123535
	.word	134398463
	.word	-28242721
	.word	-1570834715
	.word	858801279
	.word	-1033339145
	.word	-477815718
	.word	66639991
	.word	1284874023
	.word	-245005516
	.word	185403772
	.word	-1571024739
	.word	-1190417401
	.word	1114417860
	.word	-143881691
	.word	504779732
	.word	-954222700
	.word	1208158728
	.word	1143915109
	.word	1640614293
	.word	-1261142960
	.word	-43269093
	.word	1733342011
	.word	1757390658
	.zero	12
	.type	table1024, %object
	.size	table1024, 516
table1024:
	.word	0
	.word	221995154
	.word	-1176491130
	.word	-764350046
	.word	-578929908
	.word	1802805170
	.word	-677084580
	.word	1757210746
	.word	385906426
	.word	-2032605998
	.word	-580465733
	.word	-669797595
	.word	-1628009163
	.word	909891251
	.word	-1434682667
	.word	-192527639
	.word	-1524932299
	.word	-876720415
	.word	-1092850783
	.word	1674614761
	.word	583235999
	.word	-2003340063
	.word	-2062172033
	.word	-1072827776
	.word	899052791
	.word	-129726912
	.word	1594754104
	.word	2135377469
	.word	-1767771749
	.word	-1249196290
	.word	601221093
	.word	-948521400
	.word	-2097636233
	.word	425445421
	.word	-806357837
	.word	1306659452
	.word	6011236
	.word	948497667
	.word	450044348
	.word	-2016380192
	.word	-904853864
	.word	-1609847235
	.word	234956466
	.word	-1331433107
	.word	-471803713
	.word	1808700229
	.word	1724347613
	.word	1616805957
	.word	-1177121463
	.word	-1346345388
	.word	-604613827
	.word	-88583428
	.word	1371555710
	.word	134882085
	.word	1619425923
	.word	-90832354
	.word	-1457312213
	.word	-1598212397
	.word	300078721
	.word	-1184950432
	.word	-139686113
	.word	1743697189
	.word	-250309450
	.word	-751172756
	.word	1420321574
	.word	-1386019263
	.word	35773646
	.word	882677532
	.word	2117976915
	.word	469403423
	.word	-500515454
	.word	1239829903
	.word	1809749723
	.word	2029925504
	.word	-1224063123
	.word	-243232812
	.word	-784324111
	.word	-632416807
	.word	-1717854491
	.word	760760529
	.word	-487480828
	.word	-1238141144
	.word	979406859
	.word	-1144498617
	.word	1474832662
	.word	210203815
	.word	-357529722
	.word	2024025559
	.word	770376123
	.word	-766998106
	.word	-129459460
	.word	-1614279034
	.word	-1869547188
	.word	907922345
	.word	1569162294
	.word	1737313984
	.word	-1041129759
	.word	-879245001
	.word	-1049838034
	.word	1403088406
	.word	-282056401
	.word	1604149326
	.word	850030342
	.word	-842736196
	.word	-1167556705
	.word	446184297
	.word	-88474740
	.word	-81235700
	.word	2079749124
	.word	999989450
	.word	1085710378
	.word	1829890452
	.word	429193274
	.word	77115267
	.word	-1442873225
	.word	708689667
	.word	1927440907
	.word	287684578
	.word	890958141
	.word	-1401368093
	.word	663240941
	.word	-754642163
	.word	222531565
	.word	1693905025
	.word	-434100867
	.word	1000053327
	.word	-277522028
	.word	-777231838
	.word	499385292