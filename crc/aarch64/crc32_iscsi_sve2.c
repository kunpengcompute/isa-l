#include <arm_sve.h>
#include <arm_acle.h>
#include <arm_neon.h>

#define P1_LOW_B0  0x0dfe
#define P1_LOW_B1  0xf20c
#define P1_HIGH_B0 0x7d27
#define P1_HIGH_B1 0x493c
#define P0_LOW_B0  0xaab8
#define P0_LOW_B1  0xdd45
#define P0_HIGH_b0 P1_HIGH_B0
#define P0_HIGH_b1 P1_HIGH_B1
#define BR_HIGH_B0 0x13f1
#define BR_HIGH_B1 0xdea7
#define BR_HIGH_B2 0x0
#define BR_LOW_B0  0x76f1
#define BR_LOW_B1  0x05ec
#define BR_LOW_B2  0x1

extern unsigned int crc32_iscsi_crc_ext(unsigned char *, int , unsigned int);

#define CRC_AINLINE static  __inline __attribute__((always_inline))

CRC_AINLINE void extract_sve256_to_neon(svuint64_t sve_vec, uint64x2_t *low, uint64x2_t *high) {
    //1. 创建对齐的存储缓冲区
    uint8_t buffer[32] __attribute__((aligned(32)));

    //2. 将SVE向量存储到内存
    svbool_t pg = svptrue_b8();
    svst1(pg, buffer, sve_vec);

    //3. 从缓冲区加载低128位和高128位到NEON寄存器
    *low = vld1q_u64((const uint64_t*)buffer);
    *high = vld1q_u64((const uint64_t*)(buffer + 16));
}

CRC_AINLINE void crc32_fold_256b_to_128b(
    uint64x2_t *v_x0,
    uint64x2_t *v_x1
)
{
    //1. 构造多项式常量向量
    uint64_t p1_low  = ((uint64_t)P1_LOW_B1 << 16) | P1_LOW_B0;
    uint64_t p1_high = ((uint64_t)P1_HIGH_B1 << 16) | P1_HIGH_B0;
    poly64x2_t v_p1 = vcombine_p64(vcreate_p64(p1_low), vcreate_p64(p1_high));

    //2. 临时变量
    poly128_t v_tmp_high, v_tmp_low;

    poly64_t p1_low_scalar = vget_lane_p64(vget_low_p64(v_p1), 0);
    poly64_t x0_low_scalar = vget_lane_p64(vget_low_p64((poly64x2_t)*v_x0), 0);

    //3.第一轮折叠：处理v_x0,结果累积到 v_x1
    v_tmp_high = vmull_high_p64(v_p1, (poly64x2_t)*v_x0);
    v_tmp_low  = vmull_p64(p1_low_scalar, x0_low_scalar);
    *v_x1 = veorq_u64(*v_x1, (uint64x2_t)v_tmp_high);
    *v_x1 = veorq_u64(*v_x1, (uint64x2_t)v_tmp_low);
}

CRC_AINLINE void crc32_fold_2048b_to_256b(
    svuint64_t *x1, svuint64_t *x2,
    svuint64_t *x3, svuint64_t *x4,
    svuint64_t *x5, svuint64_t *x6,
    svuint64_t *x7, svuint64_t *x8
)
{
    svbool_t pg = svptrue_b64();
    static const uint32_t p4[] __attribute__((aligned(16))) = {0x6992cea2, 0x00000000, 0xd3b6092, 0x00000000,
                                                               0x6992cea2, 0x00000000, 0xd3b6092, 0x00000000 };
    static const uint32_t p3[] __attribute__((aligned(16))) = {0x740eef02, 0x00000000, 0x9e4addf8, 0x00000000,
                                                               0x740eef02, 0x00000000, 0x9e4addf8, 0x00000000 };                                                         
    static const uint32_t p2[] __attribute__((aligned(16))) = {0x3da6d0cb, 0x00000000, 0xba4fc28e, 0x00000000,
                                                               0x3da6d0cb, 0x00000000, 0xba4fc28e, 0x00000000 };

    svuint64_t x0   = svld1_u64(pg, (uint64_t *)p4);
    svuint64_t x00  = svld1_u64(pg, (uint64_t *)p3);
    svuint64_t x000 = svld1_u64(pg, (uint64_t *)p2);

    svuint64_t x9 = svpmullt_pair_u64(*x1, x0);
    *x1 = svpmullb_pair_u64(*x1, x0);
    *x1 = sveor3_u64(*x1, x9, *x5);

    svuint64_t x10 = svpmullt_pair_u64(*x2, x0);
    *x2 = svpmullb_pair_u64(*x2, x0);
    *x2 = sveor3_u64(*x2, x10, *x6);

    svuint64_t x11 = svpmullt_pair_u64(*x3, x0);
    *x3 = svpmullb_pair_u64(*x3, x0);
    *x3 = sveor3_u64(*x3, x11, *x7);

    svuint64_t x12 = svpmullt_pair_u64(*x4, x0);
    *x4 = svpmullb_pair_u64(*x4, x0);
    *x4 = sveor3_u64(*x4, x12, *x8);

    svuint64_t x13 = svpmullt_pair_u64(*x1, x00);
    *x1 = svpmullb_pair_u64(*x1, x00);
    *x1 = sveor3_u64(*x1, x13, *x3);
    
    svuint64_t x14 = svpmullt_pair_u64(*x2, x00);
    *x2 = svpmullb_pair_u64(*x2, x00);
    *x2 = sveor3_u64(*x2, x14, *x4);

    svuint64_t x15 = svpmullt_pair_u64(*x1, x000);
    *x1 = svpmullb_pair_u64(*x1, x000);
    *x1 = sveor3_u64(*x1, x15, *x2);
}

CRC_AINLINE uint64x2_t crc32_final_fold(uint64x2_t *v_x3) {
    // 1. 构造多项式常量
    uint64_t p0_low = ((uint64_t)P0_LOW_B1 << 16) | P0_LOW_B0;
    poly64_t d_p0_low2 = vget_lane_p64(vcreate_p64(p0_low), 0);

    //2. 构造 v_p1
    uint64_t p1_low = ((uint64_t)P1_LOW_B1 << 16) | P1_LOW_B0;
    uint64_t p1_high = ((uint64_t)P1_HIGH_B1 << 16) | P1_HIGH_B0;
    poly64x2_t v_p1 = vcombine_p64(vcreate_p64(p1_low), vcreate_p64(p1_high));

    //3. 提取 v_x3的高64位
    uint64_t d_tmp_high = vgetq_lane_u64(*v_x3, 1);

    //4. 执行第一次多项式乘法
    //mov d_p0_low, v_p1.d[1]
    poly64_t d_x3_low     = vget_lane_p64(vget_low_p64((poly64x2_t)*v_x3), 0);
    //pmull v_x3.1q, v_x3.1d, v_p0.1d
    poly128_t v_x3_result = vmull_p64(d_x3_low, vget_lane_p64(vget_high_p64(v_p1), 0));

    //5. 高低位异或操作
    uint64x2_t v_tmp_high = vsetq_lane_u64(d_tmp_high, v_tmp_high, 0);
    v_tmp_high = veorq_u64(v_tmp_high, (uint64x2_t)v_x3_result);

    //6. 提取低32位
    uint32_t s_x3 = vgetq_lane_u32(v_tmp_high, 0);

    //7. 字节移位，获取高32bit
    // d_tmp_high >>= 32;
    v_tmp_high = vextq_u8(v_tmp_high, v_tmp_high, 4);

    //8. 第二次多项式乘法
    v_x3_result = vmull_p64(d_p0_low2, vget_lane_p64(vcreate_p64(s_x3), 0));

    //9. 更新结果
    *v_x3 = vsetq_lane_u64(vgetq_lane_u64((uint64x2_t)v_x3_result, 0), *v_x3, 0);
    *v_x3 = vsetq_lane_u64(0, *v_x3, 1);

    return v_tmp_high;
}

CRC_AINLINE uint32_t barrett_reduction(uint64x2_t v_x3, uint64x2_t v_tmp_high) {
    //1. 构造 br_high 多项式
    uint64_t br_high = ((uint64_t)BR_HIGH_B2 << 32) |
                       ((uint64_t)BR_HIGH_B1 << 16) |
                       BR_HIGH_B0;
    poly64_t d_br_high = vget_lane_u64(vcreate_p64(br_high), 0);

    //2. 构造 br_low 多项式
    uint64_t br_low = ((uint64_t)BR_LOW_B2 << 32) |
                      ((uint64_t)BR_LOW_B1 << 16) |
                      BR_LOW_B0;
    poly64_t d_br_low = vget_lane_u64(vcreate_p64(br_low), 0);


    v_tmp_high = veorq_u64(v_tmp_high, (uint64x2_t)v_x3);
    //3. 提取 v_tmp_high 的低32位
    uint32_t s_x3 = vgetq_lane_u32((uint32x4_t)v_tmp_high, 0);
    //4. 第一次多项式乘法
    poly128_t v_x3_result = vmull_p64(d_br_high, vget_lane_p64(vcreate_p64(s_x3), 0));

    //5. 提取结果的低32位
    s_x3 = vgetq_lane_u32((uint32x4_t)v_x3_result, 0);

    //6. 第二次多项式乘法
    v_x3_result = vmull_p64(d_br_low, vget_lane_u64(vcreate_p64(s_x3), 0));

    uint64x2_t result = veorq_u64((uint64x2_t)v_x3_result, (uint64x2_t)v_tmp_high);

    //7. 提取最终结果
    uint32_t w_seed = vgetq_lane_u32((uint32x4_t)result, 1);

    return w_seed;
}

uint32_t crc32_iscsi_sve2(const unsigned char *buf, int len, uint32_t crc)
{
    static const uint32_t p4[] __attribute__((aligned(16))) = { 0xdcb17aa4, 0x00000000, 0xb9e02b86, 0x00000000, 0xdcb17aa4, 0x00000000, 0xb9e02b86, 0x00000000 };
    uint32_t crc_result = crc;
    svuint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;

    svbool_t pg = svptrue_b64();
    const int vector_bytes = svcntb();

    const int block_size = vector_bytes * 8;  //处理8个vector数据
    if(len < block_size) {
        crc_result = crc32_iscsi_crc_ext(buf, len, crc);
        return crc_result;
    } 

    /*初始加载 */
    x1 = svld1_u64(pg, (const uint64_t *)(buf));
    x2 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes));
    x3 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 2));
    x4 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 3));
    x5 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 4));
    x6 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 5));
    x7 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 6));
    x8 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 7));

    svbool_t pg_first = svwhilelt_b64(0, 1);
    x1 = sveor_u64_z(pg, x1, svsplice_u64(pg_first,
                                    svdup_n_u64((uint64_t)crc),
                                    svdup_n_u64(0)));
    // x0就是多项式
    x0 = svld1_u64(pg, (uint64_t*)p4);
    buf += block_size;
    len -= block_size;
    
    /*主循环处理 */
    while (len >= block_size) {
        //计算高部分
        x9  = svpmullt_pair_u64(x1, x0);
        x10 = svpmullt_pair_u64(x2, x0);
        x11 = svpmullt_pair_u64(x3, x0);
        x12 = svpmullt_pair_u64(x4, x0);
        x13 = svpmullt_pair_u64(x5, x0);
        x14 = svpmullt_pair_u64(x6, x0);
        x15 = svpmullt_pair_u64(x7, x0);
        x16 = svpmullt_pair_u64(x8, x0);

        //计算低部分
        x1 = svpmullb_pair_u64(x1, x0);
        x2 = svpmullb_pair_u64(x2, x0);
        x3 = svpmullb_pair_u64(x3, x0);
        x4 = svpmullb_pair_u64(x4, x0);
        x5 = svpmullb_pair_u64(x5, x0);
        x6 = svpmullb_pair_u64(x6, x0);
        x7 = svpmullb_pair_u64(x7, x0);
        x8 = svpmullb_pair_u64(x8, x0);

        //加载下一组数据
        svuint64_t y1 = svld1_u64(pg, (const uint64_t *)(buf));
        svuint64_t y2 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes));
        svuint64_t y3 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 2));
        svuint64_t y4 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 3));
        svuint64_t y5 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 4));
        svuint64_t y6 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 5));
        svuint64_t y7 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 6));
        svuint64_t y8 = svld1_u64(pg, (const uint64_t *)(buf + vector_bytes * 7));
        
        //高低部分合并
        x1 = sveor3_u64(x1, x9,  y1);
        x2 = sveor3_u64(x2, x10, y2);
        x3 = sveor3_u64(x3, x11, y3);
        x4 = sveor3_u64(x4, x12, y4);
        x5 = sveor3_u64(x5, x13, y5);
        x6 = sveor3_u64(x6, x14, y6);
        x7 = sveor3_u64(x7, x15, y7);
        x8 = sveor3_u64(x8, x16, y8);

        buf += block_size;
        len -= block_size;
    }

    crc32_fold_2048b_to_256b(&x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8);
    uint64x2_t v0, v1;
    extract_sve256_to_neon(x1, &v0, &v1);
    crc32_fold_256b_to_128b(&v0, &v1);

    //128b fold to 64b
    uint64x2_t high_tmp = crc32_final_fold(&v1);
    //barrent_reduction
    crc_result = barrett_reduction(v1. high_tmp);

    if (len > 0) {
        crc_result = crc32_iscsi_crc_ext(buf, len, crc_result);
    }

    return crc_result;
}