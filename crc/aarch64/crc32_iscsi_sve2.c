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

#define CRC_AINLINE static  __inline __attribute__((always_inline))

static const uint32_t crc32_table_iscsi_refl[256] = {
        0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4, 0xC79A971F, 0x35F1141C, 0x26A1E7E8,
        0xD4CA64EB, 0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B, 0x4D43CFD0, 0xBF284CD3,
        0xAC78BF27, 0x5E133C24, 0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B, 0xD7C45070,
        0x25AFD373, 0x36FF2087, 0xC494A384, 0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54,
        0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B, 0x20BD8EDE, 0xD2D60DDD, 0xC186FE29,
        0x33ED7D2A, 0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35, 0xAA64D611, 0x580F5512,
        0x4B5FA6E6, 0xB93425E5, 0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA, 0x30E349B1,
        0xC288CAB2, 0xD1D83946, 0x23B3BA45, 0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A,
        0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A, 0x7DA08661, 0x8FCB0562, 0x9C9BF696,
        0x6EF07595, 0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48, 0x86E18AA3, 0x748A09A0,
        0x67DAFA54, 0x95B17957, 0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687, 0x0C38D26C,
        0xFE53516F, 0xED03A29B, 0x1F682198, 0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927,
        0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38, 0xDBFC821C, 0x2997011F, 0x3AC7F2EB,
        0xC8AC71E8, 0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7, 0x61C69362, 0x93AD1061,
        0x80FDE395, 0x72966096, 0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789, 0xEB1FCBAD,
        0x197448AE, 0x0A24BB5A, 0xF84F3859, 0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46,
        0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9, 0xB602C312, 0x44694011, 0x5739B3E5,
        0xA55230E6, 0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36, 0x3CDB9BDD, 0xCEB018DE,
        0xDDE0EB2A, 0x2F8B6829, 0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C, 0x456CAC67,
        0xB7072F64, 0xA457DC90, 0x563C5F93, 0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043,
        0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C, 0x92A8FC17, 0x60C37F14, 0x73938CE0,
        0x81F80FE3, 0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC, 0x1871A4D8, 0xEA1A27DB,
        0xF94AD42F, 0x0B21572C, 0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033, 0xA24BB5A6,
        0x502036A5, 0x4370C551, 0xB11B4652, 0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
        0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D, 0xEF087A76, 0x1D63F975, 0x0E330A81,
        0xFC588982, 0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D, 0x758FE5D6, 0x87E466D5,
        0x94B49521, 0x66DF1622, 0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2, 0xFF56BD19,
        0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED, 0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530,
        0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F, 0x49547E0B, 0xBB3FFD08, 0xA86F0EFC,
        0x5A048DFF, 0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0, 0xD3D3E1AB, 0x21B862A8,
        0x32E8915C, 0xC083125F, 0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540, 0x590AB964,
        0xAB613A67, 0xB831C993, 0x4A5A4A90, 0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F,
        0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE, 0x24AA3F05, 0xD6C1BC06, 0xC5914FF2,
        0x37FACCF1, 0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321, 0xAE7367CA, 0x5C18E4C9,
        0x4F48173D, 0xBD23943E, 0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81, 0x34F4F86A,
        0xC69F7B69, 0xD5CF889D, 0x27A40B9E, 0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E,
        0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351
};

static unsigned int crc32_iscsi_base(unsigned char *buffer, int len, unsigned int crc_init)
{
        unsigned int crc;
        unsigned char *p_buf;
        unsigned char *p_end = buffer + len;

        p_buf = buffer;
        crc = crc_init;

        while (p_buf < p_end) {
                crc = (crc >> 8) ^ crc32_table_iscsi_refl[(crc & 0x000000FF) ^ *p_buf++];
        }
        return crc;
}

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
        crc_result = crc32_iscsi_base(buf, len, crc);
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
        crc_result = crc32_iscsi_base(buf, len, crc_result);
    }

    return crc_result;
}