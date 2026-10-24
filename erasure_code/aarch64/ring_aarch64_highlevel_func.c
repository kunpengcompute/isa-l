#include "erasure_code.h"
#include "simd_opt.h"
#include "test.h"
#include "ring_internal.h"

#define RING_KEY(k, p) \
    ((((unsigned int)(k)) << 8) | ((unsigned int)(p)))

void ring_encode_data_simd(int len, int k, int rows, int node_size, unsigned char **data,
                     unsigned char **coding)
{
    switch (RING_KEY(k, rows)) {
    case RING_KEY(4, 2):
        ring_4_2_encode_data(
            k, node_size, data, coding);
        return;
    case RING_KEY(8, 3):
        ring_8_3_encode_data(
            len, k, rows, node_size, data, coding);
        return;
    case RING_KEY(10, 1):
        ring_10_1_encode_data(
            len, k, rows, node_size, data, coding);
        return;
    }
}

void ring_decode_data_simd(int len, int m, int k, int node_size, unsigned char **buffs,
                     unsigned char *src_err_list, int nerrs, unsigned char **temp_buffs)
{
    switch (RING_KEY(k, m-k)) {
    case RING_KEY(4, 2):
        ring_4_2_decode_data(
            node_size, buffs, src_err_list, nerrs, temp_buffs);
        return;
    case RING_KEY(8, 3):
        ring_8_3_decode_data(
            len, m, k, node_size, buffs, src_err_list, nerrs, temp_buffs);
        return;
    case RING_KEY(10, 1):
        ring_10_1_decode_data(
            len, m, k, node_size, buffs, src_err_list, nerrs, temp_buffs);
        return;
    }
}