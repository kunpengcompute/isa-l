#ifndef RING_INTERNAL_H
#define RING_INTERNAL_H

#include <stddef.h>
#include <string.h>

#define RING_ROW_FLAG      0x80u
#define RING_RECOVERY_FLAG 0x40u

#define RING_MAX_K      10
#define RING_MAX_PARITY 3
#define RING_MAX_SHARDS 11
#define RING_META_SIZE  11

#define RING_OP_ENCODE 0
#define RING_OP_DECODE 1

#define RING_NOT_HANDLED 0
#define RING_HANDLED     1
#define RING_INVALID     -1

static inline int
ring_config_id(int k, int parity)
{
        if (k == 4 && parity == 2)
                return 0;

        if (k == 8 && parity == 3)
                return 1;

        if (k == 10 && parity == 1)
                return 2;

        return -1;
}

static inline int
ring_config_parity(int config, int k)
{
        if (config == 0 && k == 4)
                return 2;
        if (config == 1 && k == 8)
                return 3;
        if (config == 2 && k == 10)
                return 1;
        return 0;
}

static inline unsigned char
ring_row_code(int config, int recovery, int index)
{
        return (unsigned char) (RING_ROW_FLAG | (recovery ? RING_RECOVERY_FLAG : 0u) |
                                ((unsigned int) config << 4) | (unsigned int) index);
}

static inline int
ring_read_row(const unsigned char *row, int k, int *config, int *recovery, int *index)
{
        unsigned int code = row[0];
        int parity;

        if (!(code & RING_ROW_FLAG))
                return 0;

        *config = (int) ((code >> 4) & 3u);
        *recovery = !!(code & RING_RECOVERY_FLAG);
        *index = (int) (code & 15u);

        switch (*config) {
        case 0:
                if (k != 4)
                        return 0;
                parity = 2;
                break;

        case 1:
                if (k != 8)
                        return 0;
                parity = 3;
                break;

        case 2:
                if (k != 10)
                        return 0;
                parity = 1;
                break;

        default:
                return 0;
        }

        if (*index >= (*recovery ? k : k + parity))
                return 0;

        for (int j = 1; j < k; j++)
                if (row[j] != 0)
                        return 0;

        return 1;
}

void
ring_4_2_encode_data(int k, int node_size, unsigned char **data, unsigned char **coding);

void
ring_4_2_decode_data(int node_size, unsigned char **buffs, unsigned char *src_err_list, int nerrs,
                     unsigned char **temp_buffs);
void
ring_8_3_encode_data(int len, int k, int rows, int node_size, unsigned char **data,
                     unsigned char **coding);

void
ring_8_3_decode_data(int len, int m, int k, int node_size, unsigned char **buffs,
                     unsigned char *src_err_list, int nerrs, unsigned char **temp_buffs);
void
ring_10_1_encode_data(int len, int k, int rows, int node_size, unsigned char **data,
                      unsigned char **coding);
void
ring_10_1_decode_data(int len, int m, int k, int node_size, unsigned char **buffs,
                      unsigned char *src_err_list, int nerrs, unsigned char **temp_buffs);
#endif