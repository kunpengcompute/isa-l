/**
   1) The build command: g++ ec_perf_test.cpp -O3 -lisal -o ec_perf_test
   2) The test command: ./ec_perf_test.cpp -c 10000 -b 4k -m 1G -d 4 -p 2
 */
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include <chrono>
#include <getopt.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define TEST_REPEAT 100
using namespace std;
#ifdef __cplusplus
    extern "C" {
#endif
void ec_encode_data(int len, int k, int rows, unsigned char *gftbls, unsigned char **data, unsigned char **coding);

void ec_init_tables(int k, int rows, unsigned char *a, unsigned char *gftbls);

void gf_gen_rs_matrix(unsigned char *a, int m, int k);
#ifdef __cplusplus
    }
#endif

struct option g_longOptions[] = {
    {"memory", required_argument, NULL, 'm'},
    {"block", required_argument, NULL, 'b'},
    {"count", required_argument, NULL, 'c'},
    {"data", required_argument, NULL, 'd'},
    {"parity", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'}
};

void Usage() {
    std::cout << "    --block     -b     blocksize [1~]" << std::endl;
    std::cout << "    --count     -c     running count [~]" << std::endl;
    std::cout << "    --memory    -m     memory size eg. 1GB 500MB" << std::endl;
    std::cout << "    --data      -d     datanum [4]" << std::endl;
    std::cout << "    --parity    -p     paritynum [2]" << std::endl;
    std::cout << "    --help      -h     help" << std::endl;
}

using EcPerfParam = struct EcPerfParam {
    uint64_t memorySize;
    uint64_t blockSize;
    uint32_t count;
    uint32_t dataNum;
    uint32_t parityNum;
};

enum ec_block_uint {
    BYTE = 1,
    KB = 1024,
    MB = 1024 * 1024,
    GB = 1024 * 1024 * 1024
};

uint32_t str2byte(const char *str) {
    char num[100] = {0};
    enum ec_block_uint unit = BYTE;
    for (uint32_t i = 0; i < strlen(str); i++) {
        if (str[i] == 'k' || str[i] == 'K') {
            unit = KB;
            break;
        } else if (str[i] == 'm' || str[i] == 'M') {
            unit = MB;
            break;
        } else if (str[i] == 'g' || str[i] == 'G') {
            unit = GB;
            break;
        } else {
            num[i] = str[i];
        }
    }
    return unit * atoi(num);
}

int ParserArgument(EcPerfParam &param, int argc, char **argv) {
    int opt = 0;
    uint64_t blockSize;
    uint64_t memory;
    uint64_t dataNum;
    uint64_t parityNum;
    int count;
    while ((opt = getopt_long(argc, argv, "hb:c:m:s:d:p:", g_longOptions, NULL)) != -1) {
        switch (opt) {
            case 'b':
                blockSize = str2byte(optarg);
                if (blockSize <= 0) {
                    Usage();
                    exit(0);
                } 
                param.blockSize = blockSize;
                break;

            case 'd':
                dataNum = atoi(optarg);
                param.dataNum = dataNum;
                break;

            case 'p':
                parityNum = str2byte(optarg);
                param.parityNum = parityNum;
                break;   

            case 'c':
                count = atoi(optarg);
                if (count > 0) {
                    param.count = count;
                } else {
                    Usage();
                    exit(0);
                }
                break;

            case 'm':
                memory = str2byte(optarg);
                if (memory > 0) {
                    param.memorySize = memory;
                } else {
                    Usage();
                    exit(0);
                }
                break;
            case 'h':
                Usage();
                exit(0);
                break;
            
            default:
                Usage();
                exit(0);
                break;
        }
    }
    return 0;
}

#define MAX_STRIP_NUM 256 * 1024
uint8_t *g_stripes[MAX_STRIP_NUM][30];

int EcPerfTest(const EcPerfParam &param, uint8_t *buf, uint8_t **block) {
    
    uint32_t stripNum = param.memorySize / param.blockSize / (param.dataNum + param.parityNum);
    stripNum = (stripNum > MAX_STRIP_NUM) ? MAX_STRIP_NUM : stripNum;
    uint8_t a[30 * 30];
    gf_gen_rs_matrix(a, param.dataNum + param.parityNum, param.dataNum);
    uint32_t i = 0;
    uint32_t *strip = (uint32_t *)malloc(sizeof(uint32_t) * param.count * TEST_REPEAT);
    for(i = 0; i < param.count * TEST_REPEAT; i++) {
        strip[i] = rand() % (stripNum);
    }
    i = 0;
    uint8_t g_tbls[30 * 30 * 32];
    ec_init_tables(param.dataNum, param.parityNum, &a[param.dataNum * param.dataNum], g_tbls);
    auto start = std::chrono::high_resolution_clock::now();
    while (i < param.count * TEST_REPEAT) {
        uint32_t n = strip[i];
        ec_encode_data(param.blockSize, param.dataNum, param.parityNum, g_tbls, g_stripes[n], (unsigned char **)(g_stripes[n] + param.dataNum));
        i++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "ec_encode_data cost time " << (end - start).count() / (param.count * TEST_REPEAT) << "ns" << std::endl;
    std::cout << "ec_encode_data cost bandwidth " << 1.0 * param.blockSize * TEST_REPEAT *
     param.count * param.dataNum / (std::chrono::duration_cast<chrono::microseconds>(end -start)).count() << "MB/s" << std::endl;

    //test decode
    for (uint32_t decodeNum = 0 ; decodeNum < param.parityNum; decodeNum++) {
        i = 0;
        start = std::chrono::high_resolution_clock::now();
        while (i < param.count * TEST_REPEAT) {
            uint32_t n = strip[i];
            gf_gen_rs_matrix(a, param.dataNum + decodeNum + 1, param.dataNum);
            ec_init_tables(param.dataNum, decodeNum + 1, &a[param.dataNum * param.dataNum], g_tbls);
            ec_encode_data(param.blockSize, param.dataNum + param.parityNum, decodeNum + 1 ,g_tbls, g_stripes[n], (unsigned char **)(g_stripes[n] + param.dataNum));
            i++;
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "ec_decode_data " << decodeNum + 1 <<" cost time " << (end - start).count() / (param.count * TEST_REPEAT) << "ns" << std::endl;
        std::cout << "ec_decode_data " << decodeNum + 1 << " cost bandwidth "<< 1.0 * param.blockSize * TEST_REPEAT * param.count * param.dataNum 
              / (std::chrono::duration_cast<chrono::microseconds>(end -start)).count() << "MB/s" << std::endl;
    }
    free(strip);
    return 0;
}

int main(int argc, char **argv) {

    EcPerfParam param = {1024 * 1024 * 1024, 4096, 10000, 4, 2};
    int ret = ParserArgument(param, argc, argv);
    if (param.memorySize < param.blockSize) {
        std::cout << "memory size (" << param.memorySize << ") is smaller than block size (" << param.blockSize << ")" << std::endl;
        Usage();
        exit(0);
    }

    std::cout << "start memory size " << param.memorySize << ", block size " << param.blockSize << ", count " << param.count << std::endl;
    uint8_t *buf = (uint8_t *)malloc(param.memorySize);
    if (buf == nullptr) {
        std::cout << "malloc fault" << std::endl;
        exit(0);
    }
    uint64_t totalBlockCnt = param.memorySize / param.blockSize;
    uint8_t **blocks = (uint8_t **)malloc(sizeof(uint8_t *) * totalBlockCnt);
    for (uint64_t i = 0; i < totalBlockCnt; i++) {
        blocks[i] = buf + param.blockSize * i;
    }

    uint32_t stripNum = totalBlockCnt / (param.dataNum + param.parityNum);
    stripNum = (stripNum > MAX_STRIP_NUM) ? MAX_STRIP_NUM : stripNum;
    for (uint64_t i = 0; i < stripNum; i++) {
        for (uint32_t j = 0; j < param.dataNum + param.parityNum; j++) {
            g_stripes[i][j] = blocks[i * (param.dataNum + param.parityNum) + j];
        }
    }

    srand(time(NULL));
    int fd = open("/dev/urandom", O_RDONLY);
    #define MAX_READ_LEN (1024 * 1024)
    for (uint32_t i = 0; i < param.memorySize / MAX_READ_LEN; i++) {
        uint32_t readLen = read(fd, buf + i * MAX_READ_LEN, MAX_READ_LEN);
        if (readLen != MAX_READ_LEN) {
            std::cout << "gen random failed" << readLen << std::endl;
            exit(0);
        }
    }
    std::cout << "gen random suss" << std::endl;
    std::cout << "start test " << std::endl;
    ret = EcPerfTest(param, buf, blocks);

    free(buf);
    free(blocks);
    return 0;
}
