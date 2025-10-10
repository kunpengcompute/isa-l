/**
 1) The build command: g++ crc32c_perf_test.cpp -O3 -lisal -o crc32c_perf_test  
 2) The test command: ./crc32c_perf_test -c 10000 -m 1G -b 4k
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
using namespace std;
#ifdef __cplusplus
    extern "C" {
#endif
unsigned int crc32_iscsi(unsigned char *buffer, int len, unsigned int init_crc);
#ifdef __cplusplus
    }
#endif

struct option g_longOptions[] = {
    {"memory", required_argument, NULL, 'm'},
    {"block", required_argument, NULL, 'b'},
    {"count", required_argument, NULL, 'c'},
    {"help", no_argument, NULL, 'h'}
};

void Usage() {
    std::cout << "    --block     -b     blocksize [1~]" << std::endl;
    std::cout << "    --count     -c     running count [~]" << std::endl;
    std::cout << "    --memory    -m     memory size eg. 1GB 500MB" << std::endl;
    std::cout << "    --help      -h     help" << std::endl;
}

using Crc32cPerfParam = struct Crc32cPerfParam {
    uint64_t memorySize;
    int blockSize;
    uint32_t count;
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

int ParserArgument(Crc32cPerfParam &param, int argc, char **argv) {
    int opt = 0;
    uint64_t blockSize;
    uint64_t memory;
    int count;
    while ((opt = getopt_long(argc, argv, "hb:c:m:", g_longOptions, NULL)) != -1) {
        switch (opt) {
            case 'b':
                blockSize = str2byte(optarg);
                if (blockSize > 0) {
                    param.blockSize = blockSize;
                } else {
                    Usage();
                    exit(0);
                }
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

int Crc32cPerfTest(const Crc32cPerfParam param, uint8_t *buf, uint8_t **block) {
    uint32_t crc = 0;
    uint32_t i = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (i < param.count) {
        uint32_t n = rand() % (param.memorySize / param.blockSize);
        crc = crc32_iscsi(block[n], param.blockSize, crc);
        i++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "crc32_iscsi cost time " << (end - start).count() / param.count << "ns" << std::endl;
    std::cout << "crc32_iscsi cost bandwidth " << 1.0 * param.blockSize * param.count / (std::chrono::duration_cast<chrono::microseconds>(end - start)).count() << "MB/s" << std::endl;
    std::cout << "crc32_iscsi crc " << crc << std::endl;
    return 0;
}

int main(int argc, char **argv) {

    Crc32cPerfParam param = {3 * 1024 * 1024 * 1024, 4096, 10000};
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
    ret = Crc32cPerfTest(param, buf, blocks);

    free(buf);
    free(blocks);
    return 0;
}
