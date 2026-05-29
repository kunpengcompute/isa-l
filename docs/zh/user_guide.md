# 用户指南
请参见《[安装指南](installation_guide.md)》完成ISA-L的编译安装。
## 性能测试

### CRC32C测试

1. 进入crc目录。

    ```bash
    cd crc
    ```

2. 编译测试工具。

    ```bash
    gcc -o crc32_iscsi_perf crc32_iscsi_perf.c -I ../include -lisal
    ```

3. 运行测试程序。

    ```bash
    ./crc32_iscsi_perf
    ```

#### 关于`crc32_iscsi_perf.c`测试工具的说明

该工具默认只支持8K字节大小的数据测试，如想要测试其它块大小，需要修改源码中`TEST_LEN`宏，将其值改为想要测试的大小（如4K，则调整为：`4 * 1024`）。

### 纠删码测试

1. 进入erasure_code目录。

    ```bash
    cd erasure_code
    ```

2. 编译测试工具。

    ```bash
    gcc -o erasure_code_perf erasure_code_perf.c -I ../include -lisal
    ```

3. 运行测试工具。

    ```bash
    ./erasure_code_perf -k 4 -p 2 -e 1
    ```

#### 参数说明

| 参数     | 说明           |
|--------|--------------|
| `-k 4` | 数据块数量为 4     |
| `-p 2` | 校验块数量为 2     |
| `-e 1` | 模拟擦除/丢失 1 个块 |

## 修订记录

| 发布日期  | 修改说明       |
|-------|----------|
| 2026-06-30 | 第一次正式发布。|