# User Guide

Install ISA-L by referring to [Installation Guide](installation_guide.md).

## Performance Tests

### CRC32C Test

1. Go to the `crc` directory.

    ```bash
    cd crc
    ```

2. Compile the test tool.

    ```bash
    gcc -o crc32_iscsi_perf crc32_iscsi_perf.c -I ../include -lisal
    ```

3. Run the test program.

    ```bash
    ./crc32_iscsi_perf
    ```

#### Description of the `crc32_iscsi_perf.c` Test Tool

By default, this tool supports only 8 KB data tests. If you want to test other block sizes, you need to change the value of the `TEST_LEN` macro in the source code to the desired size (for example, change it to `4 * 1024` to use 4 KB block size).

### EC Test

1. Go to the `erasure_code` directory.

    ```bash
    cd erasure_code
    ```

2. Compile the test tool.

    ```bash
    gcc -o erasure_code_perf erasure_code_perf.c -I ../include -lisal
    ```

3. Run the test tool.

    ```bash
    ./erasure_code_perf -k 4 -p 2 -e 1
    ```

#### Parameters

| Parameter    | Description          |
|--------|--------------|
| `-k 4` | The number of data blocks is 4.    |
| `-p 2` | The number of parity blocks is 2.    |
| `-e 1` | Emulates erasure or loss of one block.|

## Change History

| Date | Description      |
|-------|----------|
| 2026-06-30 | This is the first official release.|
