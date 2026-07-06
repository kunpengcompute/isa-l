# Installation Guide

## Environment Requirements

Before using the ISA-L EC feature, ensure that the processor and instruction set meet the requirements of the respective optimized version. The following table lists the requirements.

| Optimized Version      | Instruction Dependency                   | Processor       |
| -------------- | --------------------------- | ------------- |
| CRC32C six-way parallelism| CRC32                       | New Kunpeng 920 processor model|
| Mixed scalar-vector    | CRC32 and SVPMULL (SVE instruction set)| Kunpeng 950 processor|

## Downloading the Source Code

- Directly pull the source code of the optimized branch.

```bash
git clone -b dev-isal-2.31-for-arm https://gitcode.com/boostkit/isa-l.git
```

- Alternatively, download the source code from GitHub and apply the optimization patch.

```bash
git clone -b v2.31 https://github.com/boostkit/isa-l.git
cd isa-l
wget https://gitcode.com/boostkit/isa-l/blob/master/arm-for-ec-crc32c-optimization.patch
patch -p1 < arm-for-ec-crc32c-optimization.patch
```

## Compilation and Installation

```bash
./autogen.sh
./configure --prefix=/usr --libdir=/usr/lib64 --enable-crc32c-dispatcher=cache_hit
make -j
make install
```

### Description of `--enable-crc32c-dispatcher` Configuration

- Use `--enable-crc32c-dispatcher=cache_hit` to enable CRC32C calculation that is friendly to cache hits.
- Use `--enable-crc32c-dispatcher=cache_miss` to enable CRC32C calculation that is friendly to cache misses.
- By default, CRC32C calculation that is friendly to cache misses is used.

## Change History

| Date | Description      |
|-------|----------|
| 2026-06-30 | This is the first official release.|
