# 安装指南

## 环境要求

在使用ISA-L EC特性之前，根据使用优化版本不同，对处理器和指令有不同的要求，详见下表。

| 优化版本       | 依赖指令                    | 处理器        |
| -------------- | --------------------------- | ------------- |
| CRC32C六路并行 | CRC32                       | 鲲鹏920新型号处理器 |
| 标向量混合     | CRC32、SVPMULL（SVE指令集） | 鲲鹏950处理器 |

## 下载源码

- 直接拉取优化分支源码。

```bash
git clone -b dev-isal-2.31-for-arm https://gitcode.com/boostkit/isa-l.git
```
- 或者从GitHub下载源码后合入优化补丁：

```bash
git clone -b v2.31 https://github.com/boostkit/isa-l.git
cd isa-l
wget https://gitcode.com/boostkit/isa-l/blob/master/arm-for-ec-crc32c-optimization.patch
patch -p1 < arm-for-ec-crc32c-optimization.patch
```

## 编译安装

```bash
./autogen.sh
./configure --prefix=/usr --libdir=/usr/lib64 \
    --enable-crc32c-dispatcher=cache_hit \
    --enable-ring-protocol \
    CFLAGS="-g -O3 -march=armv8.2-a+sha3"
make -j
make install
```

### 关于 `--enable-crc32c-dispatcher` 配置的说明

- 使用 `--enable-crc32c-dispatcher=cache_hit` 可启用对缓存命中友好的 CRC32C 计算。
- 使用 `--enable-crc32c-dispatcher=cache_miss` 可启用对缓存未全命中友好的 CRC32C 计算。
- 默认为缓存未全命中友好的 CRC32C 计算。

### 关于 `--enable-ring-protocol` 配置的说明

- 使用 `--enable-ring-protocol` 可启用环域优化算法，支持 `10+1`、`4+2`、`8+3` 三种数据块与校验块配比；其他配比沿用原有优化路径。
- 环域算法与原生算法的编解码矩阵不同。编码和解码时需使用配套算法，原生算法生成的校验块无法使用环域算法进行数据恢复。

### 关于 `CFLAGS` 配置的说明

上述编译示例使用鲲鹏 920 新型号处理器的配置。请根据目标处理器选择相应的编译选项：

| 目标处理器 | 编译选项 |
| ---------- | -------- |
| 鲲鹏 920 新型号处理器 | `CFLAGS="-g -O3 -march=armv8.2-a+sha3"` |
| 鲲鹏 950 处理器 | `CFLAGS="-g -O3 -march=armv9-a+sve2"` |

## 修订记录

| 发布日期  | 修改说明       |
|-------|----------|
| 2026-06-30 | 第一次正式发布。|
