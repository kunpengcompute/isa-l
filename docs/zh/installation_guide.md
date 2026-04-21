# 编译安装 ISA-L

## 下载源码

直接拉取优化分支源码。

```bash
git clone -b dev-isal-2.31-for-arm https://gitcode.com/boostkit/isa-l.git
```
或者从github下载源码后合入优化补丁：

```bash
git clone -b v2.31 https://github.com/boostkit/isa-l.git
cd isa-l
wget https://gitcode.com/boostkit/isa-l/blob/master/arm-for-ec-crc32c-optimization.patch
patch -p1 < arm-for-ec-crc32c-optimization.patch
```

## 编译安装

```bash
./autogen.sh
./configure --prefix=/usr --libdir=/usr/lib64 --enable-crc32c-dispatcher=cache_hit
make -j
make install
```

### 关于 `--enable-crc32c-dispatcher` 配置的说明

- 使用 `--enable-crc32c-dispatcher=cache_hit` 可启用对缓存命中友好的 CRC32C 计算。
- 使用 `--enable-crc32c-dispatcher=cache_miss` 可启用对缓存未全命中友好的 CRC32C 计算。
- 默认为缓存未全命中友好的 CRC32C 计算。