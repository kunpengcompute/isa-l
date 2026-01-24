Intel(R) Intelligent Storage Acceleration Library
=================================================

![Continuous Integration](https://github.com/intel/isa-l/actions/workflows/ci.yml/badge.svg)
[![Package on conda-forge](https://img.shields.io/conda/v/conda-forge/isa-l.svg)](https://anaconda.org/conda-forge/isa-l)
[![Coverity Status](https://scan.coverity.com/projects/29480/badge.svg)](https://scan.coverity.com/projects/intel-isa-l)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/intel/isa-l/badge)](https://securityscorecards.dev/viewer/?uri=github.com/intel/isa-l)

ISA-L is a collection of optimized low-level functions targeting storage
applications.  ISA-L includes:
* Erasure codes - Fast block Reed-Solomon type erasure codes for any
  encode/decode matrix in GF(2^8).
* CRC - Fast implementations of cyclic redundancy check.  Six different
  polynomials supported.
  - iscsi32, ieee32, t10dif, ecma64, iso64, jones64, rocksoft64.
* Raid - calculate and operate on XOR and P+Q parity found in common RAID
  implementations.
* Compression - Fast deflate-compatible data compression.
* De-compression - Fast inflate-compatible data compression.
* igzip - A command line application like gzip, accelerated with ISA-L.

Also see:
* [ISA-L for updates](https://github.com/intel/isa-l).
* For crypto functions see [isa-l_crypto on github](https://github.com/intel/isa-l_crypto).
* The [github wiki](https://github.com/intel/isa-l/wiki) including a list of
  [distros/ports](https://github.com/intel/isa-l/wiki/Ports--Repos) offering binary packages
  as well as a list of [language bindings](https://github.com/intel/isa-l/wiki/Language-Bindings).
* [Contributing](CONTRIBUTING.md).
* [Security Policy](SECURITY.md).
* Docs on [units](doc/functions.md), [tests](doc/test.md), or [build details](doc/build.md).

Building ISA-L
--------------

### Prerequisites

* Make: GNU 'make' or 'nmake' (Windows).
* Optional: Building with autotools requires autoconf/automake/libtool packages.
* Optional: Manual generation requires help2man package.

x86_64:
* Assembler: nasm. Version 2.15 or later suggested (other versions of nasm and
  yasm may build but with limited function [support](doc/build.md)).
* Compiler: gcc, clang, icc or VC compiler.

aarch64:
* Assembler: gas v2.24 or later.
* Compiler: gcc v4.7 or later.

other:
* Compiler: Portable base functions are available that build with most C compilers.

### Autotools Update

In this branch, optimizations have been implemented for CRC32C and Erasure Code calculations on ARM architecture. The specific optimizations are as follows:

* CRC32C:

Based on the CPU's supported instruction set features and to more fully utilize all logical computing units, a new implementation scheme has been added that combines the SVPMULL instruction with scalar‑vector hybrid instructions.

Targeted optimizations have been performed on the original implementation for different user‑scenario requirements:

For cache‑unfriendly scenarios, the SVPMULL instruction is adopted along with more cache‑friendly prefetch optimizations.

For cache‑friendly scenarios, loop‑unrolling optimizations and hybrid scalar‑vector parallel implementation optimizations have been applied.

* Erasure Code:

Computational optimizations were applied for the ratios of x+1, x+2, x+3.

The compilation, installation, and enabling methods are largely consistent with the original process. Specific commands are as follows:

    ```bash
    ./autogen.sh
    ./configure --enable-crc32c-dispatcher=cache_hit
    make
    sudo make install
    ```

Note regarding the ./configure configuration:

Use --enable-crc32c-dispatcher=cache_hit for cache-hit-friendly CRC32C calculation.

Use --enable-crc32c-dispatcher=cache_miss for cache-miss-friendly CRC32C calculation.

Performance testing can be performed following these steps:
CRC32C Test：
Enter the crc directory
    ```bash
    cd crc
    ```

Compile the test tool
    ```bash
    gcc -o crc32_iscsi_perf crc32_iscsi_perf.c -I ../include -lisal
    ```
Run the test program
    ```bash
    ./crc32_iscsi_perf
    ```

Erasure Code Test:
Enter the erasure_code directory
    ```bash
    cd erasure_code
    ```

Compile the test tool
    ```bash
    gcc -o erasure_code_perf erasure_code_perf.c -I ../include -lisal
    ```
Run the test program
    ```bash
    ./erasure_code_perf -k 4 -p 2 -e 1
    ```

### Makefile
To use a standard makefile run:

    make -f Makefile.unx

### Windows
On Windows use nmake to build dll and static lib:

    nmake -f Makefile.nmake

or see [details on setting up environment here](doc/build.md).

### Other make targets
Other targets include:
* `make check` : create and run tests
* `make tests` : create additional unit tests
* `make perfs` : create included performance tests
* `make ex`    : build examples
* `make other` : build other utilities such as compression file tests
* `make doc`   : build API manual

DLL Injection Attack
--------------------

### Problem

The Windows OS has an insecure predefined search order and set of defaults when trying to locate a resource. If the resource location is not specified by the software, an attacker need only place a malicious version in one of the locations Windows will search, and it will be loaded instead. Although this weakness can occur with any resource, it is especially common with DLL files.

### Solutions

Applications using libisal DLL library may need to apply one of the solutions to prevent from DLL injection attack.

Two solutions are available:
- Using a Fully Qualified Path is the most secure way to load a DLL
- Signature verification of the DLL

### Resources and Solution Details

- Security remarks section of LoadLibraryEx documentation by Microsoft: <https://docs.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa#security-remarks>
- Microsoft Dynamic Link Library Security article: <https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-security>
- Hijack Execution Flow: DLL Search Order Hijacking: <https://attack.mitre.org/techniques/T1574/001>
- Hijack Execution Flow: DLL Side-Loading: <https://attack.mitre.org/techniques/T1574/002>
