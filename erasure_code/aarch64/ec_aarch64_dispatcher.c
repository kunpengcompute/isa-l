/**************************************************************
  Copyright (c) 2019 Huawei Technologies Co., Ltd.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Huawei Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************/
#include <aarch64_multibinary.h>
#include <stddef.h>

DEFINE_INTERFACE_DISPATCHER(gf_vect_dot_prod)
{
#if defined(__linux__)
        unsigned long auxval = getauxval(AT_HWCAP);

        if (auxval & HWCAP_SVE)
                return PROVIDER_INFO(gf_vect_dot_prod_sve);
        if (auxval & HWCAP_ASIMD)
                return PROVIDER_INFO(gf_vect_dot_prod_neon);
#elif defined(__APPLE__)
        if (sysctlEnabled(SYSCTL_SVE_KEY))
                return PROVIDER_INFO(gf_vect_dot_prod_sve);
        return PROVIDER_INFO(gf_vect_dot_prod_neon);
#endif
        return PROVIDER_BASIC(gf_vect_dot_prod);
}

DEFINE_INTERFACE_DISPATCHER(gf_vect_mad)
{
#if defined(__linux__)
        unsigned long auxval = getauxval(AT_HWCAP);

        if (auxval & HWCAP_SVE)
                return PROVIDER_INFO(gf_vect_mad_sve);
        if (auxval & HWCAP_ASIMD)
                return PROVIDER_INFO(gf_vect_mad_neon);
#elif defined(__APPLE__)
        if (sysctlEnabled(SYSCTL_SVE_KEY))
                return PROVIDER_INFO(gf_vect_mad_sve);
        return PROVIDER_INFO(gf_vect_mad_neon);
#endif
        return PROVIDER_BASIC(gf_vect_mad);
}

static void *
select_native_encoder(void)
{
#if defined(__linux__)
        unsigned long auxval = getauxval(AT_HWCAP);
        unsigned long auxval2 = getauxval(AT_HWCAP2);

        if (auxval & HWCAP_SVE) {
                if (auxval2 & HWCAP2_SVE2)
                        return PROVIDER_INFO(ec_encode_data_sve2);

                return PROVIDER_INFO(ec_encode_data_sve);
        }

        if (auxval & HWCAP_ASIMD)
                return PROVIDER_INFO(ec_encode_data_neon);

#elif defined(__APPLE__)
        if (sysctlEnabled(SYSCTL_SVE_KEY))
                return PROVIDER_INFO(ec_encode_data_sve);

        return PROVIDER_INFO(ec_encode_data_neon);
#endif

        return PROVIDER_BASIC(ec_encode_data);
}

static int
ring_cpu_supported(void)
{
#if defined(__linux__) && defined(__aarch64__)
        return !!(getauxval(AT_HWCAP) & HWCAP_ASIMD);
#else
        return 0;
#endif
}
DEFINE_INTERFACE_DISPATCHER(ec_encode_data)
{
#ifdef ENABLE_RING_PROTOCOL
        if (ring_cpu_supported())
                return PROVIDER_INFO(ec_encode_data_ring);
#endif
        return select_native_encoder();
}

typedef void (*ec_encode_fn)(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                             unsigned char **coding);

void
ec_encode_data_native(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                      unsigned char **coding)
{
        static _Thread_local ec_encode_fn native_fn;

        if (native_fn == NULL)
                native_fn = (ec_encode_fn) select_native_encoder();

        native_fn(len, k, rows, g_tbls, data, coding);
}

DEFINE_INTERFACE_DISPATCHER(ec_encode_data_update)
{
#if defined(__linux__)
        unsigned long auxval = getauxval(AT_HWCAP);

        if (auxval & HWCAP_SVE)
                return PROVIDER_INFO(ec_encode_data_update_sve);
        if (auxval & HWCAP_ASIMD)
                return PROVIDER_INFO(ec_encode_data_update_neon);
#elif defined(__APPLE__)
        if (sysctlEnabled(SYSCTL_SVE_KEY))
                return PROVIDER_INFO(ec_encode_data_update_sve);
        return PROVIDER_INFO(ec_encode_data_update_neon);
#endif
        return PROVIDER_BASIC(ec_encode_data_update);
}

DEFINE_INTERFACE_DISPATCHER(gf_vect_mul)
{
#if defined(__linux__)
        unsigned long auxval = getauxval(AT_HWCAP);

        if (auxval & HWCAP_SVE)
                return PROVIDER_INFO(gf_vect_mul_sve);
        if (auxval & HWCAP_ASIMD)
                return PROVIDER_INFO(gf_vect_mul_neon);
#elif defined(__APPLE__)
        if (sysctlEnabled(SYSCTL_SVE_KEY))
                return PROVIDER_INFO(gf_vect_mul_sve);
        return PROVIDER_INFO(gf_vect_mul_neon);
#endif
        return PROVIDER_BASIC(gf_vect_mul);
}

DEFINE_INTERFACE_DISPATCHER(ec_init_tables) { return PROVIDER_BASIC(ec_init_tables); }
