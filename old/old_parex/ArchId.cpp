#include <cpu_features/cpu_features_macros.h>
#include "ArchId.h"
#include "TODO.h"

#if defined(CPU_FEATURES_ARCH_X86)
    #include <cpu_features/cpuinfo_x86.h>
#elif defined(CPU_FEATURES_ARCH_ARM)
    #include <cpu_features/cpuinfo_arm.h>
#elif defined(CPU_FEATURES_ARCH_AARCH64)
    #include <cpu_features/cpuinfo_aarch64.h>
#elif defined(CPU_FEATURES_ARCH_MIPS)
    #include <cpu_features/cpuinfo_mips.h>
#elif defined(CPU_FEATURES_ARCH_PPC)
    #include <cpu_features/cpuinfo_ppc.h>
#endif

ArchId::ArchId() {
    #if defined(CPU_FEATURES_ARCH_X86)
        cpu_features::X86Info ci = cpu_features::GetX86Info();
        if ( ci.features.avx512f )
            name = "avx512";
        else if ( ci.features.avx2 )
            name = "avx2";
        else if ( ci.features.sse2 )
            name = "sse2";
    #else
        TODO;
    #endif
}
