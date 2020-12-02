#include <asimd/SimdSize.h>
#include "P.h"

//// nsmake cpp_flag -march=native

int main() {
    // InstructionSet
    using Is = asimd::InstructionSet::Native;
    P( Is::Has<asimd::InstructionSet::Features::SSE2>::value );
    P( Is::Has<asimd::InstructionSet::Features::AVX2>::value );
    P( Is::Has<asimd::InstructionSet::Features::AVX512>::value );
    P( Is::name() );

    //

}

