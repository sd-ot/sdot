cmake_minimum_required( VERSION 3.10 )
include( scripts/cmake/gtest.cmake )
project( sdot )

# code generation
add_executable( gen_plane_cut_procs
    scripts/gen_plane_cut_procs.cpp
    scripts/internal/GenPlaneCutProc.cpp
    scripts/internal/SimdGen.cpp
    scripts/internal/SimdGenOp.cpp
    scripts/internal/PlaneCutProcPlan.cpp
    src/sdot/support/OptParm.cpp
)

# common libraries
add_library( sdot_common
    src/sdot/Grids/internal/ZCoords.cpp
    src/sdot/support/VtkOutput.cpp
)


#
function( comp_profile scalar_type size_type arch )
    set( profile ${scalar_type}_${size_type}_${arch} )
    message( "Compiling for: " ${profile} )

    # cpu flags
    if ( arch EQUAL AVX512 )
        # -mavx512pf -mavx512er -mavx512ifma -mavx512vbmi -mpopcnt -mavx512vpopcntdq -mavx512vp2intersect -mavx5124fmaps -mavx512vnni -mavx5124vnniw -mfma4
        set( cpu_flags -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mavx -mavx2 -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 )
    endif()

    if ( arch EQUAL AVX2 )
        set( cpu_flags -mavx -mavx2 -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 )
    endif()

    if( arch EQUAL SSE2 )
        set( cpu_flags -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 )
    endif()

    if( arch EQUAL NOARCH )
        set( cpu_flags )
    endif()

    set( opt_flags -fno-stack-protector -march=native -O3 -g3 )

    # generation
    list( APPEND gen_commands echo ${CMAKE_BINARY_DIR}/gen_plane_cut_procs --scalar-type ${scalar_type} --size-type ${size_type} --arch ${arch} --out ${CMAKE_SOURCE_DIR}/src/sdot/ConvexPolyhedron/internal/ConvexPolyhedron2_plane_cut_lt8_${profile} --op  ${CMAKE_SOURCE_DIR}/src/sdot/ConvexPolyhedron/internal/ConvexPolyhedron2_plane_cut_lt8_${profile}.params && )
    set( gen_commands ${gen_commands} PARENT_SCOPE )

    # library 
    add_library( sdot_${profile}
        src/sdot/ConvexPolyhedron/ConvexPolyhedron2.cpp
    )
    target_compile_definitions( sdot_${profile} PRIVATE PROFILE=${profile} TF=${scalar_type} ST=${size_type} ARCH=${arch} )
    target_compile_options( sdot_${profile} PRIVATE ${opt_flags} )
    target_link_libraries( sdot_${profile} sdot_common )

    foreach( dim IN ITEMS 2 3 )
        add_library( sdot_${profile}_${dim}D
            src/sdot/Grids/ZGrid.cpp
        )
        target_compile_definitions( sdot_${profile}_${dim}D PRIVATE PROFILE=${profile} TF=${scalar_type} ST=${size_type} ARCH=${arch} DIM=${dim} )
        target_compile_options( sdot_${profile}_${dim}D PRIVATE ${opt_flags} )
        target_link_libraries( sdot_${profile}_${dim}D sdot_${profile} )
    endforeach()

    # test
    add_executable( test_${profile} tests/main.cpp
        tests/test_ConvexPolyhedron2.cpp 
        tests/test_Grid2.cpp
    )
    target_compile_definitions( test_${profile} PRIVATE PROFILE=${profile} TF=${scalar_type} ST=${size_type} ARCH=${arch} )
    target_link_libraries( test_${profile} sdot_${profile}_2D sdot_${profile}_3D gtest )
    add_test( test_${profile} test_${profile} )

    # bench
    add_executable( bench_${profile} tests/bench_ConvexPolyhedron2.cpp )
    target_compile_definitions( bench_${profile} PRIVATE PROFILE=${profile} TF=${scalar_type} ST=${size_type} ARCH=${arch} )
    target_compile_options( bench_${profile} PRIVATE ${opt_flags} )
    target_link_libraries( bench_${profile} sdot_${profile}_2D sdot_${profile}_3D )
endfunction()

function( comp_for_profiles )
    foreach( scalar_type IN LISTS scalar_types )
        foreach( size_type IN LISTS size_types )
            foreach( arch IN LISTS archs )
                comp_profile( ${scalar_type} ${size_type} ${arch} )
            endforeach()
        endforeach()
    endforeach()

    add_custom_target( gen ${gen_commands} echo )
endfunction()

# profiles
set( scalar_types FP64      )
set( size_types   U64       )
set( archs        SSE2 AVX2 )

comp_for_profiles()



