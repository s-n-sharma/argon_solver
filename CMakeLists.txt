cmake_minimum_required(VERSION 3.10)
project(MySolverProject CXX)

# Explicitly set OpenMP flags for GCC
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_compile_options(-O3)
# Tell CMake to find each SuiteSparse component individually
# These set variables like SuiteSparse_config_LIBRARIES, AMD_LIBRARIES, etc.
find_package(SuiteSparse_config REQUIRED)
message(STATUS "SuiteSparse_config include dirs: ${SuiteSparse_config_INCLUDE_DIRS}")
find_package(AMD REQUIRED)
find_package(COLAMD REQUIRED)
find_package(CHOLMOD REQUIRED)
find_package(SPQR REQUIRED)

# Find its other dependencies
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Add your C++ executable
add_executable(my_solver
    example.cpp
    suite_sparse_sparse_qr/qr_conflict_analyzer.cpp
)

# Tell the compiler where to find the SuiteSparse headers
target_include_directories(my_solver PRIVATE ${SuiteSparse_config_INCLUDE_DIRS}) # Use variable
target_link_libraries(my_solver PRIVATE SuiteSparse::SPQR)
# Link using the explicit library variables found by find_package
target_link_libraries(my_solver PRIVATE
    ${SPQR_LIBRARIES}                 # Use library variable
    ${CHOLMOD_LIBRARIES}              # Use library variable
    ${AMD_LIBRARIES}                  # Use library variable
    ${COLAMD_LIBRARIES}               # Use library variable
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}

    gomp                              # Link gomp directly for OpenMP
)

if (TARGET SuiteSparse::config)
    target_link_libraries(my_solver PRIVATE SuiteSparse::config)
elseif (TARGET SuiteSparse::SuiteSparse_config)
    target_link_libraries(my_solver PRIVATE SuiteSparse::SuiteSparse_config)
elseif (DEFINED SuiteSparse_config_LIBRARY AND SuiteSparse_config_LIBRARY)
    target_link_libraries(my_solver PRIVATE ${SuiteSparse_config_LIBRARY})
elseif (DEFINED SuiteSparse_config_LIBRARIES AND SuiteSparse_config_LIBRARIES)
    target_link_libraries(my_solver PRIVATE ${SuiteSparse_config_LIBRARIES})
else()
    get_filename_component(_ss_cfg_cmake_dir "${SuiteSparse_config_DIR}" ABSOLUTE)
    get_filename_component(_ss_cfg_libdir "${_ss_cfg_cmake_dir}" DIRECTORY) # .../lib64/cmake -> .../lib64
    get_filename_component(_ss_cfg_libdir "${_ss_cfg_libdir}" DIRECTORY)
    find_library(SuiteSparse_config_LIBRARY_PATH
        NAMES suitesparseconfig
        PATHS "${_ss_cfg_libdir}"
        PATH_SUFFIXES "" lib lib64
        NO_DEFAULT_PATH
    )
    if (NOT SuiteSparse_config_LIBRARY_PATH)
        find_library(SuiteSparse_config_LIBRARY_PATH NAMES suitesparseconfig)
    endif()
    if (SuiteSparse_config_LIBRARY_PATH)
        message(STATUS "Linking suitesparseconfig from ${SuiteSparse_config_LIBRARY_PATH}")
        target_link_libraries(my_solver PRIVATE ${SuiteSparse_config_LIBRARY_PATH})
    else()
        message(FATAL_ERROR "Unable to locate libsuitesparseconfig; set SuiteSparse_config_DIR to the CMake package directory")
    endif()
endif()

# Additional benchmark target
add_executable(qr_conflict_benchmark
    suite_sparse_sparse_qr/benchmark_conflict_detection.cpp
    suite_sparse_sparse_qr/qr_conflict_analyzer.cpp
)
target_include_directories(qr_conflict_benchmark PRIVATE ${SuiteSparse_config_INCLUDE_DIRS})
target_link_libraries(qr_conflict_benchmark PRIVATE
    SuiteSparse::SPQR
    ${SPQR_LIBRARIES}
    ${CHOLMOD_LIBRARIES}
    ${AMD_LIBRARIES}
    ${COLAMD_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    gomp
)

if (TARGET SuiteSparse::config)
    target_link_libraries(qr_conflict_benchmark PRIVATE SuiteSparse::config)
elseif (TARGET SuiteSparse::SuiteSparse_config)
    target_link_libraries(qr_conflict_benchmark PRIVATE SuiteSparse::SuiteSparse_config)
elseif (DEFINED SuiteSparse_config_LIBRARY AND SuiteSparse_config_LIBRARY)
    target_link_libraries(qr_conflict_benchmark PRIVATE ${SuiteSparse_config_LIBRARY})
elseif (DEFINED SuiteSparse_config_LIBRARIES AND SuiteSparse_config_LIBRARIES)
    target_link_libraries(qr_conflict_benchmark PRIVATE ${SuiteSparse_config_LIBRARIES})
else()
    get_filename_component(_ss_cfg_cmake_dir "${SuiteSparse_config_DIR}" ABSOLUTE)
    get_filename_component(_ss_cfg_libdir "${_ss_cfg_cmake_dir}" DIRECTORY)
    get_filename_component(_ss_cfg_libdir "${_ss_cfg_libdir}" DIRECTORY)
    find_library(SuiteSparse_config_LIBRARY_PATH
        NAMES suitesparseconfig
        PATHS "${_ss_cfg_libdir}"
        PATH_SUFFIXES "" lib lib64
        NO_DEFAULT_PATH
    )
    if (NOT SuiteSparse_config_LIBRARY_PATH)
        find_library(SuiteSparse_config_LIBRARY_PATH NAMES suitesparseconfig)
    endif()
    if (SuiteSparse_config_LIBRARY_PATH)
        message(STATUS "Linking suitesparseconfig from ${SuiteSparse_config_LIBRARY_PATH}")
        target_link_libraries(qr_conflict_benchmark PRIVATE ${SuiteSparse_config_LIBRARY_PATH})
    else()
        message(FATAL_ERROR "Unable to locate libsuitesparseconfig; set SuiteSparse_config_DIR to the CMake package directory")
    endif()
endif()

# Givens update benchmark target
add_executable(qr_givens_benchmark
    suite_sparse_sparse_qr/benchmark_givens_updates.cpp
    suite_sparse_sparse_qr/qr_conflict_analyzer.cpp
)
target_include_directories(qr_givens_benchmark PRIVATE ${SuiteSparse_config_INCLUDE_DIRS})
target_link_libraries(qr_givens_benchmark PRIVATE
    SuiteSparse::SPQR
    ${SPQR_LIBRARIES}
    ${CHOLMOD_LIBRARIES}
    ${AMD_LIBRARIES}
    ${COLAMD_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    gomp
)

if (TARGET SuiteSparse::config)
    target_link_libraries(qr_givens_benchmark PRIVATE SuiteSparse::config)
elseif (TARGET SuiteSparse::SuiteSparse_config)
    target_link_libraries(qr_givens_benchmark PRIVATE SuiteSparse::SuiteSparse_config)
elseif (DEFINED SuiteSparse_config_LIBRARY AND SuiteSparse_config_LIBRARY)
    target_link_libraries(qr_givens_benchmark PRIVATE ${SuiteSparse_config_LIBRARY})
elseif (DEFINED SuiteSparse_config_LIBRARIES AND SuiteSparse_config_LIBRARIES)
    target_link_libraries(qr_givens_benchmark PRIVATE ${SuiteSparse_config_LIBRARIES})
else()
    get_filename_component(_ss_cfg_cmake_dir "${SuiteSparse_config_DIR}" ABSOLUTE)
    get_filename_component(_ss_cfg_libdir "${_ss_cfg_cmake_dir}" DIRECTORY)
    get_filename_component(_ss_cfg_libdir "${_ss_cfg_libdir}" DIRECTORY)
    find_library(SuiteSparse_config_LIBRARY_PATH
        NAMES suitesparseconfig
        PATHS "${_ss_cfg_libdir}"
        PATH_SUFFIXES "" lib lib64
        NO_DEFAULT_PATH
    )
    if (NOT SuiteSparse_config_LIBRARY_PATH)
        find_library(SuiteSparse_config_LIBRARY_PATH NAMES suitesparseconfig)
    endif()
    if (SuiteSparse_config_LIBRARY_PATH)
        message(STATUS "Linking suitesparseconfig from ${SuiteSparse_config_LIBRARY_PATH}")
        target_link_libraries(qr_givens_benchmark PRIVATE ${SuiteSparse_config_LIBRARY_PATH})
    else()
        message(FATAL_ERROR "Unable to locate libsuitesparseconfig; set SuiteSparse_config_DIR to the CMake package directory")
    endif()
endif()

# Dynamic update benchmark target
add_executable(dynamic_qr_benchmark
    suite_sparse_sparse_qr/main.cpp
    suite_sparse_sparse_qr/hash_givens_conflict_analyzer.cpp
)
target_include_directories(dynamic_qr_benchmark PRIVATE ${SuiteSparse_config_INCLUDE_DIRS})
target_link_libraries(dynamic_qr_benchmark PRIVATE
    SuiteSparse::SPQR
    ${SPQR_LIBRARIES}
    ${CHOLMOD_LIBRARIES}
    ${AMD_LIBRARIES}
    ${COLAMD_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    gomp
)

if (TARGET SuiteSparse::config)
    target_link_libraries(dynamic_qr_benchmark PRIVATE SuiteSparse::config)
elseif (TARGET SuiteSparse::SuiteSparse_config)
    target_link_libraries(dynamic_qr_benchmark PRIVATE SuiteSparse::SuiteSparse_config)
elseif (DEFINED SuiteSparse_config_LIBRARY AND SuiteSparse_config_LIBRARY)
    target_link_libraries(dynamic_qr_benchmark PRIVATE ${SuiteSparse_config_LIBRARY})
elseif (DEFINED SuiteSparse_config_LIBRARIES AND SuiteSparse_config_LIBRARIES)
    target_link_libraries(dynamic_qr_benchmark PRIVATE ${SuiteSparse_config_LIBRARIES})
else()
    get_filename_component(_ss_cfg_cmake_dir "${SuiteSparse_config_DIR}" ABSOLUTE)
    get_filename_component(_ss_cfg_libdir "${_ss_cfg_cmake_dir}" DIRECTORY)
    get_filename_component(_ss_cfg_libdir "${_ss_cfg_libdir}" DIRECTORY)
    find_library(SuiteSparse_config_LIBRARY_PATH
        NAMES suitesparseconfig
        PATHS "${_ss_cfg_libdir}"
        PATH_SUFFIXES "" lib lib64
        NO_DEFAULT_PATH
    )
    if (NOT SuiteSparse_config_LIBRARY_PATH)
        find_library(SuiteSparse_config_LIBRARY_PATH NAMES suitesparseconfig)
    endif()
    if (SuiteSparse_config_LIBRARY_PATH)
        message(STATUS "Linking suitesparseconfig from ${SuiteSparse_config_LIBRARY_PATH}")
        target_link_libraries(dynamic_qr_benchmark PRIVATE ${SuiteSparse_config_LIBRARY_PATH})
    else()
        message(FATAL_ERROR "Unable to locate libsuitesparseconfig; set SuiteSparse_config_DIR to the CMake package directory")
    endif()
endif()