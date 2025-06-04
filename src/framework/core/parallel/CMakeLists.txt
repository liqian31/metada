add_library(core_parallel INTERFACE)
add_library(metada::core_parallel ALIAS core_parallel)

target_sources(core_parallel
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/parallel_interface.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/parallel_factory.hpp
)

target_include_directories(core_parallel
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Ensure C++20 support
target_compile_features(core_parallel INTERFACE cxx_std_20)

AddFormatTarget(core_parallel ${CMAKE_CURRENT_SOURCE_DIR}) 