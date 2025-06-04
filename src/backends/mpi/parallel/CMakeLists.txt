add_library(mpi_parallel
    mpi_parallel.cpp
    mpi_communicator.cpp
    mpi_task_manager.cpp
)

add_library(metada::mpi_parallel ALIAS mpi_parallel)

target_sources(mpi_parallel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi_parallel.hpp
)

target_include_directories(mpi_parallel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(mpi_parallel
    PUBLIC
        metada::core_parallel
        MPI::MPI_CXX
)

# Ensure C++20 support
target_compile_features(mpi_parallel PUBLIC cxx_std_20)

AddFormatTarget(mpi_parallel ${CMAKE_CURRENT_SOURCE_DIR})

# Add tests if testing is enabled
if(BUILD_TESTING)
    add_subdirectory(tests)
endif() 