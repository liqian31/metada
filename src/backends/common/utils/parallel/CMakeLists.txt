add_library(parallel_adapter
    parallel_adapter.cpp
    parallel_adapter.hpp
)

target_include_directories(parallel_adapter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(parallel_adapter
    PUBLIC
        MPI::MPI_CXX
) 