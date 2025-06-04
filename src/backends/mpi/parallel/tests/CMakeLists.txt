add_executable(mpi_parallel_tests
    mpi_communicator_test.cpp
    mpi_task_manager_test.cpp
    mpi_parallel_test.cpp
)

target_link_libraries(mpi_parallel_tests
    PRIVATE
        metada::mpi_parallel
        GTest::gtest_main
        GTest::gmock
)

# Add test to CTest
add_test(NAME mpi_parallel_tests COMMAND mpi_parallel_tests)

# Set test properties
set_tests_properties(mpi_parallel_tests PROPERTIES
    ENVIRONMENT "OMPI_COMM_WORLD_SIZE=4"
    TIMEOUT 30
) 