#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi_parallel.hpp"

namespace metada {
namespace mpi {
namespace test {

class MPICommunicatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      int argc = 0;
      char** argv = nullptr;
      MPI_Init(&argc, &argv);
    }
  }

  void TearDown() override {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
};

TEST_F(MPICommunicatorTest, WorldCommunicator) {
  MPICommunicator comm(MPI_COMM_WORLD);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  EXPECT_TRUE(comm.isValid());
  EXPECT_EQ(comm.getRank(), world_rank);
  EXPECT_EQ(comm.getSize(), world_size);
}

TEST_F(MPICommunicatorTest, NullCommunicator) {
  MPICommunicator comm(MPI_COMM_NULL);

  EXPECT_FALSE(comm.isValid());
  EXPECT_EQ(comm.getRank(), -1);
  EXPECT_EQ(comm.getSize(), 0);
}

TEST_F(MPICommunicatorTest, CustomCommunicator) {
  MPI_Comm custom_comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &custom_comm);

  MPICommunicator comm(custom_comm);

  int custom_rank, custom_size;
  MPI_Comm_rank(custom_comm, &custom_rank);
  MPI_Comm_size(custom_comm, &custom_size);

  EXPECT_TRUE(comm.isValid());
  EXPECT_EQ(comm.getRank(), custom_rank);
  EXPECT_EQ(comm.getSize(), custom_size);

  MPI_Comm_free(&custom_comm);
}

}  // namespace test
}  // namespace mpi
}  // namespace metada