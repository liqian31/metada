#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi_parallel.hpp"

namespace metada {
namespace mpi {
namespace test {

class MPITaskManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      int argc = 0;
      char** argv = nullptr;
      MPI_Init(&argc, &argv);
    }
    task_manager_ = std::make_unique<MPITaskManager>(MPI_COMM_WORLD);
  }

  void TearDown() override {
    task_manager_.reset();
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }

  std::unique_ptr<MPITaskManager> task_manager_;
};

TEST_F(MPITaskManagerTest, InvalidTaskCount) {
  EXPECT_THROW(task_manager_->distributeTasks(0), std::runtime_error);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  EXPECT_THROW(task_manager_->distributeTasks(world_size + 1),
               std::runtime_error);
}

TEST_F(MPITaskManagerTest, SingleTask) {
  task_manager_->distributeTasks(1);

  EXPECT_EQ(task_manager_->getTaskId(), 0);
  EXPECT_EQ(task_manager_->getLocalSize(), task_manager_->getLocalSizes()[0]);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  EXPECT_EQ(task_manager_->getLocalSize(), world_size);
}

TEST_F(MPITaskManagerTest, MultipleTasks) {
  const int num_tasks = 2;
  task_manager_->distributeTasks(num_tasks);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Verify task distribution
  const auto& local_sizes = task_manager_->getLocalSizes();
  EXPECT_EQ(local_sizes.size(), num_tasks);

  int total_size = 0;
  for (int size : local_sizes) {
    total_size += size;
  }
  EXPECT_EQ(total_size, world_size);

  // Verify task ID is valid
  int task_id = task_manager_->getTaskId();
  EXPECT_GE(task_id, 0);
  EXPECT_LT(task_id, num_tasks);

  // Verify local size matches the task's size
  EXPECT_EQ(task_manager_->getLocalSize(), local_sizes[task_id]);
}

TEST_F(MPITaskManagerTest, EvenTaskDistribution) {
  const int num_tasks = 4;
  task_manager_->distributeTasks(num_tasks);

  const auto& local_sizes = task_manager_->getLocalSizes();
  EXPECT_EQ(local_sizes.size(), num_tasks);

  // Check that sizes are as even as possible
  int max_diff = 0;
  for (int i = 1; i < num_tasks; ++i) {
    max_diff =
        std::max(max_diff, std::abs(local_sizes[i] - local_sizes[i - 1]));
  }
  EXPECT_LE(max_diff, 1);  // Sizes should differ by at most 1
}

}  // namespace test
}  // namespace mpi
}  // namespace metada