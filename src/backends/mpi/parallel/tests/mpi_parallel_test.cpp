#include "mpi_parallel.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

namespace metada {
namespace mpi {
namespace test {

class MPIParallelTest : public ::testing::Test {
 protected:
  void SetUp() override { parallel_ = std::make_unique<MPIParallel>(); }

  void TearDown() override { parallel_.reset(); }

  std::unique_ptr<MPIParallel> parallel_;
};

TEST_F(MPIParallelTest, Initialization) {
  int argc = 0;
  char** argv = nullptr;
  parallel_->initialize(argc, argv);

  auto world_comm = parallel_->getWorldCommunicator();
  EXPECT_TRUE(world_comm->isValid());

  auto task_manager = parallel_->getTaskManager();
  EXPECT_NE(task_manager, nullptr);
}

TEST_F(MPIParallelTest, CommunicatorCreation) {
  int argc = 0;
  char** argv = nullptr;
  parallel_->initialize(argc, argv);

  // Get task manager and distribute tasks
  auto task_manager = parallel_->getTaskManager();
  task_manager->distributeTasks(2);  // Create 2 tasks

  // Create communicators
  parallel_->createModelComm();
  parallel_->createFilterComm();
  parallel_->createCoupleComm();

  // Check model communicator
  auto model_comm = parallel_->getModelCommunicator();
  EXPECT_TRUE(model_comm->isValid());
  EXPECT_EQ(model_comm->getSize(), task_manager->getLocalSize());

  // Check filter communicator (only task 0 should have it)
  auto filter_comm = parallel_->getFilterCommunicator();
  if (task_manager->getTaskId() == 0) {
    EXPECT_TRUE(filter_comm->isValid());
  } else {
    EXPECT_FALSE(filter_comm->isValid());
  }

  // Check couple communicator
  auto couple_comm = parallel_->getCoupleCommunicator();
  EXPECT_TRUE(couple_comm->isValid());
  EXPECT_EQ(couple_comm->getSize(), task_manager->getLocalSize());
}

TEST_F(MPIParallelTest, MultipleInitialization) {
  int argc = 0;
  char** argv = nullptr;

  // First initialization
  parallel_->initialize(argc, argv);
  auto world_comm1 = parallel_->getWorldCommunicator();

  // Second initialization should not affect existing communicators
  parallel_->initialize(argc, argv);
  auto world_comm2 = parallel_->getWorldCommunicator();

  EXPECT_EQ(world_comm1->getRank(), world_comm2->getRank());
  EXPECT_EQ(world_comm1->getSize(), world_comm2->getSize());
}

TEST_F(MPIParallelTest, Finalization) {
  int argc = 0;
  char** argv = nullptr;
  parallel_->initialize(argc, argv);

  // First finalization
  parallel_->finalize();

  // Second finalization should not throw
  EXPECT_NO_THROW(parallel_->finalize());
}

}  // namespace test
}  // namespace mpi
}  // namespace metada