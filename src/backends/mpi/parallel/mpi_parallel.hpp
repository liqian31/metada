#pragma once

#include <mpi.h>

#include "framework/core/parallel/parallel_interface.hpp"

namespace metada {
namespace mpi {

class MPICommunicator : public core::Communicator {
 public:
  explicit MPICommunicator(MPI_Comm comm);
  ~MPICommunicator() override;

  int getRank() const override;
  int getSize() const override;
  bool isValid() const override;
  MPI_Comm getMPIComm() const { return comm_; }

 private:
  MPI_Comm comm_;
  int rank_;
  int size_;
};

class MPITaskManager : public core::TaskManager {
 public:
  MPITaskManager(MPI_Comm world_comm);
  ~MPITaskManager() override;

  void distributeTasks(int num_tasks) override;
  int getTaskId() const override;
  int getLocalSize() const override;
  const std::vector<int>& getLocalSizes() const override;

 private:
  MPI_Comm world_comm_;
  int task_id_;
  int local_size_;
  std::vector<int> local_sizes_;
};

class MPIParallel : public core::ParallelInterface {
 public:
  MPIParallel();
  ~MPIParallel() override;

  void initialize(int argc, char** argv) override;

  std::shared_ptr<core::Communicator> getWorldCommunicator() const override;
  std::shared_ptr<core::Communicator> getModelCommunicator() const override;
  std::shared_ptr<core::Communicator> getFilterCommunicator() const override;
  std::shared_ptr<core::Communicator> getCoupleCommunicator() const override;

  std::shared_ptr<core::TaskManager> getTaskManager() const override;

  void finalize() override;

 private:
  void createModelComm();
  void createFilterComm();
  void createCoupleComm();

  std::shared_ptr<MPICommunicator> world_comm_;
  std::shared_ptr<MPICommunicator> model_comm_;
  std::shared_ptr<MPICommunicator> filter_comm_;
  std::shared_ptr<MPICommunicator> couple_comm_;
  std::shared_ptr<MPITaskManager> task_manager_;
};

}  // namespace mpi
}  // namespace metada