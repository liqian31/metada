#include <algorithm>
#include <stdexcept>

#include "mpi_parallel.hpp"

namespace metada {
namespace mpi {

MPITaskManager::MPITaskManager(MPI_Comm world_comm)
    : world_comm_(world_comm), task_id_(0), local_size_(0) {
  if (world_comm_ == MPI_COMM_NULL) {
    throw std::runtime_error("Invalid world communicator");
  }
}

MPITaskManager::~MPITaskManager() = default;

void MPITaskManager::distributeTasks(int num_tasks) {
  if (num_tasks <= 0) {
    throw std::runtime_error("Invalid number of tasks");
  }

  int world_size;
  MPI_Comm_size(world_comm_, &world_size);

  if (num_tasks > world_size) {
    throw std::runtime_error("Number of tasks cannot exceed world size");
  }

  // Calculate local sizes
  local_sizes_.resize(num_tasks);
  int base_size = world_size / num_tasks;
  int remainder = world_size % num_tasks;

  for (int i = 0; i < num_tasks; ++i) {
    local_sizes_[i] = base_size + (i < remainder ? 1 : 0);
  }

  // Determine task_id for this process
  int world_rank;
  MPI_Comm_rank(world_comm_, &world_rank);

  int offset = 0;
  for (int i = 0; i < num_tasks; ++i) {
    if (world_rank < offset + local_sizes_[i]) {
      task_id_ = i;
      local_size_ = local_sizes_[i];
      break;
    }
    offset += local_sizes_[i];
  }
}

int MPITaskManager::getTaskId() const {
  return task_id_;
}

int MPITaskManager::getLocalSize() const {
  return local_size_;
}

const std::vector<int>& MPITaskManager::getLocalSizes() const {
  return local_sizes_;
}

}  // namespace mpi
}  // namespace metada