#include "mpi_parallel.hpp"

#include <stdexcept>

namespace metada {
namespace mpi {

MPIParallel::MPIParallel()
    : world_comm_(nullptr),
      model_comm_(nullptr),
      filter_comm_(nullptr),
      couple_comm_(nullptr),
      task_manager_(nullptr) {}

MPIParallel::~MPIParallel() {
  // Clean up communicators
  if (model_comm_) {
    MPI_Comm_free(&model_comm_->getMPIComm());
  }
  if (filter_comm_) {
    MPI_Comm_free(&filter_comm_->getMPIComm());
  }
  if (couple_comm_) {
    MPI_Comm_free(&couple_comm_->getMPIComm());
  }
}

void MPIParallel::initialize(int argc, char** argv) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(&argc, &argv);
  }

  // Create world communicator
  world_comm_ = std::make_shared<MPICommunicator>(MPI_COMM_WORLD);

  // Create task manager
  task_manager_ = std::make_shared<MPITaskManager>(MPI_COMM_WORLD);
}

std::shared_ptr<core::Communicator> MPIParallel::getWorldCommunicator() const {
  return world_comm_;
}

std::shared_ptr<core::Communicator> MPIParallel::getModelCommunicator() const {
  return model_comm_;
}

std::shared_ptr<core::Communicator> MPIParallel::getFilterCommunicator() const {
  return filter_comm_;
}

std::shared_ptr<core::Communicator> MPIParallel::getCoupleCommunicator() const {
  return couple_comm_;
}

std::shared_ptr<core::TaskManager> MPIParallel::getTaskManager() const {
  return task_manager_;
}

void MPIParallel::createModelComm() {
  if (!world_comm_ || !task_manager_) {
    throw std::runtime_error(
        "World communicator or task manager not initialized");
  }

  int color = task_manager_->getTaskId();
  MPI_Comm mpi_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_comm_->getRank(), &mpi_comm);
  model_comm_ = std::make_shared<MPICommunicator>(mpi_comm);
}

void MPIParallel::createFilterComm() {
  if (!world_comm_ || !task_manager_) {
    throw std::runtime_error(
        "World communicator or task manager not initialized");
  }

  int color = (task_manager_->getTaskId() == 0) ? 0 : MPI_UNDEFINED;
  MPI_Comm mpi_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_comm_->getRank(), &mpi_comm);
  if (mpi_comm != MPI_COMM_NULL) {
    filter_comm_ = std::make_shared<MPICommunicator>(mpi_comm);
  }
}

void MPIParallel::createCoupleComm() {
  if (!world_comm_ || !task_manager_) {
    throw std::runtime_error(
        "World communicator or task manager not initialized");
  }

  int color = task_manager_->getTaskId();
  MPI_Comm mpi_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_comm_->getRank(), &mpi_comm);
  couple_comm_ = std::make_shared<MPICommunicator>(mpi_comm);
}

void MPIParallel::finalize() {
  int finalized;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
}

}  // namespace mpi
}  // namespace metada