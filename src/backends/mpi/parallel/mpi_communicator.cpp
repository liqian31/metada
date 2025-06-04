#include <stdexcept>

#include "mpi_parallel.hpp"

namespace metada {
namespace mpi {

MPICommunicator::MPICommunicator(MPI_Comm comm) : comm_(comm) {
  if (comm_ != MPI_COMM_NULL) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  } else {
    rank_ = -1;
    size_ = 0;
  }
}

MPICommunicator::~MPICommunicator() {
  // Note: We don't free the communicator here as it might be shared
  // The MPIParallel class is responsible for freeing communicators
}

int MPICommunicator::getRank() const {
  return rank_;
}

int MPICommunicator::getSize() const {
  return size_;
}

bool MPICommunicator::isValid() const {
  return comm_ != MPI_COMM_NULL;
}

}  // namespace mpi
}  // namespace metada