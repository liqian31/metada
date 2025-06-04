#include "parallel_factory.hpp"

#include <stdexcept>

#include "backends/mpi/parallel/mpi_parallel.hpp"

namespace metada {
namespace core {

std::shared_ptr<ParallelInterface> ParallelFactory::create(
    const std::string& backend_type) {
  if (backend_type == "mpi") {
    return std::make_shared<mpi::MPIParallel>();
  }

  throw std::runtime_error("Unsupported parallel backend type: " +
                           backend_type);
}

}  // namespace core
}  // namespace metada