#pragma once

#include <memory>
#include <string>

#include "parallel_interface.hpp"

namespace metada {
namespace core {

/**
 * @brief Factory for creating parallel processing implementations
 */
class ParallelFactory {
 public:
  /**
   * @brief Create a parallel implementation based on the backend type
   *
   * @param backend_type The type of parallel backend to create (e.g., "mpi",
   * "openmp", etc.)
   * @return std::shared_ptr<ParallelInterface> The created parallel
   * implementation
   * @throws std::runtime_error if the backend type is not supported
   */
  static std::shared_ptr<ParallelInterface> create(
      const std::string& backend_type = "mpi");

 private:
  ParallelFactory() = delete;
};

}  // namespace core
}  // namespace metada