#pragma once

#include <memory>
#include <string>
#include <vector>

namespace metada {
namespace core {

// Forward declarations
class Communicator;
class TaskManager;

/**
 * @brief Interface for parallel processing functionality
 *
 * This interface abstracts away the specific parallel implementation (MPI,
 * etc.) and provides a clean API for parallel operations.
 */
class ParallelInterface {
 public:
  virtual ~ParallelInterface() = default;

  // Initialize parallel environment
  virtual void initialize(int argc, char** argv) = 0;

  // Get communicator information
  virtual std::shared_ptr<Communicator> getWorldCommunicator() const = 0;
  virtual std::shared_ptr<Communicator> getModelCommunicator() const = 0;
  virtual std::shared_ptr<Communicator> getFilterCommunicator() const = 0;
  virtual std::shared_ptr<Communicator> getCoupleCommunicator() const = 0;

  // Get task manager
  virtual std::shared_ptr<TaskManager> getTaskManager() const = 0;

  // Finalize parallel environment
  virtual void finalize() = 0;
};

/**
 * @brief Interface for communicator operations
 */
class Communicator {
 public:
  virtual ~Communicator() = default;

  virtual int getRank() const = 0;
  virtual int getSize() const = 0;
  virtual bool isValid() const = 0;

  // Add more communicator operations as needed
  // e.g., broadcast, scatter, gather, etc.
};

/**
 * @brief Interface for task management
 */
class TaskManager {
 public:
  virtual ~TaskManager() = default;

  virtual void distributeTasks(int num_tasks) = 0;
  virtual int getTaskId() const = 0;
  virtual int getLocalSize() const = 0;
  virtual const std::vector<int>& getLocalSizes() const = 0;
};

}  // namespace core
}  // namespace metada