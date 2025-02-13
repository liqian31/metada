#ifndef METADA_FRAMEWORK_REPR_ISTATE_HPP_
#define METADA_FRAMEWORK_REPR_ISTATE_HPP_

#include <string>
#include <vector>

#include "IConfig.hpp"

using metada::framework::tools::config::IConfig;

namespace metada {
namespace framework {
namespace repr {

/**
 * @brief Abstract interface for state implementations
 *
 * This interface defines the contract that all state implementations must
 * follow. It provides a unified API for handling state variables in scientific
 * models.
 */
class IState {
 public:
  virtual ~IState() = default;

  // Core state operations
  virtual void initialize(const IConfig& config) = 0;
  virtual void reset() = 0;
  virtual void validate() const = 0;

  // Copy operations
  virtual void copyFrom(const IState& other) = 0;
  virtual void moveFrom(IState&& other) = 0;
  virtual bool equals(const IState& other) const = 0;

  // Data access
  virtual void* getData() = 0;
  virtual const void* getData() const = 0;

  // Metadata
  virtual void setMetadata(const std::string& key,
                           const std::string& value) = 0;
  virtual std::string getMetadata(const std::string& key) const = 0;

  // State information
  virtual const std::vector<std::string>& getVariableNames() const = 0;
  virtual const std::vector<size_t>& getDimensions() const = 0;
  virtual bool isInitialized() const = 0;
};

}  // namespace repr
}  // namespace framework
}  // namespace metada

#endif  // METADA_FRAMEWORK_REPR_ISTATE_HPP_