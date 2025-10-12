#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "BackendTraits.hpp"
#include "ConfigConcepts.hpp"
#include "Logger.hpp"
#include "NonCopyable.hpp"
#include "State.hpp"
#include "StateConcepts.hpp"

namespace metada::framework {

/**
 * @brief Forward declaration of Config class
 */
template <typename BackendTag>
  requires ConfigBackendType<BackendTag>
class Config;

/**
 * @brief Forward declaration of Geometry class
 */
template <typename BackendTag>
  requires GeometryBackendType<BackendTag>
class Geometry;

/**
 * @brief MACOM-specific ensemble class for single-file ensemble data
 *
 * @details This class handles MACOM-style ensemble data where all members
 * are stored in a single file. It provides lazy loading and efficient
 * data access for large ensemble datasets.
 *
 * @tparam BackendTag The backend tag type that must satisfy StateBackendType
 */
template <typename BackendTag>
  requires StateBackendType<BackendTag>
class MACOMEnsemble : public NonCopyable {
 public:
  using StateType = State<BackendTag>;
  using MatrixType = Eigen::MatrixXd;
  using VectorType = Eigen::VectorXd;

  /**
   * @brief Construct a MACOM ensemble with the given config and geometry
   * @param config Configuration object containing ensemble file path and
   * parameters
   * @param geometry Geometry object for state initialization
   */
  explicit MACOMEnsemble(const Config<BackendTag>& config,
                         const Geometry<BackendTag>& geometry)
      : config_(config),
        geometry_(geometry),
        members_(),
        size_(0),
        state_dimension_(0),
        ensemble_matrix_(),
        is_initialized_(false),
        perturbations_() {
    logger_.Info() << "MACOMEnsemble starting construction";

    // Read MACOM-specific configuration
    ensemble_file_ = config.Get("ensemble_file").asString();
    size_ = config.Get("ensemble_size").asInt();
    state_dimension_ = config.Get("state_dimension").asInt();

    logger_.Info() << "MACOM ensemble configuration:";
    logger_.Info() << "  - Ensemble file: " << ensemble_file_;
    logger_.Info() << "  - Ensemble size: " << size_;
    logger_.Info() << "  - State dimension: " << state_dimension_;
    logger_.Info() << "  - Data loading will be done on demand";
  }

  /**
   * @brief Get mutable access to an ensemble member
   * @param index Index of the member
   * @return Reference to the member state
   * @throws std::out_of_range if index is invalid
   */
  StateType& GetMember(size_t index) {
    if (index >= size_) {
      throw std::out_of_range("Ensemble member index out of range");
    }

    // Lazy initialization
    if (!is_initialized_) {
      initializeEnsemble();
    }

    return members_[index];
  }

  /**
   * @brief Get const access to an ensemble member
   * @param index Index of the member
   * @return Const reference to the member state
   * @throws std::out_of_range if index is invalid
   */
  const StateType& GetMember(size_t index) const {
    if (index >= size_) {
      throw std::out_of_range("Ensemble member index out of range");
    }

    // Lazy initialization
    if (!is_initialized_) {
      const_cast<MACOMEnsemble*>(this)->initializeEnsemble();
    }

    return members_[index];
  }

  /**
   * @brief Get the number of ensemble members
   * @return Ensemble size
   */
  size_t Size() const { return size_; }

  /**
   * @brief Get the ensemble matrix for direct access
   * @return Reference to the ensemble matrix
   */
  const MatrixType& GetEnsembleMatrix() const {
    if (!is_initialized_) {
      const_cast<MACOMEnsemble*>(this)->initializeEnsemble();
    }
    return ensemble_matrix_;
  }

  /**
   * @brief Force recomputation of the mean state
   */
  void RecomputeMean() {
    if (!mean_) {
      mean_ = std::make_unique<StateType>(members_[0].clone());
    }
    mean_->zero();
    for (size_t i = 0; i < size_; ++i) {
      *mean_ += members_[i];
    }
    *mean_ *= (1.0 / static_cast<double>(size_));
  }

  /**
   * @brief Get the mean state of the ensemble, computing it if necessary
   * @return Reference to the mean state
   */
  StateType& Mean() {
    if (!mean_) {
      RecomputeMean();
    }
    return *mean_;
  }

  /**
   * @brief Get const access to the mean state
   * @return Const reference to the mean state
   * @throws std::runtime_error if mean hasn't been computed
   */
  const StateType& Mean() const {
    if (!mean_) {
      throw std::runtime_error("Mean has not been computed");
    }
    return *mean_;
  }

  /**
   * @brief Force recomputation of all perturbations
   */
  void RecomputePerturbations() {
    RecomputeMean();
    perturbations_.clear();
    perturbations_.reserve(size_);
    for (size_t i = 0; i < size_; ++i) {
      perturbations_.push_back(
          std::make_unique<StateType>(members_[i].clone()));
      *perturbations_.back() -= *mean_;
    }
  }

  /**
   * @brief Get a perturbation, computing it if necessary
   * @param index Index of the perturbation
   * @return Reference to the perturbation state
   * @throws std::out_of_range if index is invalid
   */
  StateType& GetPerturbation(size_t index) {
    if (index >= size_) {
      throw std::out_of_range("Perturbation index out of range");
    }
    if (perturbations_.empty()) {
      RecomputePerturbations();
    }
    return *perturbations_[index];
  }

  /**
   * @brief Get const access to a perturbation
   * @param index Index of the perturbation
   * @return Const reference to the perturbation state
   * @throws std::out_of_range if index is invalid
   * @throws std::runtime_error if perturbations haven't been computed
   */
  const StateType& GetPerturbation(size_t index) const {
    if (index >= size_) {
      throw std::out_of_range("Perturbation index out of range");
    }
    if (perturbations_.empty()) {
      throw std::runtime_error("Perturbations have not been computed");
    }
    return *perturbations_[index];
  }

 private:
  const Config<BackendTag>& config_;
  const Geometry<BackendTag>& geometry_;
  std::vector<StateType> members_;
  std::unique_ptr<StateType> mean_;
  size_t size_;
  size_t state_dimension_;
  std::string ensemble_file_;
  MatrixType ensemble_matrix_;
  bool is_initialized_;
  std::vector<std::unique_ptr<StateType>> perturbations_;
  Logger<BackendTag>& logger_ = Logger<BackendTag>::Instance();

  /**
   * @brief Initialize the ensemble by loading data from file
   */
  void initializeEnsemble() {
    logger_.Info() << "Loading MACOM ensemble data from: " << ensemble_file_;

    // Load ensemble data from file
    LoadFromMacomFile(ensemble_file_, size_, state_dimension_);

    // Create State objects for each member
    members_.reserve(size_);
    for (size_t i = 0; i < size_; ++i) {
      members_.emplace_back(config_, geometry_);
      initializeMemberFromMatrix(members_[i], i);
    }

    is_initialized_ = true;
    logger_.Info() << "MACOM ensemble initialized with " << size_ << " members";
  }

  /**
   * @brief Load ensemble data from MACOM file
   * @param filename Path to the ensemble file
   * @param ensemble_size Number of ensemble members
   * @param state_dimension Dimension of each state
   */
  void LoadFromMacomFile(const std::string& filename, size_t ensemble_size,
                         size_t state_dimension) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open ensemble file: " + filename);
    }

    // Resize matrix: [state_dimension x ensemble_size]
    ensemble_matrix_.resize(state_dimension, ensemble_size);

    // Read data directly into matrix
    file.read(reinterpret_cast<char*>(ensemble_matrix_.data()),
              state_dimension * ensemble_size * sizeof(double));

    if (file.fail()) {
      throw std::runtime_error("Failed to read ensemble data from: " +
                               filename);
    }

    file.close();
    logger_.Info() << "Loaded ensemble data: " << state_dimension << " x "
                   << ensemble_size << " matrix";
  }

  /**
   * @brief Initialize a State member from ensemble matrix data
   * @param member State object to initialize
   * @param member_index Index of the ensemble member
   */
  void initializeMemberFromMatrix(StateType& member, size_t member_index) {
    // Get the ensemble data for this member from the matrix
    std::vector<double> member_data(state_dimension_);
    for (size_t i = 0; i < state_dimension_; ++i) {
      member_data[i] = ensemble_matrix_(i, member_index);
    }

    // Set the data using the backend's setFieldData method
    member.backend().setFieldData("t", member_data);

    logger_.Info() << "Initialized MACOM ensemble member " << member_index
                   << " with " << member_data.size() << " data points";
  }
};

}  // namespace metada::framework
