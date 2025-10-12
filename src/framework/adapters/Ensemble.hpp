#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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
 * @brief Adapter class for ensemble of states in data assimilation systems
 *
 * @details This class provides a type-safe interface for managing an ensemble
 * of State<BackendTag> objects. It supports ensemble operations such as mean
 * and perturbation computation, and is designed for use in ensemble-based data
 * assimilation methods like LETKF.
 *
 * @tparam BackendTag The backend tag type that must satisfy StateBackendType
 */
template <typename BackendTag>
  requires StateBackendType<BackendTag>
class Ensemble : public NonCopyable {
 public:
  using StateType = State<BackendTag>;
  using MatrixType = Eigen::MatrixXd;
  using VectorType = Eigen::VectorXd;

  /**
   * @brief Construct an ensemble of states with the given config and geometry
   * @param config Configuration object for state initialization
   * @param geometry Geometry object for state initialization
   */
  explicit Ensemble(const Config<BackendTag>& config,
                    const Geometry<BackendTag>& geometry)
      : config_(config),
        geometry_(geometry),
        members_(),
        size_(0),
        perturbations_() {
    logger_.Info() << "Ensemble starting construction";

    // Check configuration type and initialize accordingly
    // ============================================================================
    // ENSEMBLE CONFIGURATION DETECTION AND INITIALIZATION
    // ============================================================================
    // Three supported ensemble configuration methods:
    // 1. members: Multiple state files (standard method)
    // 2. ensemble_file: Single ensemble file (MACOM format)
    // 3. perturbation_method: Perturbation generation (new method)

    if (config.HasKey("members")) {
      // ========================================================================
      // METHOD 1: MULTIPLE STATE FILES
      // ========================================================================
      logger_.Info() << "Using multiple state files ensemble configuration";

      const auto member_configs = config.GetSubsectionsFromVector("members");
      size_ = member_configs.size();
      members_.reserve(size_);

      for (size_t i = 0; i < member_configs.size(); ++i) {
        const auto& member_config = member_configs[i];
        logger_.Info() << "Initializing ensemble member " << i;
        members_.emplace_back(member_config.GetSubsection("state"), geometry);
      }

      logger_.Info() << "Ensemble constructed with " << size_ << " members";
      is_single_file_mode_ = false;
      is_loaded_ = true;  // Multi-member mode is immediately loaded

    } else if (config.HasKey("ensemble_file")) {
      // ========================================================================
      // METHOD 2: SINGLE ENSEMBLE FILE (MACOM FORMAT)
      // ========================================================================
      logger_.Info() << "Using single ensemble file configuration";

      try {
        // Get ensemble configuration directly from config
        ensemble_file_ = config.Get("ensemble_file").asString();
        size_ = config.Get("ensemble_size").asInt();
        state_dimension_ = config.Get("state_dimension").asInt();

        logger_.Info() << "Single file ensemble configuration:";
        logger_.Info() << "  - Ensemble file: " << ensemble_file_;
        logger_.Info() << "  - Ensemble size: " << size_;
        logger_.Info() << "  - State dimension: " << state_dimension_;
        logger_.Info() << "  - Data loading will be done on demand";

        is_single_file_mode_ = true;
        is_loaded_ = false;
      } catch (const std::exception& e) {
        logger_.Error() << "Failed to configure single file ensemble: "
                        << e.what();
        throw;
      }

    } else if (config.HasKey("perturbation_method")) {
      // ========================================================================
      // METHOD 3: PERTURBATION GENERATION
      // ========================================================================
      logger_.Info() << "Using perturbation generation ensemble configuration";

      try {
        // Get perturbation configuration
        const auto& pert_config = config.GetSubsection("perturbation_method");
        perturbation_type_ = pert_config.Get("type").asString();
        size_ = pert_config.Get("ensemble_size").asInt();

        // Validate required fields in perturbation_method
        if (!pert_config.HasKey("file")) {
          throw std::runtime_error(
              "perturbation_method requires 'file' configuration");
        }

        logger_.Info() << "Perturbation ensemble configuration:";
        logger_.Info() << "  - Perturbation type: " << perturbation_type_;
        logger_.Info() << "  - Ensemble size: " << size_;
        logger_.Info() << "  - Initial state file: "
                       << pert_config.Get("file").asString();

        // Store configuration for later use (lazy loading)
        // Note: We store pointers to the config subsections, which are valid
        // as long as the main config object is alive
        perturbation_config_ = &pert_config;

        is_single_file_mode_ = false;
        is_loaded_ = false;  // Perturbation mode loads on demand

        logger_.Info() << "  - Ensemble generation will be done on demand";
      } catch (const std::exception& e) {
        logger_.Error() << "Failed to configure perturbation ensemble: "
                        << e.what();
        throw;
      }

    } else {
      // ========================================================================
      // INVALID CONFIGURATION
      // ========================================================================
      logger_.Error() << "Invalid ensemble configuration: must specify one of:";
      logger_.Error() << "  - 'members': Multiple state files";
      logger_.Error()
          << "  - 'ensemble_file': Single ensemble file (MACOM format)";
      logger_.Error() << "  - 'perturbation_method': Perturbation generation";
      throw std::runtime_error(
          "Invalid ensemble configuration: must specify one of 'members', "
          "'ensemble_file', or 'perturbation_method'");
    }
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

    if (is_single_file_mode_) {
      // Single-file mode: load data and create State objects on demand
      if (!is_loaded_) {
        LoadFromMacomFile(ensemble_file_, size_, state_dimension_);
        is_loaded_ = true;
      }

      // Check if we need to create the members_ vector
      if (members_.empty()) {
        members_.reserve(size_);
        // Create State objects that share geometry but have different data
        for (size_t i = 0; i < size_; ++i) {
          members_.emplace_back(config_, geometry_);
          // Initialize with data from ensemble_matrix_
          initializeMemberFromMatrix(members_[i], i);
        }
      }
      return members_[index];
    } else {
      // Perturbation mode: generate ensemble members on demand
      if (!is_loaded_) {
        GeneratePerturbationEnsemble();
        is_loaded_ = true;
      }
      return members_[index];
    }
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
    return members_[index];
  }

  /**
   * @brief Get the number of ensemble members
   * @return Ensemble size
   */
  size_t Size() const { return size_; }

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

  // ============================================================================
  // MACOM-SPECIFIC METHODS
  // ============================================================================

  /**
   * @brief Compute ensemble mean (MACOM version)
   * @return Vector containing the ensemble mean
   */
  VectorType ComputeMacomMean() const {
    if (!is_loaded_) {
      throw std::runtime_error("MACOM ensemble not initialized");
    }

    VectorType mean = VectorType::Zero(state_dimension_);
    for (size_t m = 0; m < size_; ++m) {
      mean += ensemble_matrix_.col(m);
    }
    mean /= static_cast<double>(size_);

    return mean;
  }

  /**
   * @brief Compute ensemble perturbations (MACOM version)
   * @return Matrix containing ensemble perturbations X'(P,M)
   */
  MatrixType ComputeMacomPerturbations() const {
    if (!is_loaded_) {
      throw std::runtime_error("MACOM ensemble not initialized");
    }

    VectorType mean = ComputeMacomMean();
    MatrixType perturbations = ensemble_matrix_;

    // Subtract mean from each member: X' = X - X_mean
    for (size_t m = 0; m < size_; ++m) {
      perturbations.col(m) -= mean;
    }

    return perturbations;
  }

  /**
   * @brief Compute ensemble covariance matrix (MACOM version)
   * @param mu Scaling factor (from configuration, default 1.0e-6)
   * @return Covariance matrix BB(M,M)
   */
  MatrixType ComputeMacomCovarianceMatrix(double mu = -1.0) const {
    if (!is_loaded_) {
      throw std::runtime_error("MACOM ensemble not initialized");
    }

    // Use configuration MU if not provided
    if (mu < 0.0) {
      mu = GetMuScalingFactor();
    }

    MatrixType perturbations = ComputeMacomPerturbations();
    MatrixType covariance = MatrixType::Zero(size_, size_);

    // BB(M,N) = sum_P X'(P,M) * X'(P,N) * mu
    for (size_t m = 0; m < size_; ++m) {
      for (size_t n = 0; n < size_; ++n) {
        covariance(m, n) = perturbations.col(m).dot(perturbations.col(n)) * mu;
      }
    }

    return covariance;
  }

  /**
   * @brief Perform eigenvalue decomposition of covariance matrix (MACOM
   * version)
   * @param tolerance Tolerance for eigenvalue decomposition
   * @return Pair of (eigenvalues, eigenvectors)
   */
  std::pair<VectorType, MatrixType> MacomEigenDecomposition(
      double /*tolerance*/ = 1.0e-7) const {
    if (!is_loaded_) {
      throw std::runtime_error("MACOM ensemble not initialized");
    }

    MatrixType covariance = ComputeMacomCovarianceMatrix();
    Eigen::SelfAdjointEigenSolver<MatrixType> solver(covariance);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigenvalue decomposition failed");
    }

    VectorType eigen_values = solver.eigenvalues();
    MatrixType eigen_vectors = solver.eigenvectors();

    // Sort eigenvalues in descending order
    std::vector<std::pair<double, int>> eigen_pairs;
    for (int i = 0; i < static_cast<int>(eigen_values.size()); ++i) {
      eigen_pairs.push_back(std::make_pair(eigen_values(i), i));
    }
    std::sort(
        eigen_pairs.begin(), eigen_pairs.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
          return a.first > b.first;
        });

    VectorType sorted_eigen_values = VectorType::Zero(size_);
    MatrixType sorted_eigen_vectors = MatrixType::Zero(size_, size_);

    for (size_t i = 0; i < size_; ++i) {
      sorted_eigen_values(i) = eigen_pairs[i].first;
      sorted_eigen_vectors.col(i) = eigen_vectors.col(eigen_pairs[i].second);
    }

    return std::make_pair(sorted_eigen_values, sorted_eigen_vectors);
  }

  /**
   * @brief Save ensemble to file for debugging/checking (first 10 rows only)
   * @param filename Output filename
   */
  void SaveMacomToFile(const std::string& filename) const {
    if (!is_loaded_) {
      throw std::runtime_error("MACOM ensemble not initialized");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Save in Fortran format: X(P,M) - first 10 rows only
    int max_rows = std::min(10, (int)state_dimension_);
    for (size_t m = 0; m < size_; ++m) {
      for (int p = 0; p < max_rows; ++p) {
        file << std::scientific << std::setprecision(15)
             << ensemble_matrix_(p, m) << "\n";
      }
    }
    file.close();
  }

  /**
   * @brief Get state dimension (MACOM version)
   * @return Dimension of the state vector
   */
  size_t StateDimension() const { return state_dimension_; }

  /**
   * @brief Check if single-file ensemble data is loaded
   * @return True if loaded, false otherwise
   */
  bool IsSingleFileLoaded() const { return is_loaded_; }

  /**
   * @brief Check if this is a single-file ensemble
   * @return True if using single-file format, false if using multi-file format
   */
  bool IsSingleFileMode() const { return is_single_file_mode_; }

  /**
   * @brief Get background field XB(P) from first ensemble member
   * @return Vector containing background field
   */
  VectorType GetBackgroundField() const {
    if (!is_loaded_) {
      // Auto-load ensemble data if not already loaded
      if (is_single_file_mode_) {
        const_cast<Ensemble*>(this)->LoadFromMacomFile(ensemble_file_, size_,
                                                       state_dimension_);
      } else {
        throw std::runtime_error("MACOM ensemble not initialized");
      }
    }

    return ensemble_matrix_.col(0);  // First ensemble member as background
  }

  /**
   * @brief Get ensemble matrix X(P,M)
   * @return Matrix containing all ensemble members
   */
  const MatrixType& GetEnsembleMatrix() const {
    if (!is_loaded_) {
      // Auto-load ensemble data if not already loaded
      if (is_single_file_mode_) {
        const_cast<Ensemble*>(this)->LoadFromMacomFile(ensemble_file_, size_,
                                                       state_dimension_);
      } else {
        throw std::runtime_error("MACOM ensemble not initialized");
      }
    }

    return ensemble_matrix_;
  }

  /**
   * @brief Get ensemble file path from configuration
   * @return Path to the ensemble file
   */
  std::string GetEnsembleFilePath() const {
    if (!config_.HasKey("ensemble_file")) {
      throw std::runtime_error("Ensemble file path not found in configuration");
    }
    return config_.Get("ensemble_file").asString();
  }

  /**
   * @brief Get ensemble size from configuration
   * @return Number of ensemble members
   */
  size_t GetEnsembleSize() const {
    if (!config_.HasKey("ensemble_size")) {
      throw std::runtime_error("Ensemble size not found in configuration");
    }
    return config_.Get("ensemble_size").asInt();
  }

  /**
   * @brief Get MU scaling factor from configuration
   * @return MU scaling factor for ensemble perturbations
   */
  double GetMuScalingFactor() const {
    // Try to get from macom section first, then from ensemble section
    if (config_.HasKey("macom") &&
        config_.GetSubsection("macom").HasKey("mu")) {
      return static_cast<double>(
          config_.GetSubsection("macom").Get("mu").asFloat());
    } else if (config_.HasKey("mu")) {
      return static_cast<double>(config_.Get("mu").asFloat());
    } else {
      // Default value from mod_csp_basic.f90
      return 1.0e-6;
    }
  }

  // ============================================================================
  // METHOD 1: MULTIPLE STATE FILES - Traditional Ensemble Configuration
  // ============================================================================
  // This method loads each ensemble member from a separate file.
  // Configuration format:
  // ensemble:
  //   members:
  //     - state: {file: "path1.nc", timestamp: "...", variables: [...]}
  //     - state: {file: "path2.nc", timestamp: "...", variables: [...]}
  //     - ...
  //
  // No specific methods needed for this mode - members are loaded directly
  // in the constructor and accessed via GetMember()

  // ============================================================================
  // METHOD 2: SINGLE ENSEMBLE FILE - MACOM Format
  // ============================================================================
  // This method loads all ensemble members from a single binary file.
  // Configuration format:
  // ensemble:
  //   ensemble_file: "path/to/X_MEMBER_HYCOM.DAT"
  //   ensemble_size: 5
  //   state_dimension: 74687333

  /**
   * @brief Load ensemble members from external file (MACOM format)
   * @param filename Path to the ensemble member file (X_MEMBER_HYCOM.DAT
   * format)
   * @param ensemble_size Number of ensemble members
   * @param state_dimension Dimension of state vector (from configuration)
   */
  void LoadFromMacomFile(const std::string& filename, size_t ensemble_size,
                         size_t state_dimension) {
    size_ = ensemble_size;
    state_dimension_ = state_dimension;

    // Control switch for testing - set to false to use full ensemble data
    const bool LIMIT_ENSEMBLE = false;
    const size_t MAX_ENSEMBLE_ROWS = 1000;

    // Load data from file
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open ensemble file: " + filename);
    }

    if (LIMIT_ENSEMBLE) {
      // Limited mode: use smaller state dimension for testing
      size_t limited_state_dim = MAX_ENSEMBLE_ROWS / size_;
      if (limited_state_dim == 0) {
        throw std::runtime_error(
            "MAX_ENSEMBLE_ROWS (" + std::to_string(MAX_ENSEMBLE_ROWS) +
            ") is smaller than ensemble size (" + std::to_string(size_) + ")");
      }

      // Resize ensemble matrix for limited data
      ensemble_matrix_.resize(limited_state_dim, size_);

      // Read only the limited amount of data
      size_t elements_to_read = limited_state_dim * size_;
      size_t element_count = 0;

      double value;
      while (file >> value && element_count < elements_to_read) {
        size_t m = element_count / limited_state_dim;  // Member index
        size_t p = element_count % limited_state_dim;  // State index
        ensemble_matrix_(p, m) = value;
        element_count++;
      }

      std::cout << "--> Limited mode: reading " << element_count
                << " elements for testing" << std::endl;
      std::cout << "--> Limited state dimension: " << limited_state_dim
                << " (original: " << state_dimension_ << ")" << std::endl;

    } else {
      // Full mode: use configured state dimension directly
      ensemble_matrix_.resize(state_dimension_, size_);

      // Read data directly into matrix
      size_t element_count = 0;
      double value;
      while (file >> value && element_count < state_dimension_ * size_) {
        size_t m = element_count / state_dimension_;  // Member index
        size_t p = element_count % state_dimension_;  // State index
        ensemble_matrix_(p, m) = value;
        element_count++;
      }

      if (element_count < state_dimension_ * size_) {
        throw std::runtime_error("Insufficient data in file: expected " +
                                 std::to_string(state_dimension_ * size_) +
                                 " elements, got " +
                                 std::to_string(element_count));
      }

      std::cout << "--> Full mode: reading " << element_count << " elements"
                << std::endl;
      std::cout << "--> State dimension (NP): " << state_dimension_
                << std::endl;
    }

    file.close();
    is_loaded_ = true;
  }

  /**
   * @brief Initialize a State member from ensemble matrix data
   * @param member State object to initialize
   * @param member_index Index of the ensemble member
   */
  void initializeMemberFromMatrix(StateType& member, size_t member_index) {
    // For MACOM mode, we don't need to set individual background fields
    // The ensemble data is already loaded in ensemble_matrix_
    // Each State object will share the same geometry and configuration
    // The actual data access will be handled through the ensemble_matrix_

    // Suppress unused parameter warnings
    (void)member;
    (void)member_index;

    // No additional initialization needed for now
    // The member State object is already created with shared geometry
  }

  // ============================================================================
  // METHOD 3: PERTURBATION GENERATION - New Method
  // ============================================================================
  // This method generates ensemble members by applying perturbations to an
  // initial state. Supports Gaussian and uniform perturbation types.
  // Configuration format:
  // ensemble:
  //   perturbation_method:
  //     type: "gaussian"  # or "uniform"
  //     ensemble_size: 5
  //     std_dev: 0.1      # for Gaussian
  //     # amplitude: 0.1  # for uniform
  //   initial_state:
  //     file: "path/to/initial.nc"
  //     timestamp: "..."
  //     variables: [...]

  /**
   * @brief Generate ensemble members using perturbation method
   * @details This method creates ensemble members by applying perturbations
   * to an initial state based on the configured perturbation method
   */
  void GeneratePerturbationEnsemble() {
    logger_.Info() << "Generating perturbation ensemble with " << size_
                   << " members";

    // Re-get configuration from main config to avoid pointer issues
    if (!config_.HasKey("perturbation_method")) {
      logger_.Error() << "Main config missing 'perturbation_method' section!";
      throw std::runtime_error(
          "Main config missing 'perturbation_method' section");
    }

    const auto& perturbation_config =
        config_.GetSubsection("perturbation_method");

    if (!perturbation_config.HasKey("file")) {
      logger_.Error() << "Perturbation config missing 'file' key!";
      throw std::runtime_error("Perturbation config missing 'file' key");
    }

    std::string file_path = perturbation_config.Get("file").asString();
    logger_.Info() << "Template state file path: " << file_path;

    // Check if file exists
    std::ifstream file_check(file_path);
    if (!file_check.good()) {
      logger_.Error()
          << "Initial state file does not exist or cannot be accessed: "
          << file_path;
      throw std::runtime_error("Initial state file not accessible: " +
                               file_path);
    }
    file_check.close();

    logger_.Info() << "Creating template state...";
    try {
      // Load initial state as template for cloning
      StateType template_state(perturbation_config, geometry_);
      logger_.Info() << "Template state created successfully from: "
                     << file_path;

      // Create ensemble members
      members_.clear();
      members_.reserve(size_);

      // First member: unperturbed initial state (create fresh from config)
      logger_.Info()
          << "Creating ensemble member 0 (unperturbed initial state)";
      StateType member0(perturbation_config, geometry_);

      // Ensure the first member is properly initialized
      if (!member0.isInitialized()) {
        logger_.Error() << "First state is not initialized!";
        throw std::runtime_error("First state is not initialized");
      }
      logger_.Info() << "First member initialized, size: " << member0.size();

      members_.push_back(std::move(member0));
      logger_.Info() << "Created ensemble member 0 successfully";

      // Generate perturbed members (starting from index 1)
      for (size_t i = 1; i < size_; ++i) {
        logger_.Info() << "Creating ensemble member " << i << " (perturbed)";

        // Create a new state by fresh initialization from config
        StateType member(perturbation_config, geometry_);

        // Verify the state is valid before applying perturbation
        if (!member.isInitialized()) {
          logger_.Error() << "State " << i << " is not initialized!";
          throw std::runtime_error("State is not initialized");
        }

        logger_.Info() << "State " << i
                       << " is initialized, size: " << member.size();

        // Apply perturbation to this member
        ApplyPerturbation(member, i, perturbation_config);

        // Move the perturbed state to the vector
        members_.push_back(std::move(member));

        logger_.Info() << "Created ensemble member " << i << " successfully";
      }

      logger_.Info() << "Perturbation ensemble generated successfully with "
                     << members_.size() << " members";
    } catch (const std::exception& e) {
      logger_.Error() << "Failed to create template state: " << e.what();
      throw std::runtime_error("Failed to create template state: " +
                               std::string(e.what()));
    } catch (...) {
      logger_.Error() << "Unknown error occurred while creating template state";
      throw std::runtime_error(
          "Unknown error occurred while creating template state");
    }
  }

  /**
   * @brief Apply perturbation to a state based on configured method
   * @param state State to perturb
   * @param member_index Index of the ensemble member (for seed generation)
   */
  void ApplyPerturbation(StateType& state, size_t member_index,
                         const Config<BackendTag>& perturbation_config) {
    std::string pert_type = perturbation_config.Get("type").asString();
    if (pert_type == "gaussian") {
      ApplyGaussianPerturbation(state, member_index, perturbation_config);
    } else if (pert_type == "uniform") {
      ApplyUniformPerturbation(state, member_index, perturbation_config);
    } else {
      throw std::runtime_error("Unknown perturbation type: " + pert_type);
    }
  }

  /**
   * @brief Apply Gaussian perturbation to a state
   * @param state State to perturb
   * @param member_index Index of the ensemble member
   */
  void ApplyGaussianPerturbation(
      StateType& state, size_t member_index,
      const Config<BackendTag>& perturbation_config) {
    // Get perturbation parameters
    double std_dev = 1.0;
    if (perturbation_config.HasKey("std_dev")) {
      std_dev = perturbation_config.Get("std_dev").asFloat();
    }

    // Set random seed based on member index for reproducibility
    std::srand(static_cast<unsigned int>(member_index + 12345));

    // Check if variables configuration exists
    if (!perturbation_config.HasKey("variables")) {
      logger_.Warning()
          << "No variables configuration found, skipping perturbation";
      return;
    }

    // Get variables to perturb from initial state configuration
    const auto& variables =
        perturbation_config.Get("variables").asVectorString();

    if (variables.empty()) {
      logger_.Warning() << "No variables specified for perturbation";
      return;
    }

    // Apply perturbation only to specified variables
    for (const auto& var_str : variables) {
      // Check if this variable exists in the state
      if (state.hasVariable(var_str)) {
        logger_.Info() << "Applying Gaussian perturbation to variable: "
                       << var_str;

        // Get variable data - this is a simplified approach
        // For MACOM backend, we need to access the specific variable data
        auto* data_ptr = state.template getDataPtr<double>();
        size_t state_size = state.size();

        // Safety check to prevent memory access issues
        if (data_ptr == nullptr) {
          logger_.Warning()
              << "Data pointer is null, skipping perturbation for variable: "
              << var_str;
          continue;
        }

        if (state_size == 0) {
          logger_.Warning()
              << "State size is zero, skipping perturbation for variable: "
              << var_str;
          continue;
        }

        // Apply perturbation to the variable data
        // Note: This is a simplified approach - in practice, we would need
        // to get the specific range of data for each variable
        size_t max_perturb_elements =
            std::min(state_size,
                     static_cast<size_t>(1000));  // Reduced limit for safety
        for (size_t i = 0; i < max_perturb_elements; ++i) {
          // Generate Gaussian random number using Box-Muller transform
          double u1 = static_cast<double>(std::rand()) / RAND_MAX;
          double u2 = static_cast<double>(std::rand()) / RAND_MAX;

          // Avoid log(0) and ensure u1 > 0
          if (u1 <= 0.0) u1 = 1e-10;

          double gaussian_noise = std::sqrt(-2.0 * std::log(u1)) *
                                  std::cos(2.0 * 3.14159265358979323846 * u2);

          data_ptr[i] += std_dev * gaussian_noise;
        }

        logger_.Info() << "Applied perturbation to " << max_perturb_elements
                       << " elements";
        break;  // For now, only perturb the first variable to avoid
                // over-perturbation
      } else {
        logger_.Warning() << "Variable " << var_str
                          << " not found in state, skipping";
      }
    }
  }

  /**
   * @brief Apply uniform perturbation to a state
   * @param state State to perturb
   * @param member_index Index of the ensemble member
   */
  void ApplyUniformPerturbation(StateType& state, size_t member_index,
                                const Config<BackendTag>& perturbation_config) {
    // Get perturbation parameters
    double amplitude = 1.0;
    if (perturbation_config.HasKey("amplitude")) {
      amplitude = perturbation_config.Get("amplitude").asFloat();
    }

    // Set random seed based on member index for reproducibility
    std::srand(static_cast<unsigned int>(member_index + 12345));

    // Check if variables configuration exists
    if (!perturbation_config.HasKey("variables")) {
      logger_.Warning()
          << "No variables configuration found, skipping perturbation";
      return;
    }

    // Get variables to perturb from initial state configuration
    const auto& variables =
        perturbation_config.Get("variables").asVectorString();

    if (variables.empty()) {
      logger_.Warning() << "No variables specified for perturbation";
      return;
    }

    // Apply perturbation only to specified variables
    for (const auto& var_str : variables) {
      // Check if this variable exists in the state
      if (state.hasVariable(var_str)) {
        logger_.Info() << "Applying uniform perturbation to variable: "
                       << var_str;

        // Get variable data - this is a simplified approach
        // For MACOM backend, we need to access the specific variable data
        auto* data_ptr = state.template getDataPtr<double>();
        size_t state_size = state.size();

        // Safety check to prevent memory access issues
        if (data_ptr == nullptr) {
          logger_.Warning()
              << "Data pointer is null, skipping perturbation for variable: "
              << var_str;
          continue;
        }

        if (state_size == 0) {
          logger_.Warning()
              << "State size is zero, skipping perturbation for variable: "
              << var_str;
          continue;
        }

        // Apply perturbation to the variable data
        // Note: This is a simplified approach - in practice, we would need
        // to get the specific range of data for each variable
        size_t max_perturb_elements =
            std::min(state_size,
                     static_cast<size_t>(1000));  // Reduced limit for safety
        for (size_t i = 0; i < max_perturb_elements; ++i) {
          // Generate uniform random number in [-amplitude, amplitude]
          double uniform_noise =
              2.0 * amplitude *
              (static_cast<double>(std::rand()) / RAND_MAX - 0.5);
          data_ptr[i] += uniform_noise;
        }

        logger_.Info() << "Applied perturbation to " << max_perturb_elements
                       << " elements";
        break;  // For now, only perturb the first variable to avoid
                // over-perturbation
      } else {
        logger_.Warning() << "Variable " << var_str
                          << " not found in state, skipping";
      }
    }
  }

 private:
  const Config<BackendTag>& config_;
  const Geometry<BackendTag>& geometry_;
  std::vector<StateType> members_;
  std::unique_ptr<StateType> mean_;
  size_t size_;
  std::vector<std::unique_ptr<StateType>> perturbations_;
  Logger<BackendTag>& logger_ = Logger<BackendTag>::Instance();

  // Single-file ensemble mode members
  std::string ensemble_file_;         // Path to ensemble file
  size_t state_dimension_ = 0;        // State dimension
  MatrixType ensemble_matrix_;        // Ensemble data matrix
  bool is_single_file_mode_ = false;  // Whether using single-file mode
  bool is_loaded_ = false;  // Whether ensemble is loaded (for both modes)

  // Perturbation generation mode members
  std::string
      perturbation_type_;  // Type of perturbation (e.g., "gaussian", "uniform")
  const Config<BackendTag>*
      perturbation_config_;  // Pointer to perturbation config

  // ============================================================================
  // COMMON UTILITY METHODS
  // ============================================================================
  // These methods are used by all three ensemble configuration methods
  // (Already defined above: GetMuScalingFactor, GetEnsembleSize, etc.)
};

}  // namespace metada::framework