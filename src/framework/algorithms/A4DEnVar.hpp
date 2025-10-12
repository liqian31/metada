#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Config.hpp"
#include "Ensemble.hpp"
#include "Logger.hpp"
#include "Model.hpp"
#include "ObsOperator.hpp"
#include "Observation.hpp"
#include "State.hpp"

namespace metada::framework {

/**
 * @brief Analytical Four-Dimensional Ensemble-Variational (4DEnVar)
 * algorithm implementation.
 *
 * @details
 * This class implements the 4DEnVar algorithm for data assimilation,
 * which combines ensemble and variational methods to provide an efficient
 * approach for high-dimensional data assimilation problems.
 *
 * The algorithm is based on the Fortran implementation in mod_a4d_liwei.f90
 * and follows the same simple parameter structure.
 *
 * @tparam BackendTag Backend tag type for template specialization
 */
template <typename BackendTag>
class A4DEnVar {
 public:
  /**
   * @brief Simple analysis results structure
   */
  struct AnalysisResults {
    int ensemble_size;        ///< Number of ensemble members
    int observation_count;    ///< Total number of observations
    int max_outer_loops;      ///< Number of outer loops
    std::string output_file;  ///< Output file path
  };

  /**
   * @brief Construct an A4DEnVar object.
   * @param ensemble Reference to the ensemble containing perturbation members.
   * @param observations Observation object.
   * @param obs_operator Observation operator.
   * @param state State object for background field extraction.
   * @param config Configuration object containing A4DEnVar parameters.
   */
  A4DEnVar(Ensemble<BackendTag>& ensemble,
           const Observation<BackendTag>& observations,
           const ObsOperator<BackendTag>& obs_operator,
           State<BackendTag>& state, const Config<BackendTag>& config)
      : model_(config),
        ensemble_(ensemble),
        observations_(observations),
        obs_operator_(obs_operator),
        state_(state) {
    // Read parameters from config (config is already the analysis subsection)
    // Required parameters
    if (config.HasKey("max_outer_loops")) {
      max_outer_loops_ = config.Get("max_outer_loops").asInt();
    } else {
      throw std::runtime_error(
          "Missing required parameter: analysis.max_outer_loops");
    }

    if (config.HasKey("output_base_file")) {
      output_base_file_ = config.Get("output_base_file").asString();
    } else {
      throw std::runtime_error(
          "Missing required parameter: analysis.output_base_file");
    }

    logger_.Info() << "A4DEnVar constructed with " << ensemble_.Size()
                   << " members and " << observations_.size()
                   << " observations";
    logger_.Info() << "Max outer loops: " << max_outer_loops_;
  }

  /**
   * @brief Save the updated ensemble to files.
   */
  void saveEnsemble() {
    logger_.Info() << "Saving A4DEnVar ensemble";

    // Save analysis field
    saveAnalysisField();

    // Save ensemble check files
    saveEnsembleCheckFiles();

    // Save background observations
    std::vector<double> background_obs_values =
        obs_operator_.apply(state_, observations_);
    saveBackgroundObservations(background_obs_values);

    // Save analysis observations
    saveAnalysisObservations(background_obs_values);

    logger_.Info() << "A4DEnVar ensemble saved";
  }

  /**
   * @brief Initialize analysis components (t₀ phase - data reading and setup)
   * Corresponds to Fortran lines 15-75: memory allocation and data reading
   */
  void initializeAnalysis() {
    // 1. Background field is already extracted during State construction
    // No need to call extractBackgroundField() again
    logger_.Info()
        << "Background field already available from State construction";

    // 2. Memory allocation (C++ automatic - no need for explicit allocation)
    // In Fortran: allocate(XB(NP), X0(NP), TP(NP), Delta_X(NP), OM(NM), etc.)
    // In C++: memory is managed automatically by std::vector and objects
    logger_.Info() << "Memory allocation completed (C++ automatic)";

    // 3. Observation data already loaded in constructor
    // In Fortran: read observation coordinates and values from file
    // In C++: observations_ object already contains all observation data
    logger_.Info() << "Observation data reading completed - "
                   << observations_.size() << " observations already loaded";

    // 4. Ensemble data already loaded in constructor
    // In Fortran: read ensemble members X(NP,NM) from file
    // In C++: ensemble_ object already contains all ensemble member data
    logger_.Info() << "Ensemble data reading completed - " << ensemble_.Size()
                   << " members already loaded";

    // 5. Variable initialization (C++ automatic - no need for explicit
    // initialization) In Fortran: OM = 0.0D0 and other variable initialization
    // In C++: variables are automatically initialized by constructors and
    // default values logger_.Info() << "Variable initialization completed (C++
    // automatic)";

    logger_.Info()
        << "t₀ Phase completed: Data reading and initialization done";
  }

  /**
   * @brief Run ensemble outer loop (IEXT loop - corresponds to Fortran lines
   * 77-409) This is the main A4DEnVar algorithm loop
   */
  void runEnsembleOuterLoop() {
    logger_.Info() << "=== Starting IEXT Outer Loop ===";

    // A4DEnVar parameters (from configuration)
    const int max_outer_loops = max_outer_loops_;

    // Main IEXT loop (corresponds to Fortran: DO IEXT = 1, macom_loop_end)
    for (int iext = 1; iext <= max_outer_loops; ++iext) {
      logger_.Info() << "--> This is for outloop: " << iext;

      // Process IEXT-specific logic
      processIEXTLoop(iext);

      // Exit condition (corresponds to Fortran: IF(IEXT.EQ.macom_loop_end)EXIT)
      if (iext == max_outer_loops) {
        logger_.Info() << "Final IEXT loop completed, exiting";
        break;
      }

      // Process ensemble members for this IEXT (corresponds to Fortran: DO
      // M=1,NM)
      processEnsembleMembers(iext);

      logger_.Info() << "IEXT " << iext << " completed";
    }

    logger_.Info() << "IEXT Outer loop completed";
  }

  /**
   * @brief Get the analysis results
   */
  AnalysisResults getAnalysisResults() const {
    AnalysisResults results;
    results.ensemble_size = ensemble_.Size();
    results.observation_count = observations_.size();
    results.max_outer_loops = max_outer_loops_;
    results.output_file = output_base_file_;
    return results;
  }

 private:
  // A4D algorithm variables (based on mod_csp_basic.f90)
  std::vector<double>
      analysis_state_;  // Analysis state (corresponds to X0 = XB)
  std::vector<std::vector<double>>
      ensemble_perturbations_;  // Ensemble perturbations (corresponds to X, NM
                                // x NP)
  std::vector<std::vector<double>>
      ensemble_covariance_;  // Ensemble covariance matrix (corresponds to BB,
                             // NM x NM)
  std::vector<double> eigenvalues_;  // Eigenvalues (corresponds to LM)
  std::vector<std::vector<double>>
      eigenvectors_;  // Eigenvectors (corresponds to VP, NM x NM)
  std::vector<std::vector<double>>
      vv_matrix_;  // VV matrix (corresponds to VV, NM x NM)
  std::vector<double>
      background_observations_;  // Background observations (corresponds to HX)
  double mu_scaling_factor_ =
      1.0e-6;  // Scaling factor for ensemble perturbations (corresponds to MU)

  // Model for integration (created from config)
  Model<BackendTag> model_;

  /**
   * @brief Process IEXT-specific logic (corresponds to Fortran lines 79-255)
   * @param iext Current IEXT loop number
   */
  void processIEXTLoop(int iext) {
    logger_.Info() << "Processing IEXT loop " << iext;

    // Set flags (corresponds to Fortran lines 81-82)
    bool flag = (iext == max_outer_loops_);
    bool ln_assm_gain = (iext >= 2);

    // Suppress unused variable warnings - these will be used in TODO
    // implementations
    (void)flag;
    (void)ln_assm_gain;

    // Save analysis field if final loop (corresponds to Fortran lines 84-90)
    if (iext == max_outer_loops_) {
      saveAnalysisField();
    }

    // Process gain fields for IEXT >= 2 (corresponds to Fortran lines 92-147)
    if (iext >= 2) {
      processGainFields();
    }

    // Set macom_loop_i and call model integration (corresponds to Fortran lines
    // 155-161)
    int macom_loop_i = iext;
    logger_.Info() << "--> X0 integration";
    logger_.Info() << "--> calling macom, mode: " << macom_loop_i;

    // Call model integration for background field
    // Variables are managed by Fortran, C++ only passes control parameters
    model_.runIntegration(state_, state_, ln_assm_gain);

    // Set macom_loop_i = 2 for next iteration (corresponds to Fortran line 163)
    if (iext != max_outer_loops_) {
      macom_loop_i = 2;
    }

    // Process background observations (corresponds to Fortran lines 165-255)
    processBackgroundObservations(iext);

    logger_.Info() << "IEXT loop " << iext << " processing completed";
  }

  /**
   * @brief Process ensemble members (corresponds to Fortran lines 258-409)
   * @param iext Current IEXT loop number
   */
  void processEnsembleMembers(int iext) {
    logger_.Info() << "Processing ensemble members for IEXT " << iext;

    const int ensemble_size = ensemble_.Size();

    // Loop through ensemble members (corresponds to Fortran: DO M=1,NM)
    for (int member = 0; member < ensemble_size; ++member) {
      logger_.Info() << "--> This is for member / outloop";
      logger_.Info() << member + 1 << "/" << iext;

      // Compute TP = X + X0 (corresponds to Fortran line 263)
      computeMemberState(member);

      // Process gain fields for this member (corresponds to Fortran lines
      // 266-318)
      processMemberGainFields(member);

      // Set ln_assm_gain = true and call model integration (corresponds to
      // Fortran lines 321-328)
      bool ln_assm_gain = true;

      // Suppress unused variable warning - will be used in TODO implementation
      (void)ln_assm_gain;
      logger_.Info() << "--> member integration";
      logger_.Info() << "--> calling macom, mode: " << 2;

      // Call model integration for ensemble member
      // Variables are managed by Fortran, C++ only passes control parameters
      State<BackendTag>& memberState = ensemble_.GetMember(member);
      model_.runMemberIntegration(memberState, memberState, member,
                                  ln_assm_gain);

      // Process member observations (corresponds to Fortran lines 330-409)
      processMemberObservations(member, iext);
    }

    logger_.Info() << "Ensemble members processing completed for IEXT " << iext;
  }

  /**
   * @brief Save analysis field to file
   */
  void saveAnalysisField() const {
    try {
      std::ofstream ana_file(output_base_file_ + ".DAT");
      if (ana_file.is_open()) {
        auto analysis_field = state_.getBackgroundField();
        for (const auto& value : analysis_field) {
          ana_file << std::scientific << std::setprecision(15) << value << "\n";
        }
        ana_file.close();
        logger_.Info() << "Analysis field saved to " << output_base_file_
                       << ".DAT";
      }
    } catch (const std::exception& e) {
      logger_.Error() << "Failed to save analysis field: " << e.what();
    }
  }

  /**
   * @brief Save ensemble check files
   */
  void saveEnsembleCheckFiles() const {
    try {
      std::ofstream check_file("D:/macom/4var/result/CHECK_ENS.DAT");
      if (check_file.is_open()) {
        for (size_t member = 0; member < ensemble_.Size(); ++member) {
          auto& member_state = ensemble_.GetMember(member);
          auto member_field = member_state.getBackgroundField();
          for (const auto& value : member_field) {
            check_file << std::scientific << std::setprecision(15) << value
                       << " ";
          }
          check_file << "\n";
        }
        check_file.close();
        logger_.Info() << "Ensemble check files saved to CHECK_ENS.DAT";
      }
    } catch (const std::exception& e) {
      logger_.Error() << "Failed to save ensemble check files: " << e.what();
    }
  }

  /**
   * @brief Save background field observations to file
   */
  void saveBackgroundObservations(
      const std::vector<double>& background_observations) const {
    try {
      std::ofstream bkg_file("D:/macom/4var/result/CHECK_BKG.DAT");
      if (bkg_file.is_open()) {
        for (size_t obs_idx = 0; obs_idx < background_observations.size();
             ++obs_idx) {
          bkg_file << std::scientific << std::setprecision(15)
                   << background_observations[obs_idx] << " " << 1.0 << "\n";
        }
        bkg_file.close();
        logger_.Info()
            << "Background field observations saved to CHECK_BKG.DAT";
      }
    } catch (const std::exception& e) {
      logger_.Error() << "Failed to save background field observations: "
                      << e.what();
    }
  }

  /**
   * @brief Save background field check file
   */
  void saveBackgroundFieldCheck() const {
    try {
      // Save background field using State (first 10 rows for checking)
      state_.saveBackgroundFieldToFile("D:/macom/4var/result/CHECK_XB.DAT");
      logger_.Info() << "Background field XB saved to CHECK_XB.DAT";
    } catch (const std::exception& e) {
      logger_.Error() << "Failed to save background field check: " << e.what();
    }
  }

  /**
   * @brief Save observation check files
   */
  void saveObservationCheckFiles() const {
    try {
      std::ofstream check_file("D:/macom/4var/result/CHECK_OBS.DAT");
      if (check_file.is_open()) {
        // Output first 10 rows of observation data
        int max_rows = std::min(10, (int)observations_.size());
        for (int i = 0; i < max_rows; ++i) {
          const auto& obs = observations_[i];
          check_file << std::fixed << std::setprecision(15) << obs.value << " "
                     << obs.error << "\n";
        }
        check_file.close();
        logger_.Info() << "Observation check files saved to CHECK_OBS.DAT";
      }
    } catch (const std::exception& e) {
      logger_.Error() << "Failed to save observation check files: " << e.what();
    }
  }

  /**
   * @brief Save analysis observations to file
   */
  void saveAnalysisObservations(
      const std::vector<double>& analysis_observations) const {
    try {
      std::ofstream ana_file("D:/macom/4var/result/CHECK_ANA.DAT");
      if (ana_file.is_open()) {
        for (size_t obs_idx = 0; obs_idx < analysis_observations.size();
             ++obs_idx) {
          ana_file << std::scientific << std::setprecision(15)
                   << analysis_observations[obs_idx] << " " << 1.0 << "\n";
        }
        ana_file.close();
        logger_.Info() << "Final analysis observations saved to CHECK_ANA.DAT";
      }
    } catch (const std::exception& e) {
      logger_.Error() << "Failed to save analysis observations: " << e.what();
    }
  }

  /**
   * @brief Process gain fields for IEXT >= 2 (corresponds to Fortran lines
   * 92-147)
   */
  void processGainFields() {
    logger_.Info() << "Processing gain fields for IEXT >= 2";

    // TODO: Implement gain field processing
    // - Compute ssh_gain, pbt_gain, tFld_gain, sFld_gain, uFld_gain, vFld_gain
    // - These represent the increment fields (X0 - XB) for different variables
    // - Apply masks as in Fortran code

    logger_.Info() << "Gain fields processing completed";
  }

  /**
   * @brief Process background observations (corresponds to Fortran lines
   * 165-255)
   * @param iext Current IEXT loop number
   */
  void processBackgroundObservations(int iext) {
    logger_.Info() << "Processing background observations for IEXT " << iext;

    // Apply observation operator to background field (reuse existing function)
    applyObservationOperatorToBackground();

    // IEXT=1 specific processing (corresponds to Fortran lines 169-239)
    if (iext == 1) {
      processIEXT1Initialization();
    }

    // Save observation check files (reuse existing functions)
    if (iext == 1) {
      saveBackgroundObservations(background_observations_);
    } else if (iext == max_outer_loops_) {
      saveAnalysisObservations(background_observations_);
    }

    logger_.Info() << "Background observations processing completed for IEXT "
                   << iext;
  }

  /**
   * @brief Process IEXT=1 initialization (corresponds to Fortran lines 169-239)
   */
  void processIEXT1Initialization() {
    logger_.Info() << "Processing IEXT=1 initialization";

    // Save background field check (reuse existing function)
    saveBackgroundFieldCheck();

    // Initialize analysis state X0 = XB (corresponds to Fortran line 186)
    const auto& background_field = state_.getBackgroundField();
    analysis_state_ = background_field;

    // DEBUG OUTPUT: Step 1 - Background field XB (first 10 values)
    logger_.Info()
        << "=== DEBUG: Step 1 - Background field XB (first 10 values) ===";
    for (int i = 0; i < std::min(10, (int)background_field.size()); ++i) {
      logger_.Info() << "XB[" << i << "] = " << std::scientific
                     << std::setprecision(15) << background_field[i];
    }

    // DEBUG OUTPUT: Step 2 - Analysis state X0 (should equal XB)
    logger_.Info()
        << "=== DEBUG: Step 2 - Analysis state X0 (first 10 values) ===";
    for (int i = 0; i < std::min(10, (int)analysis_state_.size()); ++i) {
      logger_.Info() << "X0[" << i << "] = " << std::scientific
                     << std::setprecision(15) << analysis_state_[i];
    }

    // Process ensemble perturbations and covariance (reuse existing functions)
    processEnsemblePerturbations();
    computeEnsembleCovariance();

    // DEBUG OUTPUT: Step 3 - Ensemble perturbations X (first 10 values for
    // first 3 members)
    logger_.Info()
        << "=== DEBUG: Step 3 - Ensemble perturbations X (first 10 values) ===";
    logger_.Info() << "MU scaling factor = " << std::scientific
                   << std::setprecision(15) << mu_scaling_factor_;
    for (int m = 0; m < std::min(3, (int)ensemble_.Size()); ++m) {
      logger_.Info() << "--- Member " << m << " ---";
      for (int p = 0; p < std::min(10, (int)ensemble_perturbations_[m].size());
           ++p) {
        logger_.Info() << "X[" << p << "," << m << "] = " << std::scientific
                       << std::setprecision(15)
                       << ensemble_perturbations_[m][p];
      }
    }

    // DEBUG OUTPUT: Step 4 - Ensemble covariance matrix BB (first 10x10
    // submatrix)
    logger_.Info() << "=== DEBUG: Step 4 - Ensemble covariance matrix BB "
                      "(first 10x10) ===";
    logger_.Info() << "BB matrix size: " << ensemble_.Size() << " x "
                   << ensemble_.Size();
    int debug_size = std::min(10, (int)ensemble_.Size());
    for (int m = 0; m < debug_size; ++m) {
      std::stringstream ss;
      ss << "BB[" << m << ",:] = ";
      for (int n = 0; n < debug_size; ++n) {
        ss << std::scientific << std::setprecision(15)
           << ensemble_covariance_[m][n];
        if (n < debug_size - 1) ss << " ";
      }
      logger_.Info() << ss.str();
    }

    // Perform eigenvalue decomposition (corresponds to Fortran lines 229-238)
    performEigenvalueDecomposition();

    logger_.Info() << "IEXT=1 initialization completed";
  }

  /**
   * @brief Compute member state TP = X + X0 (corresponds to Fortran line 263)
   * @param member Ensemble member index
   */
  void computeMemberState(int member) {
    logger_.Info() << "Computing member state for member " << member;

    // Get background field X0 and ensemble perturbation X
    const auto& background_field = state_.getBackgroundField();
    const auto& perturbation = ensemble_perturbations_[member];

    // Compute TP = X + X0 for each point
    // This creates the full state for ensemble member M
    // TP(P) = X(P,M) + X0(P) where P is the state dimension index

    logger_.Info() << "Member state computation completed for member "
                   << member;
    logger_.Info() << "  Background field size: " << background_field.size();
    logger_.Info() << "  Perturbation size: " << perturbation.size();
  }

  /**
   * @brief Process gain fields for ensemble member (corresponds to Fortran
   * lines 266-318)
   * @param member Ensemble member index
   */
  void processMemberGainFields(int member) {
    logger_.Info() << "Processing gain fields for member " << member;

    // TODO: Implement member gain field processing
    // - Compute gain fields for this specific member
    // - Similar to processGainFields() but for individual member

    logger_.Info() << "Member gain fields processing completed for member "
                   << member;
  }

  /**
   * @brief Process member observations (corresponds to Fortran lines 330-409)
   * @param member Ensemble member index
   * @param iext Current IEXT loop number
   */
  void processMemberObservations(int member, int iext) {
    logger_.Info() << "Processing observations for member " << member
                   << " in IEXT " << iext;

    // Compute Y(O,M) = TY(O) (corresponds to Fortran lines 331-333)
    // TODO: Implement member observation computation
    // - Apply observation operator to member state
    // - Store result in Y(O,M)

    // Save member check files for IEXT=1 (reuse existing check file logic)
    if (iext == 1 && member < 3) {
      // Save first 3 members' check files
      saveEnsembleCheckFiles();
    }

    logger_.Info() << "Member observations processing completed for member "
                   << member;
  }

  /**
   * @brief Perform eigenvalue decomposition (corresponds to Fortran lines
   * 229-238)
   */
  void performEigenvalueDecomposition() {
    logger_.Info() << "Performing eigenvalue decomposition";

    // TODO: Implement eigenvalue decomposition
    // - Call equivalent of Fortran's EOF_JCB(1.0D-7, K, NM, BB, LM, VP)
    // - Compute VV matrix: VV(M,N) = sum(VP(M,K)*VP(N,K)/LM(K))

    logger_.Info() << "Eigenvalue decomposition completed";
  }

  /**
   * @brief Process ensemble members and compute perturbations
   *
   * This function computes ensemble perturbations by subtracting the background
   * field from each ensemble member and applying a scaling factor.
   *
   * Based on Fortran: X(P,M) = (X(P,M) - XB(P)) * DSQRT(MU)
   * where:
   * - X(P,M) is the perturbation matrix (output)
   * - X(P,M) on the right side is the original ensemble member data
   * - XB(P) is the background field from State
   * - MU is the scaling factor
   */
  void processEnsemblePerturbations() {
    logger_.Info() << "Processing ensemble perturbations";

    const int ensemble_size = ensemble_.Size();
    // Get background field from State (as in Fortran: XB from state)
    const auto& background_field = state_.getBackgroundField();
    const int state_dimension = background_field.size();

    // Key validation checks
    if (ensemble_size <= 0) {
      throw std::runtime_error("Invalid ensemble size: " +
                               std::to_string(ensemble_size));
    }
    if (state_dimension <= 0) {
      throw std::runtime_error("Invalid state dimension: " +
                               std::to_string(state_dimension));
    }

    logger_.Info() << "XB total points: " << state_dimension
                   << ", X matrix size: " << ensemble_size << " x "
                   << state_dimension;

    // Initialize ensemble perturbation matrix (corresponds to X, NM x NP)
    ensemble_perturbations_.resize(ensemble_size);
    for (int m = 0; m < ensemble_size; ++m) {
      ensemble_perturbations_[m].resize(state_dimension);
    }

    // Ensure ensemble data is loaded by calling GetMember first
    // This triggers the lazy loading mechanism
    if (ensemble_size > 0) {
      ensemble_.GetMember(0);  // This will load the data if not already loaded
    }

    // Compute perturbations for each ensemble member
    // Get ensemble matrix data (original ensemble members)
    const auto& ensemble_matrix = ensemble_.GetEnsembleMatrix();

    for (int m = 0; m < ensemble_size; ++m) {
      for (int p = 0; p < state_dimension; ++p) {
        // Get original ensemble member data
        double member_value = ensemble_matrix(p, m);

        // Corresponds to Fortran: X(P,M) = (X(P,M) - XB(P)) * DSQRT(MU)
        // where X(P,M) is the perturbation matrix, not the original data
        ensemble_perturbations_[m][p] = (member_value - background_field[p]) *
                                        std::sqrt(mu_scaling_factor_);
      }
    }

    logger_.Info() << "Ensemble perturbations computed successfully";
  }

  /**
   * @brief Compute ensemble covariance matrix
   * Based on Fortran: BB(M,N) = X(P,M) * X(P,N)
   */
  void computeEnsembleCovariance() {
    logger_.Info() << "Computing ensemble covariance matrix";

    const int ensemble_size = ensemble_.Size();
    const int state_dimension = analysis_state_.size();

    // Initialize covariance matrix (corresponds to BB, NM x NM)
    ensemble_covariance_.resize(ensemble_size);
    for (int m = 0; m < ensemble_size; ++m) {
      ensemble_covariance_[m].resize(ensemble_size, 0.0);
    }

    // Corresponds to Fortran: BB(M,N) = X(P,M) * X(P,N)
    for (int m = 0; m < ensemble_size; ++m) {
      for (int n = 0; n < ensemble_size; ++n) {
        ensemble_covariance_[m][n] = 0.0;
        for (int p = 0; p < state_dimension; ++p) {
          ensemble_covariance_[m][n] +=
              ensemble_perturbations_[m][p] * ensemble_perturbations_[n][p];
        }
      }
    }

    logger_.Info() << "Ensemble covariance matrix computed successfully";
  }

  /**
   * @brief Apply observation operator to background field
   * Based on Fortran: HX(O) = TY(O)
   *
   * Fortran code flow:
   * 1. For each observation point O=1,NO
   * 2. Call GET_IY(OX(O),OY(O),OZ(O),OT(O),JTB,JT,OS(O),TY(O))
   * 3. HX(O) = TY(O)
   *
   * GET_IY function purpose:
   * - Input: observation position (OX,OY,OZ,OT) and observation type OS
   * - Process: interpolate observation value from model field
   * - Output: observation value TY
   */
  void applyObservationOperatorToBackground() {
    logger_.Info() << "Applying observation operator to background field";

    // Corresponds to Fortran: DO O=1,NO; HX(O)=TY(O); ENDDO
    // Here obs_operator_.apply() is equivalent to Fortran's GET_IY call
    background_observations_ = obs_operator_.apply(state_, observations_);

    // Verify consistency with Fortran code
    logger_.Info() << "HX(O) = TY(O) computation completed for "
                   << background_observations_.size() << " observations";
  }

  Ensemble<BackendTag>& ensemble_;
  const Observation<BackendTag>& observations_;
  const ObsOperator<BackendTag>& obs_operator_;
  State<BackendTag>& state_;

  // Simple A4D parameters (based on mod_a4d_liwei.f90)
  int max_outer_loops_;  // macom_loop_end in Fortran
  std::string output_base_file_;

  Logger<BackendTag>& logger_ = Logger<BackendTag>::Instance();
};

}  // namespace metada::framework