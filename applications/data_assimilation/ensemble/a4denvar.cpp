/**
 * @file a4denvar.cpp
 * @brief Driver program for the Analytical Four-Dimensional
 * Ensemble-Variational (A4DEnVar) data assimilation algorithm
 * @details This application implements the analytical four-dimensional
 * ensemble-variational data assimilation algorithm as described in Liang et al.
 * (2021). It reads configuration from a file, initializes required components
 * like ensemble members, observations and observation operators across multiple
 * time windows, and performs the analysis step with an analytical solution.
 *
 * The A4DEnVar combines the strengths of 4D-Var (temporal consistency) and
 * ensemble methods (flow-dependent error covariance) with an analytical
 * solution that avoids iterative minimization.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 *            Expected format: a4denvar <config_file>
 * @return 0 on success, 1 on failure
 */

#include "A4DEnVar.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

#include "ApplicationContext.hpp"
#include "Config.hpp"
#include "Ensemble.hpp"
#include "Geometry.hpp"
#include "MACOMBackendTraits.hpp"
#include "MACOMFortranInterface.hpp"
#include "Model.hpp"
#include "ObsOperator.hpp"
#include "Observation.hpp"
#include "State.hpp"

namespace fwk = metada::framework;
// using BackendTag = metada::traits::SimpleBackendTag;
using BackendTag = metada::traits::MACOMBackendTag;

int main(int argc, char* argv[]) {
  try {
    // Validate command line arguments
    if (argc != 2) {
      std::cerr << "Usage: a4denvar <config_file>" << std::endl;
      return 1;
    }

    // Initialize application context
    auto context = fwk::ApplicationContext<BackendTag>(argc, argv);
    auto& logger = context.getLogger();
    auto& config = context.getConfig();

    logger.Info() << "Starting A4DEnVar Data Assimilation Application";

    // // Initialize observations
    // fwk::Observation<BackendTag> observations(
    //     config.GetSubsection("observations"));
    // logger.Info() << "Loaded " << observations.size() << " observations";

    // // Initialize observation operator
    // fwk::ObsOperator<BackendTag> obs_operator(
    //     config.GetSubsection("obs_operator"));

    // // Initialize geometry
    // fwk::Geometry<BackendTag> geometry(config.GetSubsection("geometry"));

    // // Initialize state
    // fwk::State<BackendTag> state(config.GetSubsection("state"), geometry);

    // Initialize model (Fortran handles configuration internally)
    // In 4D-Var DA mode, the Fortran main program is called automatically in
    // constructor
    // fwk::Model<BackendTag> model(config.GetSubsection("model"));
    // logger.Info() << "Model initialized successfully";

    // Call the MACOM 4D-Var DA main function
    logger.Info() << "Calling MACOM 4D-Var DA main function";
    c_macom_4var_da_main();
    logger.Info() << "MACOM 4D-Var DA main function completed";

    // // Initialize ensemble
    // fwk::Ensemble<BackendTag> ensemble(config.GetSubsection("ensemble"),
    //                                    geometry);
    // logger.Info() << "Loaded ensemble with " << ensemble.Size() << "
    // members";

    // // Run A4DEnVar algorithm
    // logger.Info() << "Starting A4DEnVar algorithm";
    // fwk::A4DEnVar<BackendTag> a4denvar(ensemble, observations, obs_operator,
    //                                    state,
    //                                    config.GetSubsection("analysis"));

    // // t₀ Phase: Initialize analysis components (data reading and setup)
    // logger.Info() << "=== t₀ Phase: Data Reading and Initialization ===";
    // a4denvar.initializeAnalysis();

    // // IEXT Phase: Run ensemble outer loop (main A4DEnVar algorithm)
    // logger.Info() << "=== IEXT Phase: Main A4DEnVar Algorithm ===";
    // a4denvar.runEnsembleOuterLoop();

    // // Final Phase: Save results
    // logger.Info() << "=== Final Phase: Saving Results ===";
    // a4denvar.saveEnsemble();

    logger.Info() << "A4DEnVar algorithm completed successfully";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error in MACOM application: " << e.what() << std::endl;
    return 1;
  }
}