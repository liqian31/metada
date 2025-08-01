# Config tests
setup_test_executable(config_test ConfigTest.cpp)
add_test_format_target(config_test ${CMAKE_CURRENT_SOURCE_DIR})

# Logger tests
setup_test_executable(logger_test LoggerTest.cpp)
add_test_format_target(logger_test ${CMAKE_CURRENT_SOURCE_DIR})

# Geometry tests
setup_test_executable(geometry_test GeometryTest.cpp)
add_test_format_target(geometry_test ${CMAKE_CURRENT_SOURCE_DIR})

# Geometry iterator tests
setup_test_executable(geometry_iterator_test GeometryIteratorTest.cpp)
add_test_format_target(geometry_iterator_test ${CMAKE_CURRENT_SOURCE_DIR})

# State tests
setup_test_executable(state_test StateTest.cpp)
add_test_format_target(state_test ${CMAKE_CURRENT_SOURCE_DIR})

# Model tests
setup_test_executable(model_test ModelTest.cpp)
add_test_format_target(model_test ${CMAKE_CURRENT_SOURCE_DIR})

# Ensemble tests
setup_test_executable(ensemble_test EnsembleTest.cpp)
add_test_format_target(ensemble_test ${CMAKE_CURRENT_SOURCE_DIR})

# Increment tests
setup_test_executable(increment_test IncrementTest.cpp)
add_test_format_target(increment_test ${CMAKE_CURRENT_SOURCE_DIR})

# Location tests
setup_test_executable(location_test LocationTest.cpp)
add_test_format_target(location_test ${CMAKE_CURRENT_SOURCE_DIR})

# IObservations tests
setup_test_executable(observation_test ObservationTest.cpp)
add_test_format_target(observation_test ${CMAKE_CURRENT_SOURCE_DIR})

# Observation iterator tests
setup_test_executable(observation_iterator_test ObservationIteratorTest.cpp)
add_test_format_target(observation_iterator_test ${CMAKE_CURRENT_SOURCE_DIR})

# IObsOperator tests
setup_test_executable(obsoperator_test ObsOperatorTest.cpp)
add_test_format_target(obsoperator_test ${CMAKE_CURRENT_SOURCE_DIR})