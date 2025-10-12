# Find required dependencies for MACOM
# Ensure FindMPI is robust. It should set MPI_FOUND and imported targets like MPI::MPI_C, MPI::MPI_Fortran.

# =============================================================================
# MACOM MODE SELECTION - MUTUALLY EXCLUSIVE OPTIONS
# =============================================================================
option(MACOM_GLOBAL_MODE "Build MACOM in global mode" OFF)
option(MACOM_REGION_MODE "Build MACOM in region mode" OFF)
option(MACOM_4VAR_DA_MODE "Build MACOM in 4D-Var DA mode" ON)

# Check if any MACOM mode is enabled
set(MACOM_MODE_ENABLED FALSE)
if(MACOM_REGION_MODE OR MACOM_GLOBAL_MODE OR MACOM_4VAR_DA_MODE)
    set(MACOM_MODE_ENABLED TRUE)
endif()

# Ensure only one mode is selected if any mode is enabled
if((MACOM_REGION_MODE AND MACOM_GLOBAL_MODE) OR 
   (MACOM_REGION_MODE AND MACOM_4VAR_DA_MODE) OR 
   (MACOM_GLOBAL_MODE AND MACOM_4VAR_DA_MODE))
    message(FATAL_ERROR "MACOM modes are mutually exclusive. Please select only one.")
endif()

# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================
if(MACOM_MODE_ENABLED)
    metada_find_package(MPI REQUIRED)
    metada_find_package(HDF5 REQUIRED)
    metada_find_package(CURL REQUIRED)
else()
    # When no MACOM mode is enabled, make MPI optional
    metada_find_package(MPI OPTIONAL QUIET)
    metada_find_package(HDF5 OPTIONAL QUIET)
    metada_find_package(CURL OPTIONAL QUIET)
endif()

# Platform-specific NetCDF configuration
if(WIN32)
    # Find NetCDF Fortran for MSYS2/MINGW64
    set(NetCDF_Fortran_ROOT "C:/msys64/mingw64")
    set(NetCDF_Fortran_INCLUDE_DIRS "${NetCDF_Fortran_ROOT}/include")
    set(NetCDF_Fortran_LIBRARIES "${NetCDF_Fortran_ROOT}/lib/libnetcdff.dll.a")
else()
    # Linux/Unix platforms
    set(NetCDF_Fortran_INCLUDE_DIRS "/usr/include")
    set(NetCDF_Fortran_LIBRARIES "/usr/lib/x86_64-linux-gnu/libnetcdff.so")
    # Ensure variables are properly set
    if(NOT HDF5_LIBRARIES)
        set(HDF5_INCLUDE_DIRS /usr/include)
        set(HDF5_LIBRARIES /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so)
    endif()
    if(NOT CURL_LIBRARIES)
        set(CURL_INCLUDE_DIRS /usr/include)
        set(CURL_LIBRARIES /usr/lib/x86_64-linux-gnu/libcurl.so)
    endif()
endif()

# Find other dependencies
if(MACOM_MODE_ENABLED)
    metada_find_package(xtensor CONFIG)
    metada_find_package(xtl CONFIG)
    metada_find_package(nanoflann CONFIG)
    metada_find_package(netcdf-cxx4)
    metada_find_package(netCDF CONFIG)
else()
    metada_find_package(xtensor CONFIG OPTIONAL QUIET)
    metada_find_package(xtl CONFIG OPTIONAL QUIET)
    metada_find_package(nanoflann CONFIG OPTIONAL QUIET)
    metada_find_package(netcdf-cxx4 OPTIONAL QUIET)
    metada_find_package(netCDF CONFIG OPTIONAL QUIET)
endif()

# =============================================================================
# INTERFACE LIBRARY SETUP
# =============================================================================
# Create the MACOM backend interface library
add_library(macom_backends INTERFACE)
add_library(metada::backends::macom ALIAS macom_backends)

# Set include directories for the interface library
target_include_directories(macom_backends 
    INTERFACE  
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Set headers as interface sources to show them in IDE
target_sources(macom_backends
    INTERFACE
        MACOMModel.hpp
        MACOMGeometry.hpp
        MACOMGeometryIterator.hpp
        MACOMState.hpp 
)

# Link dependencies for C++ interface
target_link_libraries(macom_backends 
    INTERFACE
        metada::base
        NetCDF::CXX4
        nanoflann::nanoflann
)

# =============================================================================
# FORTRAN COMPILATION (ONLY IF MACOM MODE ENABLED)
# =============================================================================
if(MACOM_MODE_ENABLED)
    # Set Fortran compiler
    set(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})

    # Set compiler flags
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -cpp -fno-tree-vectorize -fallow-argument-mismatch -Wno-missing-include-dirs")

    # =============================================================================
    # MODE-SPECIFIC CONFIGURATION
    # =============================================================================
    
    # GLOBAL MODE CONFIGURATION
    if(MACOM_GLOBAL_MODE)
        message(STATUS "Building MACOM in GLOBAL mode")
        
        # Global mode: use src-globe files (exclude duplicate mpi_interfaces)
        file(GLOB SRC_FORTRAN_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/fortran/src-globe/*.f90"
        )
        # # Remove the utf8 version to avoid module conflict
        # list(REMOVE_ITEM SRC_FORTRAN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/fortran/src-globe/mod_mpi_interfaces_utf8.f90")
        # message(STATUS "Using global mode Fortran sources from src-globe/ (excluding mod_mpi_interfaces_utf8.f90)")
        
        # Global mode wrapper
        set(MACOM_WRAPPER_FILE "fortran/macom_fortran_wrapper_globe.f90")
        message(STATUS "Using globe wrapper: macom_fortran_wrapper_globe.f90")
        
        # Global mode compile definitions
        add_compile_definitions(MACOM_GLOBAL_MODE=1)
        
        # Sea ice options for global mode
        option(SEAICE_VARIABLE_SALINITY "Variable salinity" ON)
        option(SEAICE_ITD "Sea Ice Thickness Distribution" ON)
        option(SEAICE_ALLOW_FREEDRIFT "Enable free drift code" ON)
        option(SEAICE_ALLOW_DYNAMICS "Allow sea ice dynamic code" ON)
        option(SEAICE_ALLOW_MOM_ADVECTION "Enable advection terms" OFF)
        option(SEAICE_ALLOW_BOTTOMDRAG "Use parameterization of grounding ice" ON)
        option(SEAICE_CAP_SUBLIM "Used in thermodyn module" ON)
        option(OPENACC "Enable OpenACC" OFF)
        option(OPENACCGPU "Enable OpenACC GPU" OFF)
        option(SeaiceDebug "Enable Seaice Debug" OFF)
        
    # REGION MODE CONFIGURATION
    elseif(MACOM_REGION_MODE)
        message(STATUS "Building MACOM in REGION mode (sea ice auto-disabled)")
        
        # Region mode: use src-region files
        file(GLOB SRC_FORTRAN_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/fortran/src-region/*.f90"
        )
        message(STATUS "Using region mode Fortran sources from src-region/")
        
        # Region mode wrapper
        set(MACOM_WRAPPER_FILE "fortran/macom_fortran_wrapper.f90")
        message(STATUS "Using standard wrapper: macom_fortran_wrapper.f90")
        
        # Region mode compile definitions
        add_compile_definitions(BARMOD)               # For Fortran code
        add_compile_definitions(MACOM_REGION_MODE=1)  # For C++ code
        
        # Auto-disable sea ice options in region mode
        set(SEAICE_VARIABLE_SALINITY OFF)
        set(SEAICE_ITD OFF)
        set(SEAICE_ALLOW_FREEDRIFT OFF)
        set(SEAICE_ALLOW_DYNAMICS OFF)
        set(SEAICE_ALLOW_MOM_ADVECTION OFF)
        set(SEAICE_ALLOW_BOTTOMDRAG OFF)
        set(SEAICE_CAP_SUBLIM OFF)
        set(SeaiceDebug OFF)
        
    # 4D-VAR DA MODE CONFIGURATION
    elseif(MACOM_4VAR_DA_MODE)
        message(STATUS "Building MACOM in 4D-Var DA mode (sea ice auto-disabled)")
        
        # 4D-Var DA mode: use src-4var-da files
        file(GLOB SRC_FORTRAN_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/fortran/src-4var-da/*.f90"
        )
        message(STATUS "Using 4D-Var DA mode Fortran sources from src-4var-da/")
        
        # 4D-Var DA mode wrapper
        set(MACOM_WRAPPER_FILE "fortran/macom_fortran_wrapper_4var.f90")
        message(STATUS "Using 4D-Var DA wrapper: macom_fortran_wrapper_4var.f90")
        
        # 4D-Var DA mode compile definitions
        add_compile_definitions(MACOM_4VAR_DA_MODE=1) # For C++ and Fortran code
        
        # Auto-disable sea ice options in 4D-Var DA mode
        set(SEAICE_VARIABLE_SALINITY OFF)
        set(SEAICE_ITD OFF)
        set(SEAICE_ALLOW_FREEDRIFT OFF)
        set(SEAICE_ALLOW_DYNAMICS OFF)
        set(SEAICE_ALLOW_MOM_ADVECTION OFF)
        set(SEAICE_ALLOW_BOTTOMDRAG OFF)
        set(SEAICE_CAP_SUBLIM OFF)
        set(SeaiceDebug OFF)
    endif()

    # =============================================================================
    # FORTRAN COMPILE DEFINITIONS (COMMON FOR ALL MODES)
    # =============================================================================
    # Set Fortran preprocessor macro definitions
    set(DEF_FLAGS "")

    # Sea ice options - available in all modes but can be disabled
    if(SEAICE_VARIABLE_SALINITY)
        list(APPEND DEF_FLAGS "-DSEAICE_VARIABLE_SALINITY")
        add_compile_definitions(SEAICE_VARIABLE_SALINITY)
    endif()
    if(SEAICE_ITD)
        list(APPEND DEF_FLAGS "-DSEAICE_ITD")
        add_compile_definitions(SEAICE_ITD)
    endif()
    if(SEAICE_ALLOW_FREEDRIFT)
        list(APPEND DEF_FLAGS "-DSEAICE_ALLOW_FREEDRIFT")
        add_compile_definitions(SEAICE_ALLOW_FREEDRIFT)
    endif()
    if(SEAICE_ALLOW_DYNAMICS)
        list(APPEND DEF_FLAGS "-DSEAICE_ALLOW_DYNAMICS")
        add_compile_definitions(SEAICE_ALLOW_DYNAMICS)
    endif()
    if(SEAICE_ALLOW_MOM_ADVECTION)
        list(APPEND DEF_FLAGS "-DSEAICE_ALLOW_MOM_ADVECTION")
        add_compile_definitions(SEAICE_ALLOW_MOM_ADVECTION)
    endif()
    if(SEAICE_ALLOW_BOTTOMDRAG)
        list(APPEND DEF_FLAGS "-DSEAICE_ALLOW_BOTTOMDRAG")
        add_compile_definitions(SEAICE_ALLOW_BOTTOMDRAG)
    endif()
    if(SEAICE_CAP_SUBLIM)
        list(APPEND DEF_FLAGS "-DSEAICE_CAP_SUBLIM")
        add_compile_definitions(SEAICE_CAP_SUBLIM)
    endif()
    if(OPENACC)
        list(APPEND DEF_FLAGS "-DOPENACC")
        add_compile_definitions(OPENACC)
    endif()
    if(OPENACCGPU)
        list(APPEND DEF_FLAGS "-DOPENACCGPU")
        add_compile_definitions(OPENACCGPU)
    endif()
    if(SeaiceDebug)
        list(APPEND DEF_FLAGS "-DSeaiceDebug")
        add_compile_definitions(SeaiceDebug)
    endif()

    string(REPLACE ";" " " DEF_FLAGS_STR "${DEF_FLAGS}")

    # =============================================================================
    # FORTRAN LIBRARY CREATION
    # =============================================================================
    # Ensure Fortran module directory is explicitly set - use mode-specific directory
    if(MACOM_GLOBAL_MODE)
        set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules-globe)
    elseif(MACOM_REGION_MODE)
        set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules-region)
    elseif(MACOM_4VAR_DA_MODE)
        set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules-4var)
    endif()
    
    # Clear any existing module files to avoid conflicts
    file(REMOVE_RECURSE ${CMAKE_Fortran_MODULE_DIRECTORY})
    file(MAKE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY})
    
    # Create Fortran implementation library
    add_library(macom_fortran STATIC
        fortran/macom_logger.f90           
        ${MACOM_WRAPPER_FILE}
        ${SRC_FORTRAN_FILES}
    )

    # Set Fortran module output directory
    set_property(TARGET macom_fortran PROPERTY Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY})

    # Add module directory to include paths
    target_include_directories(macom_fortran PUBLIC
        ${CMAKE_Fortran_MODULE_DIRECTORY}
        ${NetCDF_Fortran_INCLUDE_DIRS}
        $<IF:$<NOT:$<BOOL:${WIN32}>>,${HDF5_INCLUDE_DIRS};${CURL_INCLUDE_DIRS},>
    )

    # Set Fortran compiler flags and macros
    target_compile_options(macom_fortran PUBLIC
        $<$<COMPILE_LANGUAGE:Fortran>:${DEF_FLAGS}>
    )

    # Link Fortran dependencies
    if(WIN32)
        target_link_libraries(macom_fortran PUBLIC HDF5 CURL)
    else()
        target_link_libraries(macom_fortran PUBLIC ${HDF5_LIBRARIES} ${CURL_LIBRARIES})
    endif()

    target_link_libraries(macom_fortran PUBLIC
        ${NetCDF_Fortran_LIBRARIES}
        MPI::MPI_Fortran
    )

    # =============================================================================
    # C++ IMPLEMENTATION LIBRARY
    # =============================================================================
    # Create C++ implementation library that links with Fortran
    add_library(macom_impl STATIC
        src/MACOMFortranInterface.cpp
    )

    # Set include directories for implementation
    target_include_directories(macom_impl
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Link C++ implementation with Fortran implementation and dependencies
    target_link_libraries(macom_impl PUBLIC
        macom_fortran
    )

    # Link interface with implementation
    target_link_libraries(macom_backends INTERFACE macom_impl)

else()
    # =============================================================================
    # NO MACOM MODE ENABLED - C++ INTERFACE WITH EMPTY FORTRAN STUBS
    # =============================================================================
    message(STATUS "MACOM modes disabled - building C++ interface with empty Fortran stubs")
    
    # Create empty Fortran library with stub implementations
    add_library(macom_fortran STATIC
        fortran/macom_fortran_wrapper_empty.f90
    )
    
    # Set Fortran module output directory
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules-empty)
    file(REMOVE_RECURSE ${CMAKE_Fortran_MODULE_DIRECTORY})
    file(MAKE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY})
    set_property(TARGET macom_fortran PROPERTY Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY})
    
    # Create C++ implementation library that links with empty Fortran
    add_library(macom_impl STATIC
        src/MACOMFortranInterface.cpp
    )

    # Set include directories for implementation
    target_include_directories(macom_impl
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Link C++ implementation with empty Fortran implementation
    target_link_libraries(macom_impl PUBLIC
        macom_fortran
    )
    
    # No MPI needed in empty mode - explicitly disable MPI
    set(MPI_FOUND FALSE)
    message(STATUS "MACOM empty mode - MPI disabled")

    # Link interface with implementation
    target_link_libraries(macom_backends INTERFACE macom_impl)
endif()

# =============================================================================
# FORMATTING TARGETS
# =============================================================================
# Add format targets
AddFormatTarget(macom_backends ${CMAKE_CURRENT_SOURCE_DIR})

