# CMake Requirement
cmake_minimum_required(VERSION 3.15)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Setup project
project(AstraSim_G2)

# Compilation target
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/g2)")


# Include src files to compile
file(GLOB srcs_common
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
)

file(GLOB srcs_g2
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

# Compile G2
if (BUILDTARGET STREQUAL "all" OR BUILDTARGET STREQUAL "g2")
    add_executable(AstraSim_G2_congestion ${srcs_g2} ${srcs_common})
    target_sources(AstraSim_G2_congestion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)

    # Copy python script to bin directory
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/network_backend/g2/network.py
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../bin/)

    # Link libraries
    target_link_libraries(AstraSim_G2_congestion LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_G2_congestion LINK_PRIVATE G2)

    # Include directories
    target_include_directories(AstraSim_G2_congestion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_G2_congestion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_G2_congestion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)

    # Properties
    # TODO: Switch to OFF after binary_function deprecation has been resolved
    set_target_properties(AstraSim_G2_congestion PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_G2_congestion
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
endif ()
