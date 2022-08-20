cmake_minimum_required(VERSION 3.16)

project(libtnnedi3cl LANGUAGES CXX)

add_library(nnedi3cl SHARED src/NNEDI3CL.cpp)

target_include_directories(nnedi3cl PRIVATE /usr/local/include/avisynth)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
if (build_type STREQUAL debug)
    target_compile_definitions(nnedi3cl PRIVATE DEBUG_BUILD)
else (build_type STREQUAL release)
    target_compile_definitions(nnedi3cl PRIVATE RELEASE_BUILD)
endif ()

message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")

target_compile_features(nnedi3cl PRIVATE cxx_std_17)

find_package(Boost REQUIRED filesystem system)

if (Boost_FOUND)
    target_include_directories(nnedi3cl PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(nnedi3cl ${Boost_LIBRARIES})
else()
    message(FATAL_ERROR "Required Boost packages not found. Perhaps add -DBOOST_ROOT?")
endif()

find_package(OpenCL REQUIRED)

if (OpenCL_FOUND)
    target_include_directories(nnedi3cl PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(nnedi3cl ${OpenCL_LIBRARIES})
else()
    message(FATAL_ERROR "Required OpenCL packages not found.")
endif()

target_link_libraries(nnedi3cl libavisynth.so)

find_package (Git)

if (GIT_FOUND)
    execute_process (COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        OUTPUT_VARIABLE ver
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set_target_properties(nnedi3cl PROPERTIES OUTPUT_NAME "nnedi3cl.${ver}")
else ()
    message (STATUS "GIT not found")
endif ()

include(GNUInstallDirs)

INSTALL(TARGETS nnedi3cl LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
