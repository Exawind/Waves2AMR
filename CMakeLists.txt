cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(Waves2AMR LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(WAVES2AMR_ENABLE_MPI "Enable MPI" OFF)

set(WAVES2AMR_GPU_BACKEND_VALUES NONE SYCL CUDA HIP)
set(WAVES2AMR_GPU_BACKEND NONE CACHE STRING "On-node, accelerated GPU backend: <NONE,SYCL,CUDA,HIP>")
set_property(CACHE WAVES2AMR_GPU_BACKEND PROPERTY STRINGS ${WAVES2AMR_GPU_BACKEND_VALUES})
if (NOT WAVES2AMR_GPU_BACKEND IN_LIST WAVES2AMR_GPU_BACKEND_VALUES)
   message(FATAL_ERROR "WAVES2AMR_GPU_BACKEND=${WAVES2AMR_GPU_BACKEND} is not allowed."
      " Must be one of ${WAVES2AMR_GPU_BACKEND_VALUES}")
endif ()

if (NOT TARGET AMReX::amrex)
   set(AMREX_REQUIRED_COMPONENTS 3D DOUBLE)
   if (WAVES2AMR_ENABLE_MPI)
      list(APPEND AMREX_REQUIRED_COMPONENTS MPI)
   endif ()
   if (NOT WAVES2AMR_GPU_BACKEND STREQUAL "NONE")
      list(APPEND AMREX_REQUIRED_COMPONENTS ${WAVES2AMR_GPU_BACKEND})
   endif ()
   find_package(AMReX CONFIG REQUIRED ${AMREX_REQUIRED_COMPONENTS} )
endif ()

if (WAVES2AMR_GPU_BACKEND STREQUAL "CUDA")
   include(AMReXTargetHelpers)
endif ()

add_library(waves_2_amr OBJECT)
target_link_libraries(waves_2_amr PUBLIC AMReX::amrex)

find_package(FFTW REQUIRED)
target_link_libraries_system(waves_2_amr PUBLIC FFTW::FFTW)

add_subdirectory(src)
add_subdirectory(include)

if (WAVES2AMR_GPU_BACKEND STREQUAL "CUDA")
   setup_target_for_cuda_compilation(waves_2_amr)
endif ()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

add_library(waves_2_amr_api)
target_link_libraries(waves_2_amr_api PUBLIC waves_2_amr)
add_library(${PROJECT_NAME}::waves_2_amr_api ALIAS waves_2_amr)

get_target_property(WAVES2AMR_INCLUDES waves_2_amr SOURCES)
list(FILTER WAVES2AMR_INCLUDES INCLUDE REGEX "\\.h")
set_target_properties(
  waves_2_amr PROPERTIES PUBLIC_HEADER "${WAVES2AMR_INCLUDES}")

install(
  TARGETS waves_2_amr_api waves_2_amr
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin
  ARCHIVE       DESTINATION lib
  LIBRARY       DESTINATION lib
  INCLUDES      DESTINATION include
  PUBLIC_HEADER DESTINATION include
  )

install(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
configure_package_config_file(
  ${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
