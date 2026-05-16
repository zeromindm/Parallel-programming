cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

set(CMAKE_CXX_STANDARD 11)
find_package(MPI REQUIRED)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

add_executable(generator src/matrix_generator.cpp)
set_target_properties(generator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

add_executable(multiplier_mpi src/matrix_multiplier_mpi.cpp)
target_link_libraries(multiplier_mpi MPI::MPI_CXX)
set_target_properties(multiplier_mpi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
