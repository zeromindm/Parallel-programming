cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

set(CMAKE_CXX_STANDARD 11)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

add_executable(generator src/matrix_generator.cpp)
add_executable(multiplier src/matrix_multiplier.cpp)

set_target_properties(generator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
set_target_properties(multiplier PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)