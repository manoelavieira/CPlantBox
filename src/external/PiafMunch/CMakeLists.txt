cmake_minimum_required(VERSION 3.15)
project(PiafMunch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(piafmunch STATIC
  PM_matrix.cpp
  PM_vector.cpp
  sparse_matrix.cpp
  index_vector.cpp
)
target_include_directories(piafmunch PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(PM_test tests/PM_test.cpp)
target_link_libraries(PM_test PRIVATE piafmunch)
