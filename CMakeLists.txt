cmake_minimum_required(VERSION 3.22)
project(DualReg)

set(CMAKE_CXX_STANDARD 14)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(DualReg main.cpp funcs.cpp PCR.cpp registration.cpp Eva.h)

target_link_libraries(DualReg ${PCL_LIBRARIES})
