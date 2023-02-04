cmake_minimum_required(VERSION 3.22)
project(hough)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(hough hough.cpp)
target_link_libraries(hough ${OpenCV_LIBS})
install(TARGETS hough RUNTIME DESTINATION bin)
