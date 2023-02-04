cmake_minimum_required(VERSION 3.0)
project(simple)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(simple simple.cpp)
target_link_libraries(simple ${OpenCV_LIBS})
install(TARGETS simple RUNTIME DESTINATION bin)


#add_executable(test test.cpp)
#target_link_libraries(test ${OpenCV_LIBS})
#install(TARGETS test RUNTIME DESTINATION bin)
