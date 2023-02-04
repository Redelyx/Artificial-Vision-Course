cmake_minimum_required(VERSION 2.8)
project(simple)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#add_definitions("-std=c++11 -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(simple a1.cpp)
target_link_libraries(simple ${OpenCV_LIBS})
install(TARGETS simple RUNTIME DESTINATION bin)


