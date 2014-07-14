cmake_minimum_required(VERSION 2.8)

project(spvis)

FILE(GLOB SOURCES src/*.cc)
add_executable(spvis ${SOURCES})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-std=c++11)
endif()

include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
target_link_libraries(spvis ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
	${GLUT_glut_LIBRARY})

