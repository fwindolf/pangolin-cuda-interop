CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

# == Dependecies =============================================

FIND_PACKAGE(CUDA REQUIRED)

FIND_PACKAGE(OpenCV COMPONENTS core highgui imgproc REQUIRED)

# == From third party directory

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/Eigen)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/Pangolin/include)
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/Pangolin/lib)

SET(DEPENDENCIES
    opencv_core
    opencv_highgui
    pangolin
    glut
    GLU
    GL
    GLEW
)

SET(APP "interop")
SET(MAIN "main.cpp")
SET(KERNEL "kernel.cu")

CUDA_ADD_EXECUTABLE(${APP}
    ${MAIN}
    ${KERNEL}
)

TARGET_LINK_LIBRARIES(${APP}
    ${DEPENDENCIES}
)

TARGET_COMPILE_OPTIONS(${APP}
	PRIVATE -std=c++11 
    PRIVATE -Wno-deprecated-declarations
)