cmake_minimum_required(VERSION 3.3)

# Find ROS build system
find_package(catkin QUIET COMPONENTS roscpp rosbag sensor_msgs cv_bridge)

# Describe ROS project
if (catkin_FOUND AND ENABLE_ROS)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp rosbag sensor_msgs cv_bridge
            INCLUDE_DIRS src/
            LIBRARIES ov_core_lib
    )
else ()
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
    include(GNUInstallDirs)
    set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/open_vins/")
endif ()

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}

)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
)

##################################################
# Build CUDA helper library (Reflect-101 padding)
##################################################
add_library(ov_core_cuda STATIC
        src/pzj/reflect101_pad.cu
)
set_target_properties(ov_core_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ov_core_cuda PUBLIC src/)  # 让 kernel 能 include 自己项目头

##################################################
# Make the core library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/cpi/CpiV1.cpp
        src/cpi/CpiV2.cpp
        src/sim/BsplineSE3.cpp
        src/track/TrackBase.cpp
        src/track/TrackAruco.cpp
        src/track/TrackDescriptor.cpp
        src/track/TrackKLT.cpp
        src/track/TrackSIM.cpp
        src/types/Landmark.cpp
        src/feat/Feature.cpp
        src/feat/FeatureDatabase.cpp
        src/feat/FeatureInitializer.cpp
        src/utils/print.cpp
        src/pzj/pyramid_gpu.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")

add_library(ov_core_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
# 1) 链接 CUDA Runtime（现代写法）
target_link_libraries(ov_core_lib
        PRIVATE
        ov_core_cuda
        ${CUDA_LIBRARIES}           # 若用旧式 find_package(CUDA) 就写 ${CUDA_LIBRARIES}
        ${thirdparty_libraries}
)
# 2) 把 CUDA 头加入 ov_core_lib（保险）
target_include_directories(ov_core_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

target_include_directories(ov_core_lib PUBLIC src/)
install(TARGETS ov_core_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)



##################################################
# Make binary files!
##################################################

if (catkin_FOUND AND ENABLE_ROS)

    add_executable(test_tracking src/test_tracking.cpp)
    target_link_libraries(test_tracking ov_core_lib ${thirdparty_libraries})
    install(TARGETS test_tracking
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

endif ()

add_executable(test_webcam src/test_webcam.cpp)
target_link_libraries(test_webcam ov_core_lib ${thirdparty_libraries})
install(TARGETS test_webcam
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(test_profile src/test_profile.cpp)
target_link_libraries(test_profile ov_core_lib ${thirdparty_libraries})
install(TARGETS test_profile
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



