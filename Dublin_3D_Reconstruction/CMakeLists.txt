cmake_minimum_required(VERSION 3.14)

# 1. 项目名称
project(Dublin_3D_Reconstruction LANGUAGES CXX)

# 2. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 3. 寻找依赖库 (只找 PCL，不找 CGAL)
# ==========================================
find_package(PCL 1.10 REQUIRED)

# ==========================================
# 4. 全局包含路径与定义
# ==========================================
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ==========================================
# 5. 定义可执行文件
# ==========================================

# --- 模块 1: 预处理 (Preprocess) ---
add_executable(step1_preprocess src/01_preprocess.cpp)
target_link_libraries(step1_preprocess ${PCL_LIBRARIES})
if(MSVC)
    set_property(TARGET step1_preprocess PROPERTY LINK_FLAGS "/SUBSYSTEM:CONSOLE")
endif()

# --- 模块 2: 地面分割 (CSF Ground Segmentation) ---
add_executable(step2_ground src/02_ground_segmentation.cpp)
target_link_libraries(step2_ground ${PCL_LIBRARIES})
if(MSVC)
    set_property(TARGET step2_ground PROPERTY LINK_FLAGS "/SUBSYSTEM:CONSOLE")
endif()

# --- 模块 3 (方案A): 欧式聚类与规则分割 ---
# 这是你报告中提到的主方案之一，不需要 CGAL
add_executable(step3_rules src/03_segment_euclidean.cpp)
target_link_libraries(step3_rules ${PCL_LIBRARIES})
if(MSVC)
    set_property(TARGET step3_rules PROPERTY LINK_FLAGS "/SUBSYSTEM:CONSOLE")
endif()

# --- 模块 3 (方案B): 随机森林 (已暂时禁用) ---
# 下面的代码已被注释，不会触发 CGAL/GMP 报错
# add_executable(step3_rf src/03_segment_rf.cpp)
# target_link_libraries(step3_rf CGAL::CGAL ${PCL_LIBRARIES})

# --- 模块 4: 泊松重建 (Poisson Reconstruction) ---
add_executable(step4_reconstruction src/04_reconstruction_poisson.cpp)
target_link_libraries(step4_reconstruction ${PCL_LIBRARIES})
if(MSVC)
    set_property(TARGET step4_reconstruction PROPERTY LINK_FLAGS "/SUBSYSTEM:CONSOLE")
endif()