#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <queue>
#include <limits>

// PCL相关头文件
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace csf {

    // 点结构体（与原始CSF相同）
    struct Point {
        double x, y, z;
        int index;
        Point() : x(0), y(0), z(0), index(-1) {}
        Point(double _x, double _y, double _z, int _idx = -1)
            : x(_x), y(_y), z(_z), index(_idx) {}
    };

    // 布料网格节点（与原始CSF相同）
    struct Node {
        double x, y, z;          // 当前位置
        double tmp_x, tmp_y, tmp_z; // 临时位置
        bool movable;            // 是否可移动
        int index_x, index_y;    // 网格索引

        Node() : x(0), y(0), z(0), tmp_x(0), tmp_y(0), tmp_z(0),
            movable(true), index_x(0), index_y(0) {}
        Node(double _x, double _y, double _z)
            : x(_x), y(_y), z(_z), tmp_x(_x), tmp_y(_y), tmp_z(_z),
            movable(true), index_x(0), index_y(0) {}
    };

    // CSF参数（与CloudCompare相同）
    struct Parameters {
        // 主要参数
        double cloth_resolution = 1.0;     // 布料网格分辨率 (cloth_resolution)
        double max_cloth_height = 0.1;     // 最大布料高度 (max_cloth_height)
        double class_threshold = 0.5;      // 分类阈值 (class_threshold)
        int rigidness = 2;                 // 布料刚性 (rigidness): 1-柔软, 2-中等, 3-刚性
        int iterations = 500;              // 迭代次数 (iterations)
        double time_step = 0.65;           // 时间步长 (time_step)

        // 高级参数
        bool slope_smooth = true;          // 坡度平滑 (slope_smooth)
        double max_slope = 45.0;           // 最大坡度 (max_slope in degrees)
        double smoothing_threshold = 0.1;  // 平滑阈值

        // 布料扩展参数
        double cloth_buffer = 2.0;         // 布料边界扩展
        int border_padding = 3;            // 边界填充

        // 预处理参数
        bool remove_outliers = false;      // 是否移除离群点
        double outlier_threshold = 1.0;    // 离群点阈值

        // 后处理参数
        bool postprocessing = true;        // 是否后处理
    };

    // CloudCompare CSF算法实现
    class CloudCompareCSF {
    private:
        Parameters params_;
        std::vector<Point> points_;
        std::vector<std::vector<Node>> cloth_grid_;
        std::vector<double> ground_height_map_;
        double min_x_, min_y_, max_x_, max_y_;
        int grid_width_, grid_height_;
        double cell_size_;

    public:
        CloudCompareCSF(const Parameters& params = Parameters()) : params_(params) {}

        // 主要过滤函数（与CloudCompare接口类似）
        void filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
            pcl::PointCloud<pcl::PointXYZ>::Ptr ground,
            pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground) {

            std::cout << "========================================" << std::endl;
            std::cout << "CloudCompare CSF算法实现" << std::endl;
            std::cout << "========================================" << std::endl;

            auto total_start = std::chrono::high_resolution_clock::now();

            // 步骤1: 导入点云
            importCloud(cloud);

            // 步骤2: 初始化布料
            initializeCloth();

            // 步骤3: 执行布料模拟
            clothSimulation();

            // 步骤4: 计算地面高度图
            computeGroundHeight();

            // 步骤5: 分类点云
            classifyPoints(cloud, ground, non_ground);

            // 步骤6: 后处理
            if (params_.postprocessing) {
                postProcess(ground, non_ground);
            }

            auto total_end = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                total_end - total_start);

            std::cout << "总处理时间: " << total_duration.count() / 1000.0 << " 秒" << std::endl;
        }

        void setParameters(const Parameters& params) {
            params_ = params;
        }

    private:
        // 步骤1: 导入点云
        void importCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
            std::cout << "\n[1/6] 导入点云..." << std::endl;

            points_.clear();
            points_.reserve(cloud->size());

            // 获取边界
            min_x_ = min_y_ = std::numeric_limits<double>::max();
            max_x_ = max_y_ = -std::numeric_limits<double>::max();

            for (size_t i = 0; i < cloud->size(); ++i) {
                const auto& p = cloud->points[i];
                points_.push_back(Point(p.x, p.y, p.z, i));

                if (p.x < min_x_) min_x_ = p.x;
                if (p.x > max_x_) max_x_ = p.x;
                if (p.y < min_y_) min_y_ = p.y;
                if (p.y > max_y_) max_y_ = p.y;
            }

            // 扩展边界（用于布料）
            min_x_ -= params_.cloth_buffer;
            max_x_ += params_.cloth_buffer;
            min_y_ -= params_.cloth_buffer;
            max_y_ += params_.cloth_buffer;

            std::cout << "  点云数量: " << points_.size() << std::endl;
            std::cout << "  点云范围: [" << min_x_ << ", " << max_x_ << "] x ["
                << min_y_ << ", " << max_y_ << "]" << std::endl;
        }

        // 步骤2: 初始化布料（与CloudCompare相同）
        void initializeCloth() {
            std::cout << "\n[2/6] 初始化布料网格..." << std::endl;

            cell_size_ = params_.cloth_resolution;

            // 计算网格尺寸
            double width = max_x_ - min_x_;
            double height = max_y_ - min_y_;

            grid_width_ = static_cast<int>(std::ceil(width / cell_size_)) + 1;
            grid_height_ = static_cast<int>(std::ceil(height / cell_size_)) + 1;

            std::cout << "  网格尺寸: " << grid_width_ << " x " << grid_height_ << std::endl;
            std::cout << "  网格分辨率: " << cell_size_ << " m" << std::endl;

            // 初始化布料网格
            cloth_grid_.resize(grid_height_);
            for (int i = 0; i < grid_height_; ++i) {
                cloth_grid_[i].resize(grid_width_);
            }

            // 找到最高点
            double max_z = -std::numeric_limits<double>::max();
            for (const auto& p : points_) {
                if (p.z > max_z) max_z = p.z;
            }

            double initial_height = max_z + params_.max_cloth_height;

            // 初始化每个节点
            for (int i = 0; i < grid_height_; ++i) {
                for (int j = 0; j < grid_width_; ++j) {
                    double x = min_x_ + j * cell_size_;
                    double y = min_y_ + i * cell_size_;

                    cloth_grid_[i][j] = Node(x, y, initial_height);
                    cloth_grid_[i][j].index_x = j;
                    cloth_grid_[i][j].index_y = i;

                    // 固定边界
                    if (i < params_.border_padding || i >= grid_height_ - params_.border_padding ||
                        j < params_.border_padding || j >= grid_width_ - params_.border_padding) {
                        cloth_grid_[i][j].movable = false;
                    }
                }
            }

            std::cout << "  布料初始高度: " << initial_height << " m" << std::endl;
        }

        // 步骤3: 布料模拟（核心算法）
        void clothSimulation() {
            std::cout << "\n[3/6] 执行布料模拟..." << std::endl;
            std::cout << "  迭代次数: " << params_.iterations << std::endl;
            std::cout << "  布料刚性: " << params_.rigidness << std::endl;

            // 创建KD树用于加速最近点搜索
            pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& p : points_) {
                search_cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
            }

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            kdtree.setInputCloud(search_cloud);

            // 布料模拟迭代
            for (int iter = 0; iter < params_.iterations; ++iter) {
                // 第一步: 计算重力作用
                applyGravity();

                // 第二步: 计算内部力（弹簧力）
                applyInternalForces();

                // 第三步: 处理地形碰撞
                handleTerrainCollision(kdtree);

                // 第四步: 更新节点位置
                updateNodePositions();

                // 进度显示
                if ((iter + 1) % 50 == 0) {
                    std::cout << "  进度: " << iter + 1 << "/" << params_.iterations << std::endl;
                }
            }
        }

        // 应用重力
        void applyGravity() {
            double g = -9.8 * params_.time_step * params_.time_step;

            for (int i = 0; i < grid_height_; ++i) {
                for (int j = 0; j < grid_width_; ++j) {
                    if (cloth_grid_[i][j].movable) {
                        cloth_grid_[i][j].tmp_z = cloth_grid_[i][j].z + g;
                    }
                }
            }
        }

        // 应用内部力（弹簧模型）
        void applyInternalForces() {
            // 根据刚性设置弹簧系数
            double spring_coeff = 0.0;
            switch (params_.rigidness) {
            case 1: spring_coeff = 0.1; break;  // 柔软
            case 2: spring_coeff = 0.3; break;  // 中等
            case 3: spring_coeff = 0.5; break;  // 刚性
            default: spring_coeff = 0.3; break;
            }

            // 与四个邻居的弹簧连接
            for (int i = 1; i < grid_height_ - 1; ++i) {
                for (int j = 1; j < grid_width_ - 1; ++j) {
                    if (!cloth_grid_[i][j].movable) continue;

                    // 邻居节点
                    Node& node = cloth_grid_[i][j];

                    // 上邻居
                    Node& up = cloth_grid_[i - 1][j];
                    double dz_up = up.tmp_z - node.tmp_z;
                    node.tmp_z += spring_coeff * dz_up;

                    // 下邻居
                    Node& down = cloth_grid_[i + 1][j];
                    double dz_down = down.tmp_z - node.tmp_z;
                    node.tmp_z += spring_coeff * dz_down;

                    // 左邻居
                    Node& left = cloth_grid_[i][j - 1];
                    double dz_left = left.tmp_z - node.tmp_z;
                    node.tmp_z += spring_coeff * dz_left;

                    // 右邻居
                    Node& right = cloth_grid_[i][j + 1];
                    double dz_right = right.tmp_z - node.tmp_z;
                    node.tmp_z += spring_coeff * dz_right;
                }
            }
        }

        // 处理地形碰撞
        void handleTerrainCollision(pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree) {
            pcl::PointXYZ search_point;
            std::vector<int> point_idx(1);
            std::vector<float> point_sqr_dist(1);

            for (int i = 0; i < grid_height_; ++i) {
                for (int j = 0; j < grid_width_; ++j) {
                    if (!cloth_grid_[i][j].movable) continue;

                    Node& node = cloth_grid_[i][j];

                    // 搜索最近的原始点
                    search_point.x = node.x;
                    search_point.y = node.y;
                    search_point.z = 0;  // 在XY平面上搜索

                    if (kdtree.nearestKSearch(search_point, 1, point_idx, point_sqr_dist) > 0) {
                        double terrain_z = points_[point_idx[0]].z;

                        // 如果布料低于地形，将其推高
                        if (node.tmp_z < terrain_z - params_.class_threshold) {
                            node.tmp_z = terrain_z - params_.class_threshold;
                        }
                    }
                }
            }
        }

        // 更新节点位置
        void updateNodePositions() {
            for (int i = 0; i < grid_height_; ++i) {
                for (int j = 0; j < grid_width_; ++j) {
                    if (cloth_grid_[i][j].movable) {
                        cloth_grid_[i][j].z = cloth_grid_[i][j].tmp_z;
                    }
                }
            }
        }

        // 步骤4: 计算地面高度图
        void computeGroundHeight() {
            std::cout << "\n[4/6] 计算地面高度图..." << std::endl;

            ground_height_map_.resize(points_.size(), 0.0);

            // 创建布料高度网格的KD树
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloth_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (int i = 0; i < grid_height_; ++i) {
                for (int j = 0; j < grid_width_; ++j) {
                    const Node& node = cloth_grid_[i][j];
                    cloth_cloud->push_back(pcl::PointXYZ(node.x, node.y, node.z));
                }
            }

            pcl::KdTreeFLANN<pcl::PointXYZ> cloth_kdtree;
            cloth_kdtree.setInputCloud(cloth_cloud);

            // 为每个点计算最近的布料高度
            pcl::PointXYZ search_point;
            std::vector<int> indices(1);
            std::vector<float> distances(1);

            for (size_t i = 0; i < points_.size(); ++i) {
                const Point& p = points_[i];
                search_point.x = p.x;
                search_point.y = p.y;
                search_point.z = p.z;

                if (cloth_kdtree.nearestKSearch(search_point, 1, indices, distances) > 0) {
                    int cloth_idx = indices[0];
                    ground_height_map_[i] = cloth_cloud->points[cloth_idx].z;
                }
            }
        }

        // 步骤5: 分类点云
        void classifyPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
            pcl::PointCloud<pcl::PointXYZ>::Ptr ground,
            pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground) {
            std::cout << "\n[5/6] 分类点云..." << std::endl;

            pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);
            pcl::PointIndices::Ptr non_ground_indices(new pcl::PointIndices);

            ground_indices->indices.reserve(points_.size() * 70 / 100);
            non_ground_indices->indices.reserve(points_.size() * 30 / 100);

            for (size_t i = 0; i < points_.size(); ++i) {
                const Point& p = points_[i];
                double height_diff = std::abs(p.z - ground_height_map_[i]);

                if (height_diff < params_.class_threshold) {
                    ground_indices->indices.push_back(p.index);
                }
                else {
                    non_ground_indices->indices.push_back(p.index);
                }

                // 进度显示
                if (i % (points_.size() / 10) == 0 && i > 0) {
                    std::cout << "  进度: " << (i * 100 / points_.size()) << "%" << std::endl;
                }
            }

            // 提取地面点
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud);

            extract.setIndices(ground_indices);
            extract.setNegative(false);
            extract.filter(*ground);

            // 提取非地面点
            extract.setIndices(non_ground_indices);
            extract.setNegative(false);
            extract.filter(*non_ground);

            std::cout << "  地面点: " << ground->size()
                << " (" << (ground->size() * 100.0 / cloud->size()) << "%)" << std::endl;
            std::cout << "  非地面点: " << non_ground->size()
                << " (" << (non_ground->size() * 100.0 / cloud->size()) << "%)" << std::endl;
        }

        // 步骤6: 后处理
        void postProcess(pcl::PointCloud<pcl::PointXYZ>::Ptr ground,
            pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground) {
            std::cout << "\n[6/6] 执行后处理..." << std::endl;

            // 简单的形态学开运算去除小区域
            morphologicalOpening(ground);
            std::cout << "  后处理完成" << std::endl;
        }

        // 形态学开运算
        void morphologicalOpening(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
            if (cloud->empty()) return;

            // 创建2D投影网格
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cloud, min_pt, max_pt);

            double cell_size = cell_size_;
            int cols = static_cast<int>((max_pt.x - min_pt.x) / cell_size) + 1;
            int rows = static_cast<int>((max_pt.y - min_pt.y) / cell_size) + 1;

            std::vector<std::vector<bool>> grid(rows, std::vector<bool>(cols, false));

            // 填充网格
            for (const auto& p : cloud->points) {
                int col = static_cast<int>((p.x - min_pt.x) / cell_size);
                int row = static_cast<int>((p.y - min_pt.y) / cell_size);

                if (col >= 0 && col < cols && row >= 0 && row < rows) {
                    grid[row][col] = true;
                }
            }

            // 侵蚀
            auto eroded = grid;
            int radius = 1;
            for (int r = radius; r < rows - radius; ++r) {
                for (int c = radius; c < cols - radius; ++c) {
                    if (grid[r][c]) {
                        bool keep = true;
                        for (int dr = -radius; dr <= radius && keep; ++dr) {
                            for (int dc = -radius; dc <= radius && keep; ++dc) {
                                if (!grid[r + dr][c + dc]) {
                                    keep = false;
                                }
                            }
                        }
                        eroded[r][c] = keep;
                    }
                }
            }

            // 膨胀
            auto dilated = eroded;
            for (int r = radius; r < rows - radius; ++r) {
                for (int c = radius; c < cols - radius; ++c) {
                    if (eroded[r][c]) {
                        for (int dr = -radius; dr <= radius; ++dr) {
                            for (int dc = -radius; dc <= radius; ++dc) {
                                dilated[r + dr][c + dc] = true;
                            }
                        }
                    }
                }
            }

            // 重新提取点云
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& p : cloud->points) {
                int col = static_cast<int>((p.x - min_pt.x) / cell_size);
                int row = static_cast<int>((p.y - min_pt.y) / cell_size);

                if (col >= 0 && col < cols && row >= 0 && row < rows && dilated[row][col]) {
                    filtered->push_back(p);
                }
            }

            *cloud = *filtered;
        }
    };

} // namespace csf

// 可视化函数
void visualizeCloudCompareResults(pcl::PointCloud<pcl::PointXYZ>::Ptr ground,
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground) {

    pcl::visualization::PCLVisualizer viewer("CloudCompare CSF结果");

    // 地面点云（绿色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        ground_color(ground, 0, 255, 0);
    viewer.addPointCloud<pcl::PointXYZ>(ground, ground_color, "ground");

    // 非地面点云（红色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        non_ground_color(non_ground, 255, 0, 0);
    viewer.addPointCloud<pcl::PointXYZ>(non_ground, non_ground_color, "non_ground");

    viewer.setBackgroundColor(0.1, 0.1, 0.1);
    viewer.addCoordinateSystem(5.0);
    viewer.setCameraPosition(0, 0, 50, 0, 0, 0, 0, -1, 0);
    viewer.initCameraParameters();

    std::cout << "\n按q键退出可视化窗口..." << std::endl;
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

// 主函数（模拟CloudCompare界面）
int main(int argc, char** argv) {
    std::string input_file = "../../data/clean_cloud.pcd";

    if (argc > 1) {
        input_file = argv[1];
    }

    std::cout << "========================================" << std::endl;
    std::cout << "CloudCompare CSF算法 - C++实现" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "输入文件: " << input_file << std::endl;
    std::cout << "========================================" << std::endl;

    // 加载点云
    auto load_start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud) == -1) {
        std::cerr << "错误：无法加载PCD文件！" << std::endl;
        return -1;
    }
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_end - load_start);

    std::cout << "? 点云加载完成" << std::endl;
    std::cout << "  点云数量: " << cloud->size() << " 点" << std::endl;
    std::cout << "  加载时间: " << load_duration.count() << " ms" << std::endl;

    // 显示点云信息
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    std::cout << "\n点云信息:" << std::endl;
    std::cout << "  X范围: " << min_pt.x << " 到 " << max_pt.x << std::endl;
    std::cout << "  Y范围: " << min_pt.y << " 到 " << max_pt.y << std::endl;
    std::cout << "  Z范围: " << min_pt.z << " 到 " << max_pt.z << std::endl;
    std::cout << "  高度范围: " << (max_pt.z - min_pt.z) << " m" << std::endl;

    // 设置CSF参数（与CloudCompare默认值相同）
    csf::Parameters params;

    // 根据点云大小自适应调整参数
    double area = (max_pt.x - min_pt.x) * (max_pt.y - min_pt.y);
    double height_range = max_pt.z - min_pt.z;


    if (height_range > 50.0) {
        // 大高度范围：陡峭地形
        params.cloth_resolution = 3.0;
        params.class_threshold = 3.0;
        params.rigidness = 3;
        std::cout << "\n检测到陡峭地形，使用陡峭模式参数:" << std::endl;
    }
    else {
        // 默认参数（与CloudCompare默认值相同）
        std::cout << "\n使用默认CSF参数:" << std::endl;
    }

    std::cout << "  布料分辨率: " << params.cloth_resolution << " m" << std::endl;
    std::cout << "  分类阈值: " << params.class_threshold << " m" << std::endl;
    std::cout << "  布料刚性: " << params.rigidness << " (1=软,2=中,3=硬)" << std::endl;
    std::cout << "  迭代次数: " << params.iterations << std::endl;

    // 创建结果容器
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 执行CSF分割
    std::cout << "\n========================================" << std::endl;
    std::cout << "开始CSF地面分割..." << std::endl;
    std::cout << "========================================" << std::endl;

    auto process_start = std::chrono::high_resolution_clock::now();

    csf::CloudCompareCSF csf_filter(params);
    csf_filter.filter(cloud, ground_cloud, non_ground_cloud);

    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        process_end - process_start);

    // 结果统计
    std::cout << "\n========================================" << std::endl;
    std::cout << "分割结果统计:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "处理时间: " << process_duration.count() / 1000.0 << " 秒" << std::endl;
    std::cout << "地面点数量: " << ground_cloud->size()
        << " (" << (ground_cloud->size() * 100.0 / cloud->size()) << "%)" << std::endl;
    std::cout << "非地面点数量: " << non_ground_cloud->size()
        << " (" << (non_ground_cloud->size() * 100.0 / cloud->size()) << "%)" << std::endl;

    // 保存结果
    std::string ground_file = "ground.pcd";
    std::string non_ground_file = "non_ground.pcd";

    pcl::io::savePCDFileASCII(ground_file, *ground_cloud);
    pcl::io::savePCDFileASCII(non_ground_file, *non_ground_cloud);

    std::cout << "\n结果已保存:" << std::endl;
    std::cout << "地面点云: " << ground_file << std::endl;
    std::cout << "非地面点云: " << non_ground_file << std::endl;

    // 可选：可视化
    bool enable_visualization = true;
    if (enable_visualization) {
        std::cout << "\n正在打开可视化窗口..." << std::endl;
        visualizeCloudCompareResults(ground_cloud, non_ground_cloud);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "CloudCompare CSF处理完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}