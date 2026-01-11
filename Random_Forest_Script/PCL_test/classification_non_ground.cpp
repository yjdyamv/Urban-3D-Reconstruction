#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

// PCL 基础
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>

// 搜索
#include <pcl/search/kdtree.h>

// 特征
#include <pcl/features/normal_3d_omp.h>

// 分割
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>

// 滤波
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZ PointT;
typedef pcl::Normal NormalT;

// 辅助函数
float getClusterAvgCurvature(const pcl::PointCloud<NormalT>::Ptr& cloud_normals,
    const std::vector<int>& indices) {
    if (indices.empty()) return 1.0f;
    double sum_curv = 0.0;
    for (int idx : indices) {
        sum_curv += cloud_normals->points[idx].curvature;
    }
    return (float)(sum_curv / indices.size());
}

int main(int argc, char** argv) {
    // 1. 读取
    std::string input_filename = "Off-ground points.pcd";
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    std::cout << "[1/5] 读取点云..." << std::endl;
    if (pcl::io::loadPCDFile<PointT>(input_filename, *cloud) == -1) {
        PCL_ERROR("无法读取文件 \n");
        return (-1);
    }
    std::cout << "    -> 点数: " << cloud->size() << std::endl;

    // 2. 法向量
    std::cout << "[2/5] 计算法向量..." << std::endl;
    pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::PointCloud<NormalT>::Ptr normals(new pcl::PointCloud<NormalT>);
    pcl::NormalEstimationOMP<PointT, NormalT> ne;
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(50);
    ne.compute(*normals);

    // 3. 第一轮：区域生长
    pcl::RegionGrowing<PointT, NormalT> reg;
    reg.setMinClusterSize(100);
    reg.setMaxClusterSize(10000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    // 保持 6.0 度
    reg.setSmoothnessThreshold(6.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);

    std::vector<pcl::PointIndices> smooth_clusters;
    reg.extract(smooth_clusters);

    pcl::PointCloud<PointT>::Ptr buildings_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr remainder_cloud(new pcl::PointCloud<PointT>);
    std::vector<int> remainder_indices_map;

    std::vector<bool> is_smooth(cloud->size(), false);
    for (const auto& cluster : smooth_clusters) {
        for (int idx : cluster.indices) {
            is_smooth[idx] = true;
            buildings_cloud->points.push_back(cloud->points[idx]);
        }
    }

    for (size_t i = 0; i < cloud->size(); ++i) {
        if (!is_smooth[i]) {
            remainder_cloud->points.push_back(cloud->points[i]);
            remainder_indices_map.push_back(i);
        }
    }
    remainder_cloud->width = remainder_cloud->points.size();
    remainder_cloud->height = 1;
    remainder_cloud->is_dense = true;

    // 4. 第二轮：救援行动
    if (!remainder_cloud->empty()) {
        std::cout << "[4/5] 第二轮: 救援分析 (加入面积权重)..." << std::endl;

        pcl::search::KdTree<PointT>::Ptr tree_rough(new pcl::search::KdTree<PointT>);
        tree_rough->setInputCloud(remainder_cloud);

        std::vector<pcl::PointIndices> rough_clusters;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(0.5);
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(10000000);
        ec.setSearchMethod(tree_rough);
        ec.setInputCloud(remainder_cloud);
        ec.extract(rough_clusters);

        pcl::PointCloud<PointT>::Ptr final_vegetation(new pcl::PointCloud<PointT>);
        int rescued_count = 0;

        for (const auto& cluster : rough_clusters) {
            pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>);
            std::vector<int> original_indices;

            for (int idx : cluster.indices) {
                cluster_cloud->points.push_back(remainder_cloud->points[idx]);
                original_indices.push_back(remainder_indices_map[idx]);
            }

            // PCA
            pcl::PCA<PointT> pca;
            pca.setInputCloud(cluster_cloud);
            Eigen::Vector3f values = pca.getEigenValues(); // L1 > L2 > L3

            float sum = values[0] + values[1] + values[2];
            float e3 = values[2] / sum; // 厚度
            float flatness = (values[1] - values[2]) / values[0]; // 板状度

            Eigen::Matrix3f vectors = pca.getEigenVectors();
            Eigen::Vector3f normal = vectors.col(2);

            PointT min_p, max_p;
            pcl::getMinMax3D(*cluster_cloud, min_p, max_p);
            double height = max_p.z - min_p.z;
            double area_approx = (max_p.x - min_p.x) * (max_p.y - min_p.y);

            float avg_curv = getClusterAvgCurvature(normals, original_indices);

            bool rescue = false;

            // --- 判定逻辑 ---

            bool is_high = height > 2.0;
            bool is_not_vertical = std::abs(normal[2]) > 0.2;

            // 规则 A: 墙面 (Strict)
            if (e3 < 0.06 && std::abs(normal[2]) < 0.5 && avg_curv < 0.05) {
                rescue = true;
            }

            // 规则 B: 房顶 (分级特赦)
            else if (is_high && is_not_vertical) {

                // 级别 1: 【光滑顶】 (宽松)
                if (avg_curv < 0.08 && e3 < 0.30 && flatness > 0.15) {
                    rescue = true;
                }

                // 级别 2: 【薄糙顶】 (宽松)
                else if (avg_curv < 0.20 && e3 < 0.10 && flatness > 0.20) {
                    rescue = true;
                }

                // 级别 3: 【板状糙顶】 (关键改动！)
                // 之前是 flatness > 0.35 才放行，现在我们分两档：
                // 档位 A: 如果面积很小，依然要求超级扁 (0.35)
                // 档位 B: 如果面积够大 (> 15.0)，可以放宽到 0.25 !!!
                //        (因为树很少有这么大且连续的扁平面)
                else if (e3 < 0.30 && avg_curv < 0.18) {
                    if (flatness > 0.35) {
                        rescue = true;
                    }
                    else if (flatness > 0.25 && area_approx > 15.0) {
                        rescue = true;
                    }
                }
            }

            // 规则 C: 超大物体
            if (area_approx > 35.0 && e3 < 0.25 && flatness > 0.15 && avg_curv < 0.12) {
                rescue = true;
            }

            // --- 否决逻辑 (精准打击) ---

            // 1. 绝对粗糙线
            if (avg_curv > 0.20) rescue = false;

            // 2. 厚糙树木 (Veto 2)
            // 条件：又厚(>0.12) 又糙(>0.08)
            if (e3 > 0.12 && avg_curv > 0.08) {
                // 复活金牌：
                // 1. 超级扁 (flatness > 0.35) -> 免死
                // 2. 比较扁 (flatness > 0.25) 且 比较大 (area > 15.0) -> 免死
                // 3. 比较扁 (flatness > 0.25) 且 比较滑 (curv < 0.12) -> 免死
                // 否则 -> 死刑

                bool strong_plank = flatness > 0.35;
                bool big_plank = (flatness > 0.25 && area_approx > 15.0);
                bool smooth_plank = (flatness > 0.25 && avg_curv < 0.12);

                if (!strong_plank && !big_plank && !smooth_plank) {
                    rescue = false;
                }
            }

            // 3. 土豆形状
            if (flatness < 0.10) rescue = false;

            // 4. 矮胖子
            if (height < 1.5 && e3 > 0.08) rescue = false;

            if (rescue) {
                *buildings_cloud += *cluster_cloud;
                rescued_count++;
            }
            else {
                *final_vegetation += *cluster_cloud;
            }
        }

        remainder_cloud = final_vegetation;
        std::cout << "    -> 成功救回 " << rescued_count << " 个组件。" << std::endl;
    }

    // 5. 保存
    std::cout << "[5/5] 保存结果..." << std::endl;
    pcl::PCDWriter writer;
    if (!buildings_cloud->empty()) writer.write<PointT>("buildings.pcd", *buildings_cloud, false);
    if (!remainder_cloud->empty()) writer.write<PointT>("vegetation.pcd", *remainder_cloud, false);

    std::cout << "完成。" << std::endl;
    return 0;
}