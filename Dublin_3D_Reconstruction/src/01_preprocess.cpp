#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// 滤波
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>

// 坐标与变换
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

int main(int argc, char** argv)
{
    // ==============================
    // 0. 点云指针定义
    // ==============================
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sor(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centered(new pcl::PointCloud<pcl::PointXYZ>);

    // ==============================
    // 1. 读取点云
    // ==============================
    std::cout << "Step 1: 读取原始点云..." << std::endl;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../../data/DublinCity.pcd", *cloud_raw) == -1)
    {
        PCL_ERROR("无法读取点云文件！\n");
        return -1;
    }
    std::cout << "原始点数: " << cloud_raw->size() << std::endl;

    // ==============================
    // 2. 数据范围诊断（用于报告）
    // ==============================
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud_raw, minPt, maxPt);
    std::cout << "【Bounding Box 诊断】" << std::endl;
    std::cout << "X: " << minPt.x << " ~ " << maxPt.x << std::endl;
    std::cout << "Y: " << minPt.y << " ~ " << maxPt.y << std::endl;
    std::cout << "Z: " << minPt.z << " ~ " << maxPt.z << std::endl;

    // ==============================
    // 3. PassThrough 裁剪（辅助去除极端幽灵点）
    // ==============================
    std::cout << "Step 2: PassThrough 空间裁剪..." << std::endl;
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud_raw);

    pass.setFilterFieldName("x");
    pass.setFilterLimits(-5000.0, 5000.0);
    pass.filter(*cloud_roi);

    pass.setInputCloud(cloud_roi);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-5000.0, 5000.0);
    pass.filter(*cloud_roi);

    pass.setInputCloud(cloud_roi);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-200.0, 500.0);
    pass.filter(*cloud_roi);

    std::cout << "裁剪后点数: " << cloud_roi->size() << std::endl;

    // ==============================
    // 4. SOR 统计离群点去除（严格对应报告 4.1.2）
    // ==============================
    std::cout << "Step 3: Statistical Outlier Removal (SOR)..." << std::endl;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_roi);
    sor.setMeanK(5);                // 最小邻居数
    sor.setStddevMulThresh(1.0);    // 统计阈值
    sor.filter(*cloud_sor);

    std::cout << "SOR 后点数: " << cloud_sor->size() << std::endl;

    // ==============================
    // 5. VoxelGrid 降采样（0.3m）
    // ==============================
    std::cout << "Step 4: VoxelGrid 降采样..." << std::endl;
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud_sor);
    voxel.setLeafSize(0.2f, 0.2f, 0.2f);
    voxel.filter(*cloud_downsampled);

    std::cout << "降采样后点数: " << cloud_downsampled->size() << std::endl;

    // ==============================
    // 6. 坐标标准化（中心化）
    // ==============================
    std::cout << "Step 5: 坐标标准化（中心化）..." << std::endl;
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_downsampled, centroid);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -centroid[0], -centroid[1], -centroid[2];

    pcl::transformPointCloud(*cloud_downsampled, *cloud_centered, transform);

    std::cout << "中心坐标: "
        << centroid[0] << ", "
        << centroid[1] << ", "
        << centroid[2] << std::endl;

    // ==============================
    // 7. 保存结果
    // ==============================
    pcl::io::savePCDFileBinary("clean_cloud.pcd", *cloud_centered);
    std::cout << "预处理完成，文件已保存为 clean_cloud.pcd" << std::endl;

    return 0;
}
