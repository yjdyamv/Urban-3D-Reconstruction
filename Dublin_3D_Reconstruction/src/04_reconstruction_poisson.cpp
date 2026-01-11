#include <iostream>
#include <string>
#include <vector>

// PCL 基础
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>

// 法向量估计
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>

// 表面重建
#include <pcl/surface/poisson.h>

// 只是为了拼接 XYZ 和 Normal
#include <pcl/features/moment_of_inertia_estimation.h> 

using PointT = pcl::PointXYZ;
using PointNT = pcl::PointNormal;

int main(int argc, char** argv)
{
    // ==========================================
    // 1. 读取建筑物点云
    // ==========================================
    std::string input_file = "buildings.pcd"; // 来自 Step 3 的输出
    std::string output_file = "buildings_mesh.ply";

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    std::cout << "[1/4] 读取点云: " << input_file << " ..." << std::endl;

    if (pcl::io::loadPCDFile<PointT>(input_file, *cloud) == -1)
    {
        PCL_ERROR("无法读取文件，请确认 Step 3 是否已运行并生成了 buildings.pcd\n");
        return -1;
    }
    std::cout << "    -> 点数: " << cloud->size() << std::endl;

    // ==========================================
    // 2. 估计法向量 (Poisson 重建必须项)
    // ==========================================
    // Python脚本中你用了平均距离的2倍作为半径。
    // 在PCL中，为了稳健，我们通常使用 K-近邻 (K-Search) 或 半径搜索。
    // 这里使用 K=50，与你Python脚本中的 max_nn=50 保持一致。

    std::cout << "[2/4] 估计法向量 (OpenMP加速)..." << std::endl;
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(50); // 对应 Python: max_nn=50
    ne.compute(*normals);

    // 将 XYZ 和 Normal 合并到一个 PointCloud 中 (PCL Poisson 需要这种格式)
    pcl::PointCloud<PointNT>::Ptr cloud_with_normals(new pcl::PointCloud<PointNT>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    // ==========================================
    // 3. 执行泊松重建 (Poisson Reconstruction)
    // ==========================================
    std::cout << "[3/4] 执行泊松重建 (Depth=11)..." << std::endl;
    pcl::Poisson<PointNT> poisson;
    poisson.setDepth(11);              // 对应 Python: depth=11 (八叉树深度)
    poisson.setSolverDivide(8);        // 默认值，求解器深度
    poisson.setIsoDivide(8);           // 默认值，等值面提取深度
    poisson.setPointWeight(4.0f);      // 点权重，建议值

    // 设置输入
    poisson.setInputCloud(cloud_with_normals);

    // 执行重建
    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    std::cout << "    -> 重建完成，包含三角面片数: " << mesh.polygons.size() << std::endl;

    // ==========================================
    // 4. 保存结果
    // ==========================================
    std::cout << "[4/4] 保存 Mesh 到: " << output_file << " ..." << std::endl;
    pcl::io::savePLYFile(output_file, mesh);

    std::cout << "全部完成！请使用 CloudCompare 查看 " << output_file << std::endl;

    return 0;
}