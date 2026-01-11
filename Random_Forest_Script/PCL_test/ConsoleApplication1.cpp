#if defined (_MSC_VER) && !defined (_WIN64)
#pragma warning(disable:4244)
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Real_timer.h>

// Search & Curvature dependencies
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <Eigen/Dense>

// 用于去除噪声（离群点）
#include <CGAL/remove_outliers.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Point_set_3<Point> Point_set;

typedef Point_set::Point_map Pmap;
typedef Point_set::Property_map<unsigned char> UCmap;

namespace Classification = CGAL::Classification;

typedef Classification::Label_handle Label_handle;
typedef Classification::Feature_handle Feature_handle;
typedef Classification::Label_set Label_set;
typedef Classification::Feature_set Feature_set;

typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap> Feature_generator;

// ==========================================
// 根据RGB硬编码判断类别 (Ground Truth)
// ==========================================
int get_label_by_rgb(unsigned char r, unsigned char g, unsigned char b)
{
    // 白色：未标注 -> -1
    if (r >= 250 && g >= 250 && b >= 250) return -1;
    // 绿色：植被 -> 0
    else if (g > r && g > b && g >= 80) return 0;
    // 蓝色：建筑 -> 1
    else if (b > r && b > g && b >= 80) return 1;
    // 黄色：车辆 -> 2
    else if (r >= 100 && g >= 100 && b <= 60) return 2;

    else return -1;
}

int main(int argc, char** argv)
{
    std::string filename = "s1.ply";
    if (argc > 1) filename = argv[1];

    std::ifstream in(filename, std::ios::binary);
    Point_set pts; // 【输入点云】
    std::cerr << "Reading input..." << std::endl;
    in >> pts;
    if (!in || pts.empty())
    {
        std::cerr << "Error: 无法读取点云文件或文件为空！" << std::endl;
        return EXIT_FAILURE;
    }
    std::cerr << "Read " << pts.size() << " points." << std::endl;

    // ==============================================
    // 1. 获取原始RGB属性 (用于生成训练标签)
    // ==============================================
    UCmap red_map_in, green_map_in, blue_map_in;
    bool has_red, has_green, has_blue;
    std::tie(red_map_in, has_red) = pts.property_map<unsigned char>("red");
    std::tie(green_map_in, has_green) = pts.property_map<unsigned char>("green");
    std::tie(blue_map_in, has_blue) = pts.property_map<unsigned char>("blue");

    if (!has_red || !has_green || !has_blue) {
        std::cerr << "Error: 输入点云没有RGB属性！" << std::endl;
        return EXIT_FAILURE;
    }

    // ==============================================
    // 2. 生成整数标签 (Ground Truth)并统计
    // ==============================================
    std::vector<int> training_labels(pts.size(), -1);
    std::map<int, int> label_counts;
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        training_labels[i] = get_label_by_rgb(red_map_in[i], green_map_in[i], blue_map_in[i]);
        label_counts[training_labels[i]]++;
    }
    std::cerr << "Label distribution (Input GT): Road(0):" << label_counts[0]
        << ", Veg(1):" << label_counts[1] << ", Build(2):" << label_counts[2]
        << ", Car(3):" << label_counts[3] << std::endl;

    // ==============================================
    // 3. 特征生成 (几何特征 + 曲率)
    // ==============================================
    Feature_set features;
    std::cerr << "Generating geometric features..." << std::endl;
    Feature_generator generator(pts, pts.point_map(), 3); // 3 scales
    features.begin_parallel_additions();
    generator.generate_point_based_features(features);
    features.end_parallel_additions();

    std::cerr << "Calculating curvature..." << std::endl;
    typedef CGAL::Search_traits_3<Kernel> TreeTraits;
    typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
    typedef Neighbor_search::Tree Tree;
    std::vector<Point> points_vec;
    points_vec.reserve(pts.size());
    for (auto it = pts.begin(); it != pts.end(); ++it) points_vec.push_back(pts.point(*it));
    Tree tree(points_vec.begin(), points_vec.end());
    auto curvature_pair = pts.add_property_map<float>("curvature", 0.0f);
    auto curvature_map = curvature_pair.first;
    const unsigned int k_neighbors = 15;
    for (auto it = pts.begin(); it != pts.end(); ++it) {
        Neighbor_search search(tree, pts.point(*it), k_neighbors);
        std::vector<Eigen::Vector3d> neighbors;
        for (auto nb = search.begin(); nb != search.end(); ++nb) neighbors.emplace_back(nb->first.x(), nb->first.y(), nb->first.z());
        if (neighbors.size() < 3) { curvature_map[*it] = 0.0f; continue; }
        Eigen::Vector3d centroid(0, 0, 0);
        for (auto& v : neighbors) centroid += v; centroid /= neighbors.size();
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (auto& v : neighbors) { Eigen::Vector3d diff = v - centroid; cov += diff * diff.transpose(); }
        cov /= neighbors.size();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        Eigen::Vector3d eigen_values = solver.eigenvalues();
        double sum = eigen_values.sum();
        curvature_map[*it] = static_cast<float>(sum > 0 ? eigen_values.minCoeff() / sum : 0.0);
    }
    typedef Classification::Feature::Simple_feature<Point_set, decltype(curvature_map)> Simple_feature;
    features.add<Simple_feature>(pts, curvature_map, "curvature");

    // ==============================================
    // 4. 定义类别与输出颜色
    // ==============================================
    Label_set labels;

    // 注意添加顺序影响索引：0: vegetation, 1: building, 2: car
    Label_handle vegetation = labels.add("vegetation");
    Label_handle building = labels.add("building");
    Label_handle car = labels.add("car");

    building->set_color(CGAL::IO::Color(0, 0, 255));
    car->set_color(CGAL::IO::Color(255, 255, 0));
    vegetation->set_color(CGAL::IO::Color(0, 255, 0));

    // ==============================================
    // 5. 训练分类器
    // ==============================================
    std::cerr << "Training Random Forest..." << std::endl;
    Classification::ETHZ::Random_forest_classifier classifier(labels, features);
    classifier.train(training_labels);

    // ==============================================
    // 6. 预测与优化 (Graph Cut)
    // ==============================================
    std::vector<int> label_indices(pts.size(), -1);
    std::cerr << "Classifying with Graph Cut..." << std::endl;
    Classification::classify_with_graphcut<CGAL::Parallel_if_available_tag>(
        pts, pts.point_map(), labels, classifier,
        generator.neighborhood().k_neighbor_query(15), 0.2f, 1, label_indices);

    // ==============================================
    // 7. 保存完整分类结果
    // ==============================================
    std::cerr << "Creating full classification output..." << std::endl;

    Point_set pts_out;
    pts_out.reserve(pts.size());
    for (auto it = pts.begin(); it != pts.end(); ++it) {
        pts_out.insert(pts.point(*it));
    }

    UCmap r_out = pts_out.add_property_map<unsigned char>("red", 255).first;
    UCmap g_out = pts_out.add_property_map<unsigned char>("green", 255).first;
    UCmap b_out = pts_out.add_property_map<unsigned char>("blue", 255).first;

    for (std::size_t i = 0; i < pts_out.size(); ++i)
    {
        int idx = label_indices[i];
        if (idx >= 0 && idx < labels.size()) {
            Label_handle label = labels[idx];
            const CGAL::IO::Color& color = label->color();
            r_out[i] = color.red();
            g_out[i] = color.green();
            b_out[i] = color.blue();
        }
        else {
            r_out[i] = 255; g_out[i] = 255; b_out[i] = 255;
        }
    }

    std::ofstream f("classification_result_fixed.ply", std::ios::binary);
    if (!f) {
        std::cerr << "Error: 无法创建输出文件！" << std::endl;
        return EXIT_FAILURE;
    }
    f.precision(18);
    f << pts_out;
    f.close();
    std::cerr << "Full classification result saved." << std::endl;

    // ==============================================
    // 8. 【强力去噪版】提取建筑并去除噪声
    // ==============================================
    std::cerr << "\n--- Extracting and Denoising Buildings (High Intensity) ---" << std::endl;

    Point_set building_pts;
    int building_idx = 1; // 对应 labels: vegetation(0), building(1), car(2)

    // A. 提取建筑点
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        if (label_indices[i] == building_idx)
        {
            building_pts.insert(pts.point(i));
        }
    }
    std::cerr << "Extracted " << building_pts.size() << " building points (raw)." << std::endl;

    if (!building_pts.empty())
    {
        // B. 强力去噪配置
        // ------------------------------------------------------------------
        // nb_neighbors: 60 (增大邻域范围，避免局部小团簇被误判为正常)
        // threshold_percent: 10.0 (强制剔除最稀疏的10%的点，力度很大)
        // ------------------------------------------------------------------
        const int nb_neighbors = 100;
        const double threshold_percent = 20.0;

        std::cerr << "Executing strong outlier removal (K=" << nb_neighbors
            << ", Remove Top " << threshold_percent << "%)..." << std::endl;

        building_pts.remove_from(
            CGAL::remove_outliers<CGAL::Parallel_if_available_tag>(
                building_pts,
                nb_neighbors,
                building_pts.parameters().threshold_percent(threshold_percent).threshold_distance(0.))
        );

        std::cerr << "Building points after denoising: " << building_pts.size() << std::endl;

        // C. 保存建筑点云 (统一设为蓝色，方便查看)
        UCmap r_b = building_pts.add_property_map<unsigned char>("red", 0).first;
        UCmap g_b = building_pts.add_property_map<unsigned char>("green", 0).first;
        UCmap b_b = building_pts.add_property_map<unsigned char>("blue", 255).first;

        std::string build_filename = "only_buildings_strong_denoised.ply";
        std::ofstream fb(build_filename, std::ios::binary);
        if (fb) {
            fb.precision(18);
            fb << building_pts;
            fb.close();
            std::cerr << "\nSuccess! Denoised buildings saved to: " << build_filename << std::endl;
        }
        else {
            std::cerr << "Error: Could not save building file." << std::endl;
        }
    }
    else {
        std::cerr << "Warning: No building points were classified, skipping extraction." << std::endl;
    }

    return EXIT_SUCCESS;
}