import time

import numpy as np  # 导入所需外部库
import open3d as o3d


def poisson_reconstruction(input_file, output_file, depth=9, density_threshold=0.01):
    """
    使用泊松重建算法从点云生成三角网格

    参数:
        input_file: 输入点云文件路径 (.ply, .pcd, .xyz, .pts等)
        output_file: 输出网格文件路径 (.ply, .obj, .stl等)
        depth: 重建深度，越大越精细，通常8-10 (默认9)
        density_threshold: 密度阈值，用于过滤低密度顶点，0-1之间 (默认0.01)
    """

    # 1. 读取点云
    print("正在读取点云...")
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"点云包含 {len(pcd.points)} 个点")

    # 计算点的平均距离供估计法向量使用

    # 自动计算合适的半径

    print("计算点云密度...")
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"平均点间距: {avg_dist:.4f}, 使用半径: {avg_dist * 2}")

    # 2. 检查并估计法向量
    if not pcd.has_normals():
        print("正在估计法向量...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=avg_dist * 2,  # 这个地方使用平均点距离的两倍是最合适的
                max_nn=50,
            )
        )
        # 统一法向量方向
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # 3. 执行泊松重建
    print("正在执行泊松重建...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False
    )
    print(
        f"重建完成，生成 {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角面片"
    )

    # 4. 根据密度过滤低密度顶点（可选）
    if density_threshold > 0:
        print("正在过滤低密度顶点...")
        densities = np.asarray(densities)

        # 计算密度阈值
        density_threshold_value = np.quantile(densities, density_threshold)

        # 创建顶点mask
        vertices_to_remove = densities < density_threshold_value

        # 移除低密度顶点
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(
            f"过滤后剩余 {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角面片"
        )

    # 5. 保存网格
    print(f"正在保存网格到 {output_file}...")
    o3d.io.write_triangle_mesh(output_file, mesh)
    print("完成！")

    # # 6. 可视化（可选）
    # print("显示重建结果...")
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    return mesh


# 使用示例
if __name__ == "__main__":
    start_time = time.perf_counter()
    # 替换为你的文件路径
    input_cloud = "clean_buildings.pcd"
    output_mesh = "clean_buildings_poisson.ply"

    poisson_reconstruction(
        input_cloud,
        output_mesh,
        depth=11,
        density_threshold=0.01,  # 移除密度最低的1%的顶点
    )

    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print("程序运行时间：", delta_time)
