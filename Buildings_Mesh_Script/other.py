import time

import numpy as np
import open3d as o3d


def ball_pivoting_reconstruction(input_file, output_file, radii=None):
    """
    球滚动算法重建

    优点: 保留边界、快速、适合密集点云
    缺点: 对点云密度敏感、可能产生孔洞

    参数:
        radii: 球半径列表，通常是点云平均间距的2-3倍。如果为None，自动计算
    """

    print("读取点云...")
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"点云包含 {len(pcd.points)} 个点")

    # 自动计算合适的半径
    if radii is None:
        print("计算点云密度...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 2, avg_dist * 2, avg_dist * 2]
        print(f"平均点间距: {avg_dist:.4f}, 使用半径: {radii}")

    # 估计法向量
    if not pcd.has_normals():
        print("估计法向量...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=avg_dist * 2,  # 这个半径最好选用点平均距离的两倍
                max_nn=50,
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # 执行BPA重建
    print("执行球滚动算法...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii),  # 关键：使用 DoubleVector
    )

    print(f"生成 {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面片")

    # 清理网格
    print("清理网格...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # 保存
    print(f"保存到 {output_file}...")
    o3d.io.write_triangle_mesh(output_file, mesh)

    # # 可视化
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    return mesh


# 使用示例
if __name__ == "__main__":
    # # 方法1: 自动计算半径（推荐）
    start_time = time.perf_counter()
    ball_pivoting_reconstruction("clean_buildings.pcd", "clean_buildings_bpa.ply")

    end_time = time.perf_counter()
    delta_time = end_time - start_time
    print("程序运行时间：", delta_time)
