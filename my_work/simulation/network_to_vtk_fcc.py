"""
位错网络 → VTK 文件（ParaView 可以直接打开），带 FCC 滑移系编号和位错类型

用法：
  python network_to_vtk_fcc.py <.data 文件，或装有 .data 文件的文件夹> <输出文件夹> [--every N] [--max M]

  --every N   文件夹里每 N 个文件取 1 个（按步数从小到大排序），默认 1，也就是全取
  --max M     最多转换 M 个文件，默认 50，防止一次转成上万个大文件

在 ParaView 里打开生成的 .vtk 文件后，可以按下面这些量上色：
  SlipSystemID          1-12 = 标准 FCC 滑移系，13 = 不属于这 12 个
  DislocationCharacter  0 = 螺型，1 = 混合，2 = 刃型
  BurgersLineAngle      柏氏矢量和线方向的夹角（度）
  Burgers / Planes / LineDirection   三个矢量
"""
import os
import re
import glob
import argparse
import numpy as np
import pyexadis#ExaDiS 的 Python 接口
from pyexadis_base import NodeConstraints, ExaDisNet#ExaDisNet：网络数据结构
try:
    # Try importing DisNetManager from OpenDiS
    from framework.disnet_manager import DisNetManager
except ImportError:
    # Use dummy DisNetManager if OpenDiS is not available
    from pyexadis_base import DisNetManager

from typing import Dict, Tuple, Optional

# FCC的12个滑移系定义
# 滑移面 {111} 和滑移方向 <110>
# 已修正：(-111) 面和 (1-11) 面上原来各有两个方向写错（滑移面法向量和滑移方向必须垂直）
FCC_SLIP_SYSTEMS = {
    'planes': np.array([
        [ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1],  # (111) 面
        [-1,  1,  1], [-1,  1,  1], [-1,  1,  1],  # (-111) 面
        [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1],  # (1-11) 面
        [ 1,  1, -1], [ 1,  1, -1], [ 1,  1, -1],  # (11-1) 面
    ], dtype=float),  # 滑移面法向量
    'directions': np.array([
        [ 1, -1,  0], [ 1,  0, -1], [ 0,  1, -1],  # (111) 面上的滑移方向
        [ 1,  1,  0], [ 1,  0,  1], [ 0,  1, -1],  # (-111) 面上的滑移方向
        [ 1,  1,  0], [ 1,  0, -1], [ 0,  1,  1],  # (1-11) 面上的滑移方向
        [ 1, -1,  0], [ 1,  0,  1], [ 0,  1,  1],  # (11-1) 面上的滑移方向
    ], dtype=float),  # 滑移方向向量
    'names': [
        '(111)[1-10]',   '(111)[10-1]',   '(111)[01-1]',   # 滑移系 1-3
        '(-111)[110]',   '(-111)[101]',   '(-111)[01-1]',  # 滑移系 4-6
        '(1-11)[110]',   '(1-11)[10-1]',  '(1-11)[011]',   # 滑移系 7-9
        '(11-1)[1-10]',  '(11-1)[101]',   '(11-1)[011]',   # 滑移系 10-12
    ]
}
# 自检：每个滑移系的滑移面法向量和滑移方向必须垂直（点积为 0），写错了会立刻报错
assert np.allclose(np.sum(FCC_SLIP_SYSTEMS['planes'] * FCC_SLIP_SYSTEMS['directions'], axis=1), 0.0), \
    "FCC_SLIP_SYSTEMS 里有滑移面和滑移方向不垂直的条目"


def normalize_vector(v):
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:  # 避免除以零
        return v
    return v / norm


def is_parallel(v1, v2, tol=1e-6):
    """判断两个向量是否平行（包括反平行）"""
    v1_norm = normalize_vector(v1)  # 归一化向量1
    v2_norm = normalize_vector(v2)  # 归一化向量2
    dot = np.abs(np.dot(v1_norm, v2_norm))  # 计算两个归一化向量的点积的绝对值
    return dot > (1.0 - tol)  # 如果点积接近1，则认为两个向量平行


def identify_fcc_slip_system(burgers, plane, tol=1e-6):
    """
    识别FCC滑移系，返回具体的滑移系编号
    
    Parameters:
    -----------
    burgers : array-like, 伯格斯矢量
    plane : array-like, 滑移面法向量
    tol : float, 容差
    
    Returns:
    --------
    int : 1-12 表示对应的FCC滑移系编号，13 表示其他滑移情况
    """
    burgers = np.array(burgers, dtype=float)
    plane = np.array(plane, dtype=float)
    
    # 检查是否匹配12个FCC滑移系中的任意一个
    for i in range(12):
        fcc_plane = FCC_SLIP_SYSTEMS['planes'][i]
        fcc_dir = FCC_SLIP_SYSTEMS['directions'][i]
        
        # 检查滑移面和滑移方向是否都匹配
        if is_parallel(plane, fcc_plane, tol) and is_parallel(burgers, fcc_dir, tol):
            return i + 1  # 返回滑移系编号 1-12
    
    return 13  # 不属于标准12个滑移系


def compute_angle_burgers_line(burgers, line_vector):
    """
    计算伯格斯矢量与位错线段的夹角（度数）
    
    Parameters:
    -----------
    burgers : array-like, 伯格斯矢量
    line_vector : array-like, 位错线段方向向量
    
    Returns:
    --------
    float : 夹角（度数，范围 0-90度）
    """
    b_norm = normalize_vector(burgers)
    l_norm = normalize_vector(line_vector)#归一化位错线向量
    
    # 计算夹角的余弦值
    cos_angle = np.abs(np.dot(b_norm, l_norm))  # 取绝对值，使角度在0-90度范围内
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差导致超出[-1,1]
    
    # 转换为角度
    angle_rad = np.arccos(cos_angle)#计算夹角的弧度值
    angle_deg = np.degrees(angle_rad)#转换为度数
    
    return angle_deg


def compute_slip_system_labels_and_angles(burgers_vectors, plane_vectors, line_vectors, tol=1e-6):
    """
    批量计算滑移系标签和伯格斯矢量与位错线段的夹角
    
    Parameters:
    -----------
    burgers_vectors : ndarray, shape (n, 3), 伯格斯矢量数组
    plane_vectors : ndarray, shape (n, 3), 滑移面法向量数组
    line_vectors : ndarray, shape (n, 3), 位错线段方向向量数组
    tol : float, 容差
    
    Returns:
    --------
    labels : ndarray, 滑移系标签数组 (1-12 或 13)
    angles : ndarray, 伯格斯矢量与位错线段夹角数组（度数）
    """
    n = burgers_vectors.shape[0]#位错段数量
    labels = np.zeros(n, dtype=int)#滑移系标签数组
    angles = np.zeros(n, dtype=float)#夹角数组
    
    for i in range(n):#遍历每个位错段
        labels[i] = identify_fcc_slip_system(burgers_vectors[i], plane_vectors[i], tol)
        angles[i] = compute_angle_burgers_line(burgers_vectors[i], line_vectors[i])
    
    return labels, angles


def write_vtk_fcc(N, vtkfile: str, segprops={}, pbc_wrap=True, 
                  identify_slip_system=True, slip_tol=1e-6, verbose=True):
    """
    Write dislocation network in vtk format with FCC slip system identification
    
    Parameters:
    -----------
    N : DisNetManager, 位错网络管理器
    vtkfile : str, 输出VTK文件路径
    segprops : dict, 额外的段属性
    pbc_wrap : bool, 是否进行周期性边界条件包裹
    identify_slip_system : bool, 是否识别FCC滑移系
    slip_tol : float, 滑移系识别容差
    verbose : bool, 是否打印统计信息
    """
    data = N.export_data()
    
    # cell
    cell = pyexadis.Cell(**data["cell"])#用 data["cell"] 的字段构造一个 Cell 对象（** 解包字典为关键字参数）
    cell_origin, cell_center, h = np.array(cell.origin), np.array(cell.center()), np.array(cell.h)#取出晶胞原点 origin、中心 center、晶胞矩阵 h，并转成 numpy 数组
    c = cell_origin + np.array([np.zeros(3), h[0], h[1], h[2], h[0]+h[1],
                                h[0]+h[2], h[1]+h[2], h[0]+h[1]+h[2]])#计算晶胞8个顶点的坐标  origin + (0, a, b, c, a+b, a+c, b+c, a+b+c)
    
    # nodes
    nodes = data.get("nodes")#取节点数据字典
    rn = nodes.get("positions")#取节点坐标数组 rn（每个节点一个 3D 坐标）
    
    # segments
    segs = data.get("segs")#取段数据字典
    segsnid = segs.get("nodeids")#取段的节点索引数组 segsnid（每个段由两个节点索引定义）
    r1 = np.array(cell.closest_image(Rref=np.array(cell.center()), R=rn[segsnid[:,0]]))#取每段第一个端点的坐标，并映射到“离 cell.center 最近的周期像”
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:,1]]))#取每段第二个端点的坐标，并映射到“离 r1 最近的周期像”
    b = segs.get("burgers")#取段的伯格斯矢量数组 b
    p = segs.get("planes")#取段的滑移面法向量数组 p
    
    # PBC wrapping
    if np.all(np.array(cell.is_periodic()) == 0): #若晶胞三个方向都不是周期的（is_periodic 全为 0），强制关闭 pbc_wrap
        pbc_wrap = False
        
    if pbc_wrap:
        eps = 1e-10#很小的数，用于避免除 0 与边界判断抖动
        hinv = np.linalg.inv(h)#计算晶胞矩阵 h 的逆矩阵 hinv
        is_periodic = np.array(cell.is_periodic())#取周期性标志（每个方向 0/1）
        
        def outside_box(p):#判断点 p 是否在晶胞外部
            s = np.matmul(hinv, p - cell_origin)#计算分数坐标 s，使得 p = origin + h*s（取决于 h 的定义）
            return np.any(((s < -eps)|(s > 1.0+eps))&(is_periodic))#若在任一“周期方向”上分数坐标超出 [0,1]（留 eps 容差），返回 True
        
        def facet_intersection_position(r1, r2, i):#求线段 r1->r2 首次与盒子哪个面相交，以及相交位置
            s1 = np.matmul(hinv, r1 - cell_origin)
            s2 = np.matmul(hinv, r2 - cell_origin)
            t = s2 - s1#分数坐标下的方向向量（参数化用）
            t0 = -(s1 - 0.0) / (t + eps)#分别计算与 s=0 三个面的交点参数（避免 t=0 用 eps）
            t1 = -(s1 - 1.0) / (t + eps)#分别计算与 s=1 三个面的交点参数
            s = np.hstack((t0, t1))#把 6 个候选参数拼成长度 6 的数组（3个“0面”+3个“1面”）
            s[s < eps] = 1.0#把非常小/负的参数设为 1.0（相当于忽略“不在前方”的交点）
            facet = np.argmin(s)# 取最小的正参数对应的面 → 即最先撞到的盒子面索引 facet∈[0..5]
            if s[facet] < 1.0:#如果交点出现在 r1->r2 的线段内部（参数<1）
                pos = np.matmul(h, s1 + s[facet]*t) + cell_origin#计算交点的笛卡尔坐标：origin + h*(s1 + alpha*t)
                sfacet = s[facet]#交点参数
            else:
                facet = -1#用 -1 表示“没撞到面/无需切割”
                pos = r2#交点设为 r2（线段终点）
                sfacet = 1.0
            return pos, facet, sfacet#返回交点坐标、面索引、交点参数
            
        segsid = []#新段列表中每一段对应的“原始段 id”
        rsegs = []#新段的端点坐标列表（每段存 [r_start, r_end]）
        for i in range(segsnid.shape[0]):#遍历每一原始段
            n1, n2 = segsnid[i]#取该段的两个节点 id
            r1 = np.array(cell.closest_image(Rref=cell_center, R=rn[n1]))# 取端点1坐标并选离盒子中心最近的周期像
            r2 = np.array(cell.closest_image(Rref=r1, R=rn[n2]))# 取端点2坐标并选离端点1最近的周期像
            out = outside_box(r2)#判断 r2 是否在盒子外（周期方向）
            while out:#如果在盒子外，说明该段跨越了周期边界，需要切段
                pos, facet, sfacet = facet_intersection_position(r1, r2, i)#求从 r1 到 r2 首次撞到盒子边界的位置 pos
                if facet < 0: break#facet=-1 表示没有可切割的面，退出循环
                segsid.append(i)#新产生的这一小段来自原始段 i
                rsegs.append([r1, pos])#把切割出来的这一小段 [r1, pos] 存入新段列表
                r1 = pos + (1-2*np.floor(facet/3)) * (1.0-2*eps)*h[:,facet%3]#把 r1 “穿越周期边界”跳到对面继续;(1-2*floor(facet/3)) 给出 +1 或 -1 的方向因子
                r2 = np.array(cell.closest_image(Rref=r1, R=rn[n2]))#重新计算 r2 的最近周期像（现在参考点换成新的 r1）
                out = outside_box(r2)#再次判断是否还在盒子外：可能一段跨多次周期，需要多次切割
            segsid.append(i)#退出 while 后，把剩余最后一段也记录下来
            rsegs.append([r1, r2])#保存最后一段端点：r1 -> r2（此时 r2 应已在盒子内或无需再切）
        
        segsid = np.array(segsid).astype(int)#把 segsid 转成整型 numpy 数组
        nsegs = segsid.shape[0]#新段总数（切割后可能大于原始段数）
        rsegs = np.array(rsegs).reshape(-1,3)#把新段端点列表转成 numpy 数组，形状 (nsegs*2, 3)
        b = b[segsid]#根据 segsid 取出新段对应的伯格斯矢量
        p = p[segsid]#根据 segsid 取出新段对应的滑移面法向量
        for k, v in segprops.items():#根据 segsid 取出新段对应的其他属性
            segprops[k] = v[segsid]#更新 segprops 中每个属性的值
    else:
        nsegs = segsnid.shape[0]#原始段总数
        rsegs = np.hstack((r1, r2)).reshape(-1,3)#把原始段端点拼接成形状 (nsegs*2, 3) 的数组
        segsid = np.arange(nsegs)
    
    # ============================================
    # 计算位错线段方向向量
    # ============================================
    line_vectors = rsegs[1::2] - rsegs[0::2]  # 每段的方向向量 = r2 - r1
    
    # ============================================
    # 计算FCC滑移系标签和伯格斯矢量与位错线段的夹角
    # ============================================
    if identify_slip_system:
        slip_system_labels, burgers_line_angles = compute_slip_system_labels_and_angles(
            b, p, line_vectors, tol=slip_tol
        )
        if verbose:
            # 打印统计：每个滑移系有多少段、螺型/混合/刃型各有多少段
            print(f"  线段数（周期边界处切开之后）: {nsegs}")
            ids, counts = np.unique(slip_system_labels, return_counts=True)
            for sid, cnt in zip(ids, counts):
                name = FCC_SLIP_SYSTEMS['names'][sid - 1] if sid <= 12 else '其他（不在12个标准滑移系里）'
                print(f"    滑移系 {sid:>2} {name}: {cnt} 段")
            n_screw = int(np.sum(burgers_line_angles < 30.0))
            n_edge = int(np.sum(burgers_line_angles >= 60.0))
            print(f"  位错类型: 螺型 {n_screw}，混合 {nsegs - n_screw - n_edge}，刃型 {n_edge}")
    
    # ============================================
    # 写入VTK文件 (使用with语句确保文件正确关闭)
    # ============================================
    with open(vtkfile, 'w') as f:
        # VTK文件头
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Dislocation Network with FCC Slip System Analysis\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")#指定数据集类型为非结构化网格
        
        # ============================================
        # 写入点坐标 (POINTS)
        # ============================================
        total_points = c.shape[0] + 2*nsegs#总点数 = 8个晶胞顶点 + 每段2个端点
        f.write("POINTS %d double\n" % total_points)
        
        # 写入晶胞顶点
        for point in c:
            f.write("%.10e %.10e %.10e\n" % (point[0], point[1], point[2]))#格式化输出，保留10位小数
        
        # 写入位错线段端点
        for point in rsegs:#rsegs 已经是 (nsegs*2, 3) 的数组
            f.write("%.10e %.10e %.10e\n" % (point[0], point[1], point[2]))
        
        # ============================================
        # 写入单元连接关系 (CELLS)
        # ============================================
        total_cells = 1 + nsegs  # 1个晶胞 + nsegs条线段
        cell_list_size = 9 + 3*nsegs  # 晶胞: 9 (1+8), 线段: 3*nsegs (每段3个数)
        f.write("\nCELLS %d %d\n" % (total_cells, cell_list_size))
        
        # 晶胞单元 (HEXAHEDRON, 8个顶点)
        f.write("8 0 1 4 2 3 5 7 6\n")
        
        # 位错线段单元 (LINE, 每段2个顶点)
        offset = c.shape[0]  # 线段顶点从第8个点开始
        for i in range(nsegs):#遍历每一段
            p1 = offset + 2*i
            p2 = offset + 2*i + 1
            f.write("2 %d %d\n" % (p1, p2))
        
        # ============================================
        # 写入单元类型 (CELL_TYPES)
        # ============================================
        f.write("\nCELL_TYPES %d\n" % total_cells)
        f.write("12\n")  # 晶胞类型: VTK_HEXAHEDRON = 12
        for i in range(nsegs):
            f.write("3\n")  # 线段类型: VTK_LINE = 3
        
        # ============================================
        # 写入单元数据 (CELL_DATA)
        # ============================================
        f.write("\nCELL_DATA %d\n" % total_cells)
        
        # --------------------------------------------
        # 1. 伯格斯矢量 (VECTORS)
        # --------------------------------------------
        f.write("\nVECTORS Burgers double\n")
        f.write("0.0 0.0 0.0\n")  # 晶胞的占位值
        for vec in b:
            f.write("%.10e %.10e %.10e\n" % (vec[0], vec[1], vec[2]))
        
        # --------------------------------------------
        # 2. 滑移面法向量 (VECTORS)
        # --------------------------------------------
        f.write("\nVECTORS Planes double\n")
        f.write("0.0 0.0 0.0\n")  # 晶胞的占位值
        for vec in p:
            f.write("%.10e %.10e %.10e\n" % (vec[0], vec[1], vec[2]))
        
        # --------------------------------------------
        # 3. 位错线段方向向量 (VECTORS) - 新增
        # --------------------------------------------
        f.write("\nVECTORS LineDirection double\n")
        f.write("0.0 0.0 0.0\n")  # 晶胞的占位值
        for vec in line_vectors:
            vec_norm = normalize_vector(vec)
            f.write("%.10e %.10e %.10e\n" % (vec_norm[0], vec_norm[1], vec_norm[2]))
        
        # ============================================
        # 4. 滑移系标签 (SCALARS) - 整型
        # ============================================
        if identify_slip_system:
            f.write("\nSCALARS SlipSystemID int 1\n")
            f.write("LOOKUP_TABLE default\n")
            f.write("0\n")  # 晶胞的占位值
            for label in slip_system_labels:
                f.write("%d\n" % label)
            
            # ============================================
            # 5. 伯格斯矢量与位错线段夹角 (SCALARS) - 浮点型
            # ============================================
            f.write("\nSCALARS BurgersLineAngle double 1\n")
            f.write("LOOKUP_TABLE default\n")
            f.write("0.0\n")  # 晶胞的占位值
            for angle in burgers_line_angles:
                f.write("%.10e\n" % angle)
            
            # ============================================
            # 6. 位错类型分类 (SCALARS) - 整型
            # 0 = 螺型 (0-30度), 1 = 混合 (30-60度), 2 = 刃型 (60-90度)
            # ============================================
            f.write("\nSCALARS DislocationCharacter int 1\n")
            f.write("LOOKUP_TABLE default\n")
            f.write("0\n")  # 晶胞的占位值
            for angle in burgers_line_angles:
                if angle < 30.0:
                    char_type = 0  # 螺型
                elif angle < 60.0:
                    char_type = 1  # 混合
                else:
                    char_type = 2  # 刃型
                f.write("%d\n" % char_type)
        
        # ============================================
        # 7. 写入其他段属性
        # ============================================
        for k, v in segprops.items():#遍历其他段属性
            vals = np.atleast_2d(v.T).T#把 v 强制变成二维列向量/二维数组，统一后续写入逻辑
            if vals.shape[0] != nsegs:#检查：属性长度必须等于段数（切段后的 nsegs）
                raise ValueError(f'segprop "{k}" must have the same size as the number of segments')#不一致就报错，避免写错数据
            
            # 判断数据类型,判断属性数据是整数还是浮点数，并写入相应的VTK标量头。
            if np.issubdtype(vals.dtype, np.integer):#整数类型
                f.write("\nSCALARS %s int %d\n" % (str(k), vals.shape[1]))#写入 SCALARS 头
            else:
                f.write("\nSCALARS %s double %d\n" % (str(k), vals.shape[1]))
            
            f.write("LOOKUP_TABLE default\n")#声明使用默认的颜色查找表（Color Lookup Table）。
            
            # 晶胞占位值
            for j in range(vals.shape[1]):#遍历每一列
                f.write("0 " if j < vals.shape[1]-1 else "0\n")#写入晶胞占位值
            
            # 段属性值
            for row in vals:#遍历每一行
                for j, val in enumerate(row):# 遍历每一列（每个分量）
                    if j < len(row)-1:
                        f.write("%.10e " % val)# 非最后一列，加空格
                    else:
                        f.write("%.10e\n" % val)# 最后一列，加换行


def list_data_files(input_path):
    """如果 input_path 是文件就直接用；是文件夹就找里面所有 .data 文件，按步数从小到大排序"""
    if os.path.isfile(input_path):
        return [input_path]
    files = glob.glob(os.path.join(input_path, '*.data'))
    def step_of(path):
        m = re.search(r'(\d+)\.data$', os.path.basename(path))  # 文件名形如 config.200600.data
        return int(m.group(1)) if m else -1
    return sorted(files, key=lambda path: (step_of(path), path))


def main():
    parser = argparse.ArgumentParser(description='把 ParaDiS/ExaDiS 的 .data 构型文件转成带 FCC 滑移系信息的 VTK 文件')
    parser.add_argument('input', help='.data 文件，或装有 .data 文件的文件夹')
    parser.add_argument('output_dir', help='输出 .vtk 文件的文件夹（不存在会自动创建）')
    parser.add_argument('--every', type=int, default=1, help='每 N 个文件取 1 个，默认 1')
    parser.add_argument('--max', type=int, default=50, dest='max_files', help='最多转换多少个文件，默认 50')
    args = parser.parse_args()

    from pyexadis_utils import read_paradis   # 读 ParaDiS 格式的 .data 文件

    files = list_data_files(args.input)
    if not files:
        print(f"在 {args.input} 里没有找到 .data 文件")
        return
    selected = files[::max(1, args.every)][:args.max_files]
    print(f"找到 {len(files)} 个 .data 文件，本次转换其中 {len(selected)} 个")

    os.makedirs(args.output_dir, exist_ok=True)
    pyexadis.initialize()

    for i, data_file in enumerate(selected, 1):
        try:
            print(f"\n[{i}/{len(selected)}] {data_file}")
            N = read_paradis(data_file)
            base_name = os.path.splitext(os.path.basename(data_file))[0]
            output_file = os.path.join(args.output_dir, f"{base_name}.vtk")
            write_vtk_fcc(N, output_file, verbose=True)
            print(f"  已写出: {output_file}")
        except Exception as e:
            print(f"转换 {data_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

    pyexadis.finalize()
    print("\n完成")


if __name__ == '__main__':
    main()
