import numpy as np


class AEDisplacementRecorder:
    """
    原位计算AE位移场的记录器。
    需要在每一步的 step_begin 调用 snapshot_before_step()，
    再在同一步的 step_post_integrate 调用 record_step()。
    """

    def __init__(self, field_points, core_a, nu, output_file="ae_displacement.dat"):
        """
        field_points : 形状(N,3)的数组，N个场点坐标（单位：b，跟模拟内部坐标一致）
        core_a       : 非奇异核心宽度（单位同field_points，即"多少个b"）
        nu           : 泊松比
        output_file  : 输出文件名
        """
        self.field_points = np.asarray(field_points, dtype=float)
        self.n_points = len(self.field_points)
        self.core_a = core_a
        self.nu = nu
        self.u_total = np.zeros((self.n_points, 3))
        self.output_file = output_file
        self._file_handle = None
        self._old_positions = None
        self._step_count = 0

        # Levi-Civita符号，初始化一次，后面重复使用
        self._EPS = np.zeros((3, 3, 3))
        self._EPS[0, 1, 2] = self._EPS[1, 2, 0] = self._EPS[2, 0, 1] = 1.0
        self._EPS[0, 2, 1] = self._EPS[2, 1, 0] = self._EPS[1, 0, 2] = -1.0

    # ---------- 数学部分：立体角 ----------

    def _solid_angle_triangle(self, a, b, c):
        """van Oosterom & Strackee 有符号立体角公式"""
        na, nb, nc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        numerator = np.dot(a, np.cross(b, c))
        denominator = (na * nb * nc
                       + np.dot(a, b) * nc
                       + np.dot(b, c) * na
                       + np.dot(c, a) * nb)
        return 2.0 * np.arctan2(numerator, denominator)

    def _solid_angle_polygon(self, verts, x):
        """四边形立体角：扇形三角剖分求和"""
        rel = verts - x
        omega = 0.0
        for i in range(1, len(verts) - 1):
            omega += self._solid_angle_triangle(rel[0], rel[i], rel[i + 1])
        return omega

    # ---------- 数学部分：单段线段的弹性位移(第二+三项) ----------

    def _disp_line_segment(self, x, x1, x2, b):
        """单段线段对位移的贡献（第二+三项，不含立体角）"""
        nu, a, EPS = self.nu, self.core_a, self._EPS

        seg = x2 - x1
        L = np.linalg.norm(seg)
        if L < 1.0e-30:
            return np.zeros(3)
        t = seg / L

        xm1 = x - x1
        proj = np.dot(xm1, t)
        x0 = x1 + proj * t
        d = x0 - x
        d2 = np.dot(d, d)

        e1v = x1 - x0
        e2v = x2 - x0
        s1 = np.dot(e1v, t)
        s2 = np.dot(e2v, t)
        c = d2 + a * a

        Ra1 = np.sqrt(c + s1 * s1)
        Ra2 = np.sqrt(c + s2 * s2)

        I_1_Ra   = np.log(s2 + Ra2)            - np.log(s1 + Ra1)
        I_1_Ra3  = s2 / (c * Ra2)              - s1 / (c * Ra1)
        I_s_Ra3  = (-1.0 / Ra2)                - (-1.0 / Ra1)
        I_s2_Ra3 = (-s2/Ra2 + np.log(s2+Ra2))  - (-s1/Ra1 + np.log(s1+Ra1))

        u = np.zeros(3)
        for m in range(3):
            term2 = 0.0
            for i in range(3):
                for k in range(3):
                    term2 += b[i] * EPS[m, i, k] * t[k]
            term2 *= -I_1_Ra / (4.0 * np.pi)

            term3 = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        epsbt = EPS[i, j, k] * b[i] * t[k]
                        if abs(epsbt) < 1.0e-30:
                            continue
                        delta_mj = 1.0 if m == j else 0.0
                        bracket = (delta_mj * I_1_Ra
                                   - d[m]*d[j]              * I_1_Ra3
                                   - (d[m]*t[j]+t[m]*d[j])  * I_s_Ra3
                                   - t[m]*t[j]               * I_s2_Ra3)
                        term3 += epsbt * bracket
            term3 *= -1.0 / (8.0 * np.pi * (1.0 - nu))

            u[m] = term2 + term3

        return u

    # ---------- 主循环：每步调用两次 ----------

    def snapshot_before_step(self, N):
        """在 step_begin 调用一次：存这一步开始时的节点位置"""
        nodes_data = N.get_nodes_data()
        self._old_positions = nodes_data["positions"]

    def record_step(self, N):
        """在 step_post_integrate 调用一次：算这一步的位移增量"""
        if self._old_positions is None:
            raise RuntimeError("record_step 必须在 snapshot_before_step 之后调用")

        u_step = np.zeros((self.n_points, 3))
        nodes_data = N.get_nodes_data()
        new_positions = nodes_data["positions"]

        segs_data = N.get_segs_data()
        nodeids = segs_data["nodeids"]
        burgers = segs_data["burgers"]

        for seg_i in range(len(nodeids)):
            row_node, row_nbr = nodeids[seg_i]
            b = burgers[seg_i]

            node_new = new_positions[row_node]
            nbr_new_raw = new_positions[row_nbr]
            node_old = self._old_positions[row_node]
            nbr_old_raw = self._old_positions[row_nbr]

            nbr_new = N.cell.closest_image(Rref=node_new, R=nbr_new_raw)
            nbr_old = N.cell.closest_image(Rref=node_old, R=nbr_old_raw)

            quad = np.array([node_new, nbr_new, nbr_old, node_old])

            for p in range(self.n_points):
                xf = self.field_points[p]
                u_new = self._disp_line_segment(xf, node_new, nbr_new, b)
                u_old = self._disp_line_segment(xf, node_old, nbr_old, b)
                d_omega = self._solid_angle_polygon(quad, xf)
                du = (u_new - u_old) - b * d_omega / (4.0 * np.pi)
                u_step[p] += du

        self.u_total += u_step
        self._write_line(u_step)
        self._old_positions = None
        self._step_count += 1

    def _write_line(self, u_step):
        if self._file_handle is None:
            self._file_handle = open(self.output_file, "w")
            header = "# step"
            for p in range(self.n_points):
                header += f"  du{p}_x du{p}_y du{p}_z"
            for p in range(self.n_points):
                header += f"  u{p}_x u{p}_y u{p}_z"
            self._file_handle.write(header + "\n")

        line = f"{self._step_count}"
        for p in range(self.n_points):
            line += f" {u_step[p,0]:.10e} {u_step[p,1]:.10e} {u_step[p,2]:.10e}"
        for p in range(self.n_points):
            line += f" {self.u_total[p,0]:.10e} {self.u_total[p,1]:.10e} {self.u_total[p,2]:.10e}"
        self._file_handle.write(line + "\n")
        self._file_handle.flush()