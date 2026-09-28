import numpy as np
import proxsuite


class MPCPlanner:
    def __init__(self, N: int, dt: float, z_com: float):
        self.N = N
        self.dt = dt
        self.z_com = z_com
        self.g = 9.81
        
        self.omega = np.sqrt(self.g / self.z_com)
        self.a = np.exp(self.omega * self.dt)
        self.beta = np.exp(-self.omega * self.dt)
        
        # Decision variables: [xi_0..xi_N-1, zmp_0..zmp_N-1]
        self.n_vars = 4 * self.N
        self.n_eq = 2 * self.N
        self.n_in = 4 * self.N
        
        # Cost Weights
        self.Q_xi = 10.0
        self.Q_zmp = 500.0
        self.R_zmp = 5.0
        
        self.qp = proxsuite.proxqp.dense.QP(self.n_vars, self.n_eq, self.n_in)

    def compute_trajectory(self, com_meas: np.ndarray, xi_meas: np.ndarray, zmp_ref: np.ndarray, support_polys: list) -> tuple[np.ndarray, np.ndarray]:
        P = np.zeros((self.n_vars, self.n_vars))
        q = np.zeros(self.n_vars)
        A = np.zeros((self.n_eq, self.n_vars))
        b = np.zeros(self.n_eq)
        C = np.zeros((self.n_in, self.n_vars))
        l = np.zeros(self.n_in)
        u = np.zeros(self.n_in)

        # 1. Cost Matrix Construction
        for i in range(self.N):
            xi_i = i * 2
            zmp_i = self.N * 2 + i * 2
            
            P[xi_i:xi_i+2, xi_i:xi_i+2] = self.Q_xi * np.eye(2)
            P[zmp_i:zmp_i+2, zmp_i:zmp_i+2] += self.Q_zmp * np.eye(2)
            q[zmp_i:zmp_i+2] = -self.Q_zmp * zmp_ref[i]
            
            if i < (self.N - 1):
                next_zmp_i = zmp_i + 2
                P[zmp_i:zmp_i+2, zmp_i:zmp_i+2] += self.R_zmp * np.eye(2)
                P[next_zmp_i:next_zmp_i+2, next_zmp_i:next_zmp_i+2] += self.R_zmp * np.eye(2)
                P[zmp_i:zmp_i+2, next_zmp_i:next_zmp_i+2] -= self.R_zmp * np.eye(2)
                P[next_zmp_i:next_zmp_i+2, zmp_i:zmp_i+2] -= self.R_zmp * np.eye(2)

        # 2. Discrete DCM Dynamics (xi_k = a * xi_k-1 + (1-a) * zmp_k-1)
        for i in range(self.N):
            eq_i = i * 2
            xi_i = i * 2
            zmp_i = self.N * 2 + i * 2
            
            A[eq_i:eq_i+2, xi_i:xi_i+2] = np.eye(2)
            
            if i == 0:
                # First step: xi_0 = a * xi_meas + (1-a) * zmp_meas
                b[eq_i:eq_i+2] = self.a * xi_meas[:2]
                A[eq_i:eq_i+2, zmp_i:zmp_i+2] = -(1.0 - self.a) * np.eye(2)
            else:
                prev_xi_i = (i - 1) * 2
                prev_zmp_i = self.N * 2 + (i - 1) * 2
                A[eq_i:eq_i+2, prev_xi_i:prev_xi_i+2] = -self.a * np.eye(2)
                A[eq_i:eq_i+2, prev_zmp_i:prev_zmp_i+2] = -(1.0 - self.a) * np.eye(2)

        # 3. Foot Support Polygon Inequalities
        for i in range(self.N):
            in_i = i * 4
            zmp_i = self.N * 2 + i * 2
            foot = support_polys[i]  # [x_min, x_max, y_min, y_max]
            
            C[in_i:in_i+4, zmp_i:zmp_i+2] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
            u[in_i:in_i+4] = np.array([foot[1], -foot[0], foot[3], -foot[2]])
            l[in_i:in_i+4] = -1e20

        # Solve QP
        self.qp.init(P, q, A, b, C, l, u)
        self.qp.solve()
        
        sol = self.qp.results.x
        opt_xi = sol[:self.N * 2].reshape((self.N, 2))
        opt_zmp = sol[self.N * 2:].reshape((self.N, 2))
        
        # 4. Forward CoM Integration
        com_traj = np.zeros((self.N, 3))
        curr_com = com_meas[:2].copy()
        
        for i in range(self.N):
            curr_com = self.beta * curr_com + (1.0 - self.beta) * opt_xi[i]
            com_traj[i] = [curr_com[0], curr_com[1], self.z_com]
            
        return com_traj, opt_zmp