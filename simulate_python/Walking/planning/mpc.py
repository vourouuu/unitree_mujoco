import numpy as np
import proxsuite

class MPCPlanner:
    def __init__(self,N:int,dt:float,z_com:float):
        self.N=N
        self.dt=dt
        self.z_com=z_com
        self.g=9.81
        #page 36
        self.omega=np.sqrt(self.g/self.z_com)
        self.a=np.exp(self.omega*self.dt)
        #decision variables
        self.n_vars=4*self.N
        #equality constraints
        self.n_eq=2*self.N
        #inequality constraints xmin ymax etc
        self.n_in=4*self.N
        #Variables arbitrary for now
        self.Q_xi=10
        self.Q_zmp=1
        self.R_zmp=20
        
        self.qp=proxsuite.proxqp.dense.QP(self.n_vars,self.n_eq,self.n_in)

    def compute_trajectory(self,xi_meas:np.ndarray,zmp_ref:np.ndarray,support_polys:list)->tuple[np.ndarray, np.ndarray]:
        #Qp setup
        # Cost Hessian 2d dcm,2d zmp so 4Nx4N
        # Diagonal weights + Off diagonal rate of change 
        P=np.zeros((self.n_vars, self.n_vars))
        #Linear cost vector -Q*pzmp 4NxN
        q=np.zeros(self.n_vars)
        #Equality Constraint Matrix 2Nx4N
        A=np.zeros((self.n_eq, self.n_vars))
        #Equality constraint xi 2NxN
        b=np.zeros(self.n_eq)
        #Inequality constraint foot
        C=np.zeros((self.n_in, self.n_vars))
        #Lower Bounds 4NxN
        l=np.zeros(self.n_in)
        #Upper Bounds 4NxN
        u=np.zeros(self.n_in)

        for i in range(self.N):
            xi_i=i*2
            zmp_i=self.N*2+i*2
            #Diagonal weights 
            P[xi_i:xi_i+2,xi_i:xi_i+2] =self.Q_xi*np.eye(2)
            P[zmp_i:zmp_i+2,zmp_i:zmp_i+2]+=self.Q_zmp*np.eye(2)
            #Linear cost
            q[zmp_i:zmp_i+2]=-self.Q_zmp*zmp_ref[i]
            
            if i < (self.N-1):
                #smoothing
                next_zmp_i=zmp_i+2
                P[zmp_i:zmp_i+2, zmp_i:zmp_i+2]+=self.R_zmp*np.eye(2)
                P[zmp_i+2:zmp_i+4, zmp_i+2:zmp_i+4]+= self.R_zmp*np.eye(2)
                P[zmp_i:zmp_i+2, zmp_i+2:zmp_i+4]-= self.R_zmp*np.eye(2)
                P[zmp_i+2:zmp_i+4, zmp_i:zmp_i+2]-= self.R_zmp*np.eye(2)

        # Discrete DCM
        for i in range(self.N):
            eq_i=i*2
            xi_i=i*2
            zmp_i=self.N*2+i*2
            
            A[eq_i:eq_i+2,xi_i:xi_i+2]=np.eye(2)
            A[eq_i:eq_i+2,zmp_i:zmp_i+2]=-(1-self.a)*np.eye(2)
            
            if i==0:
                b[eq_i:eq_i+2]=self.a*xi_meas
            else:
                prev_xi_i=(i-1)*2
                
                A[eq_i:eq_i+2,prev_xi_i:prev_xi_i+2] =-self.a*np.eye(2)

        #Inequality Constraints from Polygons
        for i in range(self.N):
            in_i=i*4
            zmp_i=self.N*2+i*2
            foot=support_polys[i]# [x_min, x_max, y_min, y_max]
            
            C[in_i:in_i+4, zmp_i:zmp_i+2] = np.array([[1, 0],[-1, 0],[0, 1],[0, -1]])
            u[in_i:in_i+4] = np.array([foot[1],-foot[0],foot[3],-foot[2]])
            l[in_i:in_i+4]=-1e20#unconstrained

        self.qp.init(P,q,A,b,C,l,u)
        self.qp.solve()
        
        sol = self.qp.results.x
        opt_xi=sol[:self.N * 2].reshape((self.N, 2))
        opt_zmp=sol[self.N *2:].reshape((self.N, 2))
        
        #page 39
        beta=np.exp(-self.omega*self.dt)
        com_traj=np.zeros((self.N, 3))
        com=np.array([opt_xi[0,0],opt_xi[0,1],self.z_com])
        
        for i in range(self.N):
            if i>0:
                com[:2]=beta*com[:2]+(1-beta)*opt_xi[i-1]
            com_traj[i]=[com[0],com[1],self.z_com]
            
        return com_traj, opt_zmp