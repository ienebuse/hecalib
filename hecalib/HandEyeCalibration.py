import numpy as np
from numpy.linalg import inv, det, svd, eig, norm, pinv
from scipy.spatial.transform import Rotation as Rot

class Calibration():

    def __init__(self) -> None:
        self._Rx = None
        self._Tx = None
        self.__PI = np.pi
        pass

    def calibrate(self, A, B,X=None,rel=False):
        if(isinstance(A,list) and isinstance(B,list) and (isinstance(X,list) or X is None)):
            A,B,X = self.__calibrate(A,B,X)
        elif(isinstance(A,np.ndarray) and isinstance(B,np.ndarray) and (isinstance(X,np.ndarray) or X is None)):
            pass
        else:
            assert False, \
                "Both A and B must be either Array or List"
        A = np.array(A)
        B = np.array(B)
        assert len(A.shape) == 3 and len(B.shape) == 3, \
            "Data set must be Nx4x4"
        assert A.shape[1:] == (4,4) and B.shape[1:] == (4,4), \
            "Input dimensions must be 4x4"
            
        
        if(not rel):
            self._A = self.__rel_pose(A)
            self._B = self.__rel_pose(B)
        else:
            self._A = A
            self._B = B

        self._X = X

    def __calibrate(self, A, B, X=None):
        A_ = list(map(lambda x: self._pose_2_txfrm(x),A))
        B_ = list(map(lambda x: self._pose_2_txfrm(x),B))
        X_ = self._pose_2_txfrm(X) if X is not None else None
        return A_,B_,X_

    def __rel_pose(self,A):
        '''Computes the relative poses from the absolute pose data'''
        a = np.empty(shape=[0,4,4])
        for i in range(1,A.shape[0]):
            a = np.append(a,np.dot(inv(A[i-1]),A[i]).reshape(1,4,4),axis=0)
        return a


    def _relative_error(self):
        '''Computes the relative error based on the data'''
        sumT = 0
        sumR = 0
        N = self._A.shape[0]
        for i in range(N):
            dR = np.dot(inv(np.dot(self._Rx,self._B[i][:3,:3])),np.dot(self._A[i][:3,:3],self._Rx))
            dR = self.__get_orth_mat(dR)
            dQ = self._mat_2_quaternion(dR)
            dR = 2*np.arccos(dQ[0])
            sumR = sumR + dR
            
            dT = np.dot(self._A[i][:3,:3]-np.eye(3),self._Tx).ravel() - np.dot(self._Rx,self._B[i][:3,3]).ravel() + self._A[i][:3,3].ravel()
            sumT = sumT + norm(dT)

        dT = sumT/N
        dR = sumR/N
        return dR%self.__PI, dT

    def _absolute_error(self):
        '''Computes the absolute error from based on the supplied ground truth'''
        assert self._X is not None, "X was not provided. Absolute error can only be computed with ground truth X"

        R = self.__get_orth_mat(self._Rx)
        dR = np.dot(inv(R),self._X[:3,:3])
        dQ = self._mat_2_quaternion(dR)
        dR = 2*np.arccos(dQ[0])
        dT = norm(self._X[:3,3].ravel()-self._Tx.ravel())
        return dR, dT


    def __get_orth_mat(self,R):
        '''Get the rotation matrix that satisfies othorgonality'''
        u,s,v = svd(R)
        return np.dot(u,v)
    

    def _pose_2_txfrm(self,pose):
        '''Compute the pose given [x,y,z,a,b,c]'''
        R = self._rot(pose[3], pose[4], pose[5])
        assert np.abs(det(R) - 1) < 1e-6, "det(R) is {}".format(det(R))
        t = np.array(pose[:3]).reshape(3,-1)
        H = np.vstack([np.hstack([R,t]),[[0,0,0,1]]])
        return H

    

    def _rot(self,x,y,z,rad=False):
        '''Computes the rotation matrix, given the independent euler rotations about x,y,z'''
        if(not rad):
            x = self._get_rad(x)
            y = self._get_rad(y)
            z = self._get_rad(z)

        R_x = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        R_y = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        R_z = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        return np.dot(R_z,np.dot(R_y,R_x))

    
    def _e(self,Q,p):
        Q = Q.ravel()
        if Q.size == 3:
            q0 = 0
            q = Q
        else:
            q0 = Q[0]
            q = Q[1:]
        g1 = np.hstack([np.array([[q0]]), -q.reshape(1,3)])
        g2 = np.hstack([q.reshape(3,1), q0*np.eye(3) - np.sign(p)*self._skew(q)])
        G = np.vstack([g1,g2])
        return G

    @staticmethod
    def _get_rad(deg):
        '''Converts angle in degree to radians'''
        theta = deg*np.pi/180
        return theta
    
    @staticmethod
    def _mat_2_angle_axis(R: np.ndarray):
        '''Converts rotation matrix to axis-angle representation'''
        rotvec = Rot.from_matrix(R).as_rotvec()
        theta = norm(rotvec)
        u = rotvec/theta
        return u.reshape(3,1), theta

    @staticmethod
    def _skew(x: np.ndarray)->np.ndarray:
        '''Transforms a vector to a skew symmetric matrix'''
        x = x.flatten()
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])

    @staticmethod
    def solve_svd(A: np.ndarray)->np.ndarray:
        '''Computes the null space of a matrix based on singular value decomposition'''
        U,S,VT = svd(A)

        '''Solution using matrix kernel'''
        x = VT.T[:,-1]
        return x

    @staticmethod
    def _quaternion_2_mat(q: np.ndarray)->np.ndarray:
        '''Converts a quaternion to a rotation matrixs representation'''
        q = q.flatten()
        R = np.array([[2*(q[0]*q[0] + q[1]*q[1]) - 1, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])], \
                    [2*(q[1]*q[2] + q[0]*q[3]), 2*(q[0]*q[0] + q[2]*q[2]) - 1, 2*(q[2]*q[3] - q[0]*q[1])], \
                    [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 2*(q[0]*q[0] + q[3]*q[3]) - 1]])
        return R

    @staticmethod
    def _mat_2_quaternion(R):
        qw= 0.5*np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
        qx = (R[2,1] - R[1,2])/( 4 *qw)
        qy = (R[0,2] - R[2,0])/( 4 *qw)
        qz = (R[1,0] - R[0,1])/( 4 *qw)
        return np.array([qw, qx, qy, qz])

    @staticmethod
    def _quat_dist(q1,q2):
        dist = np.arccos(np.abs(np.dot(q1.reshape(1,-1),q2.reshape(-1,1))))
        return dist

    
    