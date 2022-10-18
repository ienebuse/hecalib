'''K. Daniilidis and E. Bayro-Corrochano, "The dual quaternion approach to hand-eye calibration,
" Proceedings of 13th International Conference on Pattern Recognition, 1996, pp. 318-322 vol.1, 
 doi: 10.1109/ICPR.1996.546041.
 '''

import numpy as np
from numpy.linalg import inv, det, svd, eig, norm, pinv
from scipy import optimize
from hecalib.HandEyeCalibration import Calibration


class Cal_Daniilidis(Calibration):
    
    def __init__(self) -> None:
        super().__init__()
        self.__S = np.empty(shape=[0,8])
        self.__RA_I = np.empty(shape=[0,3])
        self.__TA = np.empty(shape=[0,1])
        self.__TB = np.empty(shape=[0,1])

    
    def calibrate(self, A,B,X=None,rel=False):
        '''Computes the estimated rotation matrix and translation vector as well as the relative and 
           absolute (if the ground truth X is provided) rotation and translation error
        '''
        super().calibrate(A,B,X,rel)
        N = self._A.shape[0]
        I = np.eye(3)

        for i in range(N):
            An = self._A[i]
            Bn = self._B[i]
            RA = An[:3,:3]
            tA = An[:3,3].reshape(3,1)
            RB = Bn[:3,:3]
            tB = Bn[:3,3].reshape(3,1)
            
            qA = self._mat_2_quaternion(RA).reshape(-1,1)
            qB = self._mat_2_quaternion(RB).reshape(-1,1)
            
            qA_ = 0.5*np.dot(self._e(qA,1),np.vstack([[[0]],tA]))
            qB_ = 0.5*np.dot(self._e(qB,1),np.vstack([[[0]],tB]))

            s10 = np.hstack([qA[1:]-qB[1:], self._skew(qA[1:]+qB[1:])])
            s11 = np.hstack([np.zeros((3,1)),np.zeros((3,3))])
            s1 = np.hstack([s10,s11])

            s20 = np.hstack([qA_[1:]-qB_[1:], self._skew(qA_[1:]+qB_[1:])])
            s21 = np.hstack([qA[1:]-qB[1:], self._skew(qA[1:]+qB[1:])])
            s2 = np.hstack([s20,s21])

            S = np.vstack([s1,s2])        
            self.__S = np.vstack([self.__S, S])

        _,_,VT = svd(self.__S)
        UV = VT.T[:,-2:]
        u1 = UV[:4,0].reshape(-1,1)
        v1 = UV[4:,0].reshape(-1,1)
        u2 = UV[:4,1].reshape(-1,1)
        v2 = UV[4:,1].reshape(-1,1)

        a = np.dot(u1.T,u1)[0][0] 
        a_ = np.dot(u1.T,v1)[0][0]
        b = np.dot(u1.T,u2)[0][0] + np.dot(u2.T,u1)[0][0]
        b_ = np.dot(u1.T,v2)[0][0] + np.dot(u2.T,v1)[0][0]
        c = np.dot(u2.T,u2)[0][0] 
        c_ = np.dot(u2.T,v2)[0][0]

        self.__J = [[a,b,c],[a_,b_,c_]]
        x0 = np.array([0.5,0.5])
        x = optimize.newton(self.__f,x0,maxiter=100000,tol=1.48e-15)

        qx = x[0]*u1 + x[1]*u2
        qx_ = x[0]*v1 + x[1]*v2

        self._Rx = self._quaternion_2_mat(qx)

        self._Tx = 2*np.dot(inv(self._e(qx,1)),qx_)[1:]

        rel_err = self._relative_error()

        if(X is not None):
            abs_err = self._absolute_error()
            return self._Rx, self._Tx, rel_err, abs_err

        return self._Rx, self._Tx, rel_err

    
    def __f(self,x):
        C = np.array([self.__J[0][0]*x[0]**2 + self.__J[0][1]*x[0]*x[1] + self.__J[0][2]*x[1]**2, \
                    self.__J[1][0]*x[0]**2 + self.__J[1][1]*x[0]*x[1] + self.__J[1][2]*x[1]**2])
        E = C - np.array([1,0])
        return E

    