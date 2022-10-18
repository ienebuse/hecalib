'''F. C. Park and B. J. Martin, “Robot sensor calibration: solving AX
= XB on the Euclidean group,” IEEE Transactions on Robotics and
Automation, vol. 10, no. 5, pp. 717–721, 1994.'''

import numpy as np
from numpy import outer
from numpy.linalg import inv, det, svd, eig, norm, pinv
from .hand_eye_calibration import Calibration


class Cal_Park(Calibration):
    
    def __init__(self) -> None:
        super().__init__()
        # self.__M = np.zeros((3,3))
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
        M = np.zeros((3,3))

        for i in range(N):
            An = self._A[i]
            Bn = self._B[i]
            RA = An[:3,:3]
            tA = An[:3,3].reshape(3,1)
            RB = Bn[:3,:3]
            tB = Bn[:3,3].reshape(3,1)

            M += outer(self.__log(RB), self.__log(RA))

            RA_I = RA - I
            self.__RA_I = np.vstack([self.__RA_I,RA_I])
            self.__TA = np.vstack([self.__TA,tA])
            self.__TB = np.vstack([self.__TB,tB])

        self._Rx = np.dot(self.__invsqrt(np.dot(M.T, M)), M.T)
        self._Tx = self.__get_translation(self._Rx,self.__RA_I,self.__TA,self.__TB).reshape(3,1)

        rel_err = self._relative_error()

        if(X is not None):
            abs_err = self._absolute_error()
            return self._Rx, self._Tx, rel_err, abs_err

        return self._Rx, self._Tx, rel_err

    def __get_translation(self,R,RA_I,TA,TB):
        '''Computes the translation estimate based on the estimated rotation'''
        RxTB = np.dot(R,TB[:3,0]).reshape(3,1)
        for i in range(1,int((TB.shape[0])/3)):
            RxTB = np.append(RxTB,np.dot(R,TB[i*3:(i+1)*3,0].reshape(3,1)),axis=0)
        
        T = RxTB - TA
        tX = np.dot(pinv(RA_I),T)

        return tX

    @staticmethod
    def __log(R):
        '''Rotation matrix logarithm'''
        theta = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0)
        return np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))

    @staticmethod
    def __invsqrt(mat):
        u,s,v = np.linalg.svd(mat)
        return u.dot(np.diag(1.0/np.sqrt(s))).dot(v)

    
