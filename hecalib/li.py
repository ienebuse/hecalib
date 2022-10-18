'''R. Y. Tsai, R. K. Lenz et al., “A new technique for fully autonomous
and efficient 3 d robotics hand/eye calibration,” IEEE Transactions on
Robotics and Automation, vol. 5, no. 3, pp. 345–358, 1989.'''

import numpy as np
from numpy.linalg import inv, det, svd, eig, norm, pinv
from .hand_eye_calibration import Calibration


class Cal_Li(Calibration):
    
    def __init__(self) -> None:
        super().__init__()
        self.__S = np.empty(shape=[0,12])
        self.__T = np.empty(shape=[0,1])
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
        I9 = np.eye(9)

        for i in range(N):
            An = self._A[i]
            Bn = self._B[i]
            RA = An[:3,:3]
            tA = An[:3,3].reshape(3,1)
            tA_ = self._skew(tA)
            RB = Bn[:3,:3]
            tB = Bn[:3,3].reshape(3,1)
            
            s1 = np.hstack([I9 - np.kron(RB,RA), np.zeros((9,3))])
            s2 = np.hstack([np.kron(tB.T,tA_), np.dot(tA_,(I-RA))])
            S = np.vstack([s1, s2])
            T = np.vstack([np.zeros((9,1)),tA.reshape(3,1)])

            self.__S = np.vstack([self.__S,S])
            self.__T = np.vstack([self.__T,T])

        Rx_tX = self._solve_svd(self.__S)
        self._Rx,self._Tx = self.__get_rx_tx(Rx_tX)
        self._Rx = self._get_orth_mat(self._Rx)
        
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
    def __get_rx_tx(X):
        _Rx = X[:9].reshape(3,3)
        _tX = X[9:]

        w = det(_Rx)
        w = np.sign(w)/(abs(w)**(1/3))

        Rx = w*_Rx
        tX = w*_tX

        return Rx.T,tX

    
