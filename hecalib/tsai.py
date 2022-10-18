'''R. Y. Tsai, R. K. Lenz et al., “A new technique for fully autonomous
and efficient 3 d robotics hand/eye calibration,” IEEE Transactions on
Robotics and Automation, vol. 5, no. 3, pp. 345–358, 1989.'''

import numpy as np
from numpy.linalg import inv, det, svd, eig, norm, pinv
from .hand_eye_calibration import Calibration


class Cal_Tsai(Calibration):
    
    def __init__(self) -> None:
        super().__init__()
        self.__S = np.empty(shape=[0,3])
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

        for i in range(N):
            An = self._A[i]
            Bn = self._B[i]
            RA = An[:3,:3]
            tA = An[:3,3].reshape(3,1)
            RB = Bn[:3,:3]
            tB = Bn[:3,3].reshape(3,1)
            
            uA, wA = self._mat_2_angle_axis(RA)
            uB, wB = self._mat_2_angle_axis(RB)

            S = self._skew(uA + uB)
            T = uB - uA
            RA_I = RA - I

            self.__S = np.vstack([self.__S,S])
            self.__T = np.vstack([self.__T,T])
            self.__RA_I = np.vstack([self.__RA_I,RA_I])
            self.__TA = np.vstack([self.__TA,tA])
            self.__TB = np.vstack([self.__TB,tB])

        ux = self._solve_ls(self.__S,self.__T)
        uX = 2*ux/(np.sqrt(1+norm(ux)**2))
        self._Rx = (1-norm(uX)**2/2)*np.eye(3) + 0.5*(uX*uX.T + np.sqrt(4-norm(uX)**2)*self._skew(uX))
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

    
