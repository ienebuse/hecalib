'''J.C. Chou, M. Kamel, Quaternions approach to solve the kinematic 
equation of rotation, AX = XB, of a sensor-mounted robotic manipulator, 
in IEEE International Conference on Robotics and Automation (IEEE, 1988), 
pp. 656–662'''

import numpy as np
from numpy.linalg import inv, det, svd, eig, norm, pinv
from .hand_eye_calibration import Calibration


class Cal_Chou(Calibration):
    
    def __init__(self) -> None:
        super().__init__()
        self.__G = np.empty(shape=[0,4])
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

            g1 = np.hstack([np.array([[0]]), -(uA-uB).T])
            g2 = np.hstack([uA-uB, self._skew(uA)+self._skew(uB)])
            G = np.vstack([g1,g2])
            RA_I = RA - I


            self.__G = np.vstack([self.__G,G])
            self.__RA_I = np.vstack([self.__RA_I,RA_I])
            self.__TA = np.vstack([self.__TA,tA])
            self.__TB = np.vstack([self.__TB,tB])

        x = self._solve_svd(self.__G)
        self._Rx = self._quaternion_2_mat(x.ravel())
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

    
