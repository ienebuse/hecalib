'''Y.-C. Lu and J. C. Chou, “Eight-space quaternion approach for robotic
hand-eye calibration,” IEEE International Conference on Systems, Man
and Cybernetics. Intelligent Systems for the 21st Century, pp. 3316–3321,
1995'''

import numpy as np
from numpy.linalg import inv, det, svd, eig, norm, pinv
from .hand_eye_calibration import Calibration


class Cal_Lu(Calibration):
    
    def __init__(self) -> None:
        super().__init__()
    
    def calibrate(self, A,B,X=None,rel=False):
        '''Computes the estimated rotation matrix and translation vector as well as the relative and 
           absolute (if the ground truth X is provided) rotation and translation error
        '''
        super().calibrate(A,B,X,rel)
        N = self._A.shape[0]
        I = np.eye(3)
        U_sum = np.zeros((4,4))
        V_sum = np.zeros((4,4))
        W_sum = np.zeros((4,4))

        for i in range(N):
            An = self._A[i]
            Bn = self._B[i]
            RA = An[:3,:3]
            tA = An[:3,3].reshape(3,1)
            RB = Bn[:3,:3]
            tB = Bn[:3,3].reshape(3,1)
            
            qA = self._mat_2_quaternion(RA)
            qB = self._mat_2_quaternion(RB)

            a = np.dot(self._e(qB,1), self._e(tB,1) - self._e(tA,-1))
            b = self._e(qB,1) - self._e(qA,-1)

            U_sum += np.dot(a.T,a) + np.dot(b.T,b)
            V_sum += np.dot(a.T,b)
            W_sum += np.dot(b.T,b)

        W_ = W_sum - np.dot(V_sum.T,np.dot(inv(U_sum),V_sum))

        phi = np.dot(inv(W_[:2,:2]-W_[2:,:2]),W_[2:,2:]-W_[:2,2:])
        alpha = -np.dot(inv(U_sum),V_sum).T
        omega = np.dot(phi.T,np.dot(alpha[:2,:2],phi)) + np.dot(alpha[2:,:2],phi) + np.dot(phi.T, alpha[:2,2:]) + alpha[2:,2:]
        beta = np.dot(alpha,alpha.T)
        Phi = np.dot(phi.T,np.dot(beta[:2,:2],phi)) + np.dot(beta[2:,:2],phi) + np.dot(phi.T, beta[:2,2:]) + beta[2:,2:]

        h11 = -((omega[0,1]+omega[1,0]) + np.sqrt((omega[0,1]+omega[1,0])**2 - 4*omega[0,0]*omega[1,1]))/(2*omega[0,0])
        h12 = -((omega[0,1]+omega[1,0]) - np.sqrt((omega[0,1]+omega[1,0])**2 - 4*omega[0,0]*omega[1,1]))/(2*omega[0,0])

        h21 = Phi[1,1] + (Phi[0,1]+Phi[1,0])*h11 + Phi[0,0]*h11*h11
        h22 = Phi[1,1] + (Phi[0,1]+Phi[1,0])*h12 + Phi[0,0]*h12*h12

        if(h21 > 0):
            r_ = np.array([[phi[0,0]*h11 + phi[0,1]], \
                        [phi[1,0]*h11 + phi[1,1]], \
                        [h11], \
                        [1]]) * np.sqrt(1/h21)
        else:
            r_ = np.array([[phi[0,0]*h12 + phi[0,1]], \
                        [phi[1,0]*h12 + phi[1,1]], \
                        [h12], \
                        [1]]) * np.sqrt(1/h22)

        qx = np.dot(alpha.T,r_)

        self._Tx = np.dot(self.__e(qx),r_)

        self._Rx = self._quaternion_2_mat(qx).reshape(3,3)

        rel_err = self._relative_error()

        if(X is not None):
            abs_err = self._absolute_error()
            return self._Rx, self._Tx, rel_err, abs_err

        return self._Rx, self._Tx, rel_err
        

    def __e(self,Q):
        q0 = Q[0]
        q = Q[1:]
        G = np.append(-q.reshape(3,1), q0*np.eye(3) + self._skew(q), axis=1)
        return G

    
