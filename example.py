import numpy as np
from sim_data import *
from TsaiLenz import Cal_TsaiLenz

A = [HA1,HA2,HA3,HA4]
B = [HB1,HB2,HB3,HB4]
X = Hx

cal_tl = Cal_TsaiLenz()
R,T, rel_err, abs_err = cal_tl.calibrate(A,B,X)

print(R)
print(T)
print(rel_err)
print(abs_err)