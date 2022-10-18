from sim_data import *
from hecalib import Cal_Tsai,  Cal_Daniilidis, Cal_Chou

A = [HA1,HA2,HA3,HA4]
B = [HB1,HB2,HB3,HB4]
X = Hx

# cal = Cal_Tsai()
# cal = Cal_Chou()
cal = Cal_Daniilidis()
R,T, rel_err, abs_err = cal.calibrate(A,B,X)

print(R)
print(T)
print(rel_err)
print(abs_err)