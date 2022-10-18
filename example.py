from sim_data import *
from hecalib import Cal_Tsai,  Cal_Daniilidis, Cal_Chou, Cal_Park, Cal_Li

A = [HA1,HA2,HA3,HA4]
B = [HB1,HB2,HB3,HB4]
X = Hx

# cal = Cal_Tsai()
# cal = Cal_Chou()
# cal = Cal_Daniilidis()
# cal = Cal_Li()
cal = Cal_Park()
R,T, rel_err, abs_err = cal.calibrate(A,B,X)

print(R)
print(T)
print(rel_err)
print(abs_err)