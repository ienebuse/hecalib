from sim_data import *
from hecalib.TsaiLenz import Cal_TsaiLenz
from hecalib.Daniilidis import Cal_Daniilidis

A = [HA1,HA2,HA3,HA4]
B = [HB1,HB2,HB3,HB4]
X = Hx

# cal = Cal_TsaiLenz()
cal = Cal_Daniilidis()
R,T, rel_err, abs_err = cal.calibrate(A,B,X)

print(R)
print(T)
print(rel_err)
print(abs_err)