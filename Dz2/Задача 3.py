#������ 3
#��� �������� �������� �������� ���� d01= 3,0 ��/(����) d02� = 3,6 ��/(����) ������� �������� ������� ������� �� ��������� ��������������, ������ �������� ��������� h0�h�� = 2500 ���/��.

import iapws
from iapws import IAPWS97 as gas
def expenditure (d0, h0_hpv):
    Qty = d0 * (h0_hpv)
    return Qty
d0 = [3.0, 3.6] 
h0_hpv = 2500
exp = []
for d0value in d0:
    exp.append(expenditure(d0value,h0_hpv))
print(exp,'kJ/kW*h')
