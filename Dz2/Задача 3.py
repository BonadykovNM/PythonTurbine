#«адача 3
#ѕри удельных расходах вод€ного пара d01= 3,0 кг/(к¬тЈч) d02и = 3,6 кг/(к¬тЈч) оценить удельные расходы теплоты на выработку электроэнергии, прин€в разность энтальпий h0Цhпв = 2500 кƒж/кг.

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
