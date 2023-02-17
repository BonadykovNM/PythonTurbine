
#Домашняя работа №1
#Задача №1
#Построить график зависимости термического КПД паротурбинного цикла без промежуточного перегрева пара при следующих параметрах пара:  P0= 5, 10, 15, 20 MPa. Для каждого значения взять следующие значения температуры  t0= 300, 350, 400, 450, 500 градусов Цельсия,  Pk= 5 kPa. Принять давление за последней ступенью паровой турбины  P2=Pk . Термический КПД цикла оценивать без учета подогрева воды в питательном насосе и регенеративной системе.
import iapws
from iapws import IAPWS97 as gas
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List, Union

point_type = iapws.iapws97.IAPWS97
MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
to_kelvin = lambda x: x + 273.15 if x else None
## Начальные параметры:
p0 = [5, 10, 15, 20] 
t0 = [300, 350, 400, 450, 500]
t0 = map(to_kelvin,t0)   
pk = 5 * kPa
t0 = list(t0)
def eff(p_0,t_0,p_k):
    p_0 = p_0 * MPa
    point_0 = gas(P=p_0 * unit, T=t_0)
    point_condenser_inlet = gas(P=p_k * unit, s=point_0.s)
    point_pump_outlet = gas(P=p_k * unit, x=0)
    useful_energy = point_0.h - point_condenser_inlet.h
    full_energy = point_0.h - point_pump_outlet.h
    efficiency = round(useful_energy/full_energy*100,3)
    return efficiency
efficiency = dict({})
for p0value in p0:
    efficiency[p0value]= []
    for t0value in t0:
        efficiency[p0value].append(eff(p0value,t0value,pk))

x = t0

plt.figure(layout = 'constrained')
plt.plot(x, efficiency[5], label = '5 MPa')  
plt.plot(x, efficiency[10], label = '10 MPa')  
plt.plot(x, efficiency[15], label = '15 MPa')
plt.plot(x, efficiency[20], label = '20 MPa')
plt.xlabel('Temperature')
plt.ylabel('KPD')
plt.title('The dependence of efficiency on temperature at different values of P0')
plt.legend();
plt.grid()
plt.show()
