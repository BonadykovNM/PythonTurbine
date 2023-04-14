from typing import Optional, Tuple, List, Union
from typing import Dict
import yaml
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GasClass:

    name: str = field(default="Астраханское")
    preinit: bool = field(default=True)
    _params_patn: str = field(default='./DZ7/condition.yaml')
    organic: Dict[str, Dict[str, Union[int,float]]] = field(default_factory=lambda:dict)
    N2: float = field(default=0)
    CO2: float = field(default=0)
    O2: float = field(default=0)
    C0: float = field(default=0)
    H2: float = field(default=0)
    alpha: float = field(default=1)

    def __load_params(self):
        with open(self._params_patn,'r+',encoding='UTF-8') as file:
            condition = yaml.safe_load(file)
        return condition

    def __post_init__(self):
        if self.preinit:
            condition  = self.__load_params()
            condition  = condition[self.name]
            for param in condition.keys():
                setattr(self,param,condition[param])

    @staticmethod
    def _m_to_n(x):
        return (2 * x + 2)

    def _get_org (self):
        org = [kwargs for kwargs in a.organic]
        combustion_product = []
        for i in list(range(len(a.organic.keys()))):
            combustion_product.append(a.organic.get(f'{org[i]}').get('percent'))
        return combustion_product


    @property
    def heat_of_combustion(self):
        combustion_products = np.array(self._get_org())
        combustion_products = combustion_products[:4]
        coeff = np.array([358.2,637.46,860.05,1185.8]) 
        return sum(coeff*combustion_products)

    @property
    def stoichiometric_air_expenditure(self):
        combustion_products = np.array(self._get_org())  
        V0 = []
        for indx, value in enumerate(combustion_products):
            m = np.array(indx + 1)
            n = np.array(self._m_to_n(m))
            V0.append(m+n/4 * value)
        return 0.0476 * sum(V0)

    @property
    def triatomic_gases_volume (self):
        combustion_products = np.array(self._get_org())
        V0RO3 = []
        for indx, value in enumerate(combustion_products):
            m = np.array(indx + 1)
            n = np.array(self._m_to_n(m))
            V0RO3.append(m*value)
        return 0.01*(self.CO2 + sum(V0RO3))

    @property
    def water_volume (self):
        combustion_products = np.array(self._get_org())
        V0HO2 = []
        for indx, value in enumerate(combustion_products):
             m = np.array(indx + 1)
             n = np.array(self._m_to_n(m))
             V0HO2.append((n/2)*value)
        V0HO2 = 0.01*(1.61*self.stoichiometric_air_expenditure + sum(V0HO2))
        return V0HO2

    @property    
    def nitrogen_volume(self):
        return 0.79 * self.stoichiometric_air_expenditure + 0.01 * self.N2
    
    @property
    def actual_volume_of_water_vapor(self):
        return self.water_volume + 0.0161 * (self.alpha - 1)


    @property
    def full_volume_of_combustion_products(self):
        return self.triatomic_gases_volume + self.nitrogen_volume + self.actual_volume_of_water_vapor +(self.alpha - 1) * self.stoichiometric_air_flow  
    
    
    def heat_capacity_CO2(self,t):
        temp = np.array([t ** 3, t ** 2, t, 1])
        coeff = np.array([4.5784 * 10 ** (-11), -1.51719 * 10 ** (-7), 2.50113 * 10 ** (-4), 0.382325])
        return 4.1868*(sum(temp * coeff))

    
    def heat_capacity_N2(self,t):
        temp = np.array([t ** 3, t ** 2, t, 1])
        coeff = np.array([-2.24553 * 10 ** (-11),4.85082 * 10 ** (-8),-2.90598 * 10 ** (-6),0.309241])
        return 4.1868*(sum(temp*coeff))
 
    
    def heat_capacity_H20(self,t):
        temp = np.array([t ** 3, t ** 2, t, 1])
        coeff = np.array([-2.10956 * 10 ** (-11), 4.9732 * 10 ** (-8), 2.60629 * 10 ** (-5), 0.356691])
        return 4.1868*(sum(temp*coeff))

    
    def heat_capacity_air(self,t):
        temp = np.array([t ** 3, t ** 2, t, 1])
        coeff = np.array([-2.1717 * 10 ** (-11), 4.19344 * 10 ** (-8), 8.00891 * 10 ** (-6), 0.315027])
        return 4.1868*(sum(temp*coeff))

 
    def enthalpy_of_pure_combustion_products(self,t):
        return (self.triatomic_gases_volume * self.heat_capacity_CO2(t) + self.nitrogen_volume * self.heat_capacity_N2(t) + self.water_volume * self.heat_capacity_H20(t)) * t

   
    def enthalpy_air(self,t):
        return (self.stoichiometric_air_expenditure * self.heat_capacity_air(t=t))

    def relative_enthalpy(self,t):
        return self.enthalpy_of_pure_combustion_products(t) + (self.alpha - 1) / self.enthalpy_air(t)
   
    def find_PV_RT (p = None,
                    v = None,
                    t = None):
        R = 8.314 
        if p == None:
            return (R*t)/v
        elif v == None:
            return (R*t)/p
        elif t == None:
            return (p*v)/R


    
a = GasClass(name = 'Уренгойский') 


print(a.heat_of_combustion)
print(a.stoichiometric_air_expenditure)
print(a.triatomic_gases_volume)
print(a.water_volume)
print(a.heat_capacity_CO2(t=300))
print(a.heat_capacity_N2(t=300))
print(a.heat_capacity_H20(t=300))
print(a.heat_capacity_air(t=300))
print(a.enthalpy_of_pure_combustion_products(t=300))
print(a.enthalpy_air(t=300))
print(a.relative_enthalpy(t=300))