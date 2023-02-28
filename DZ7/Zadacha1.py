#Домашняя работа
#Написать модуль кода в .py файле для дайльнейшего переиспользования в других домашних работах Модуль должен как минимум:

#Считать объемы продуктов сгорания
#Этальпию воздуха и продуктов сгорания
#PV=RT
#Процессов расширения (Опционально, все равно придется потом расширяя 
import numpy as np


class Gas:
    def __init__(self,
                CH4 = 0,
                C2H6 = 0,
                C3H8 = 0,
                C4H10 = 0,
                C5H12 = 0,
                N2 = 0,
                CO2 = 0,
                O2 = 0,
                CO = 0,
                H2 = 0,
                alpha = 1,
                temperature = 0): # как мы ее задем и какая размерность 
        self.CH4 = CH4
        self.C2H6 = C2H6
        self.C3H8 = C3H8
        self.C4H10 = C4H10
        self.C5H12 = C5H12
        self.N2 = N2
        self.CO2 = CO2
        self.O2 = O2
        self.CO = CO
        self.H2 = H2
        self.alpha = alpha
        self.temperature = temperature


        summa = self.CH4 + self.C2H6 + self.C3H8 + self.C4H10 + self.C5H12 + self.N2 + self.CO2 + self.O2 + self.CO + self.H2
        if summa != 100:
            raise ValueError(f"Общее процентное содержание должно быть 100 а задано: {summa}")

    @staticmethod
    def m_to_n(x):
        return (2 * x + 2)

    @property
    def heat_of_combustion(self):
        combustion_products = np.array([self.CH4,self.C2H6,self.C3H8,self.C4H10])  
        coeff = np.array([358.2,637.46,860.05,1185.8])  
        return sum(coeff*combustion_products)

    @property
    def stoichiometric_air_flow(self):
      combustion_products = np.array([self.CH4,self.C2H6,self.C3H8,self.C4H10,self.C5H12]) 
      V0 = []
      for indx, value in enumerate(combustion_products):
          m = np.array(indx + 1)
          n = np.array(self.m_to_n(m))
          V0.append(m+n/4 * value)
      return 0.0476 * sum(V0)

    @property
    def triatomic_gases_volume (self):
       combustion_products = [self.CH4,self.C2H6,self.C3H8,self.C4H10,self.C5H12]
       V0RO2 = []
       for indx, value in enumerate(combustion_products):
           m = np.array(indx + 1)
           n = np.array(self.m_to_n(m))
           V0RO2.append(m*value)
       V0RO2 = 0.01*(self.CO2 + sum(V0RO2))
       return V0RO2

    @property
    def water_volume (self):
       combustion_products = [self.CH4,self.C2H6,self.C3H8,self.C4H10,self.C5H12]
       V0HO2 = []
       for indx, value in enumerate(combustion_products):
           m = np.array(indx + 1)
           n = np.array(self.m_to_n(m))
           V0HO2.append((n/2)*value)
       V0HO2 = 0.01*(1.61*self.stoichiometric_air_flow + sum(V0HO2))
       return V0HO2

    @property
    def nitrogen_volume(self):
        return 0.79 * self.stoichiometric_air_flow + 0.01 * self.N2
    
    @property
    def actual_volume_of_water_vapor(self):
        return self.water_volume + 0.0161 * (self.alpha - 1)

    @property
    def full_volume_of_combustion_products(self):
        return self.triatomic_gases_volume + self.nitrogen_volume + self.actual_volume_of_water_vapor +(self.alpha - 1) * self.stoichiometric_air_flow  # тут точно азот? просто в формуле чтото другое 

    @property
    def heat_capacity_CO2(self):
        temp = np.array([self.temperature ** 3, self.temperature ** 2, self.temperature, 1])
        coeff = np.array([4.5784 * 10 ** (-11), -1.51719 * 10 ** (-7), 2.50113 * 10 ** (-4), 0.382325])
        return 4.1868*(sum(temp * coeff))

    @property
    def heat_capacity_N2(self):
        temp = np.array([self.temperature ** 3, self.temperature ** 2, self.temperature, 1])
        coeff = np.array([-2.24553 * 10 ** (-11),4.85082 * 10 ** (-8),-2.90598 * 10 ** (-6),0.309241])
        return 4.1868*(sum(temp*coeff))

    @property
    def heat_capacity_H20(self):
        temp = np.array([self.temperature ** 3, self.temperature ** 2, self.temperature, 1])
        coeff = np.array([-2.10956 * 10 ** (-11), 4.9732 * 10 ** (-8), 2.60629 * 10 ** (-5), 0.356691])
        return 4.1868*(sum(temp*coeff))

    @property
    def heat_capacity_air(self):
        temp = np.array([self.temperature ** 3, self.temperature ** 2, self.temperature, 1])
        coeff = np.array([-2.1717 * 10 ** (-11), 4.19344 * 10 ** (-8), 8.00891 * 10 ** (-6), 0.315027])
        return 4.1868*(sum(temp*coeff))

    @property
    def enthalpy_of_pure_combustion_products(self):
        return (self.triatomic_gases_volume * self.heat_capacity_CO2 + self.nitrogen_volume * self.heat_capacity_N2 + self.water_volume * self.heat_capacity_H20) * self.temperature

    @property
    def enthalpy_air(self):
        return (self.stoichiometric_air_flow * self.heat_capacity_air) * self.temperature

    @property
    def relative_enthalpy(self):
        return self.enthalpy_of_pure_combustion_products + (self.alpha - 1) / self.enthalpy_air


a = Gas(90.48,2.07,0.99,1.75,0.61,3.45,0.65,temperature=373.15 )        #Астраханское месторождение
print(a.heat_of_combustion)
print(a.stoichiometric_air_flow)
print(a.triatomic_gases_volume)
print(a.water_volume)
print(a.heat_capacity_CO2,)
print(a.heat_capacity_N2)
print(a.heat_capacity_H20)
print(a.heat_capacity_air)
print(a.enthalpy_of_pure_combustion_products)
print(a.enthalpy_air)
print(a.relative_enthalpy)











