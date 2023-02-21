#Домашняя работа
#Написать модуль кода в .py файле для дайльнейшего переиспользования в других домашних работах Модуль должен как минимум:

#Считать объемы продуктов сгорания
#Этальпию воздуха и продуктов сгорания
#PV=RT
#Процессов расширения (Опционально, все равно придется потом расширя
import sys

class gas:
    def __init__(self,
                CH4=0,
                C2H6 = 0,
                C3H8 = 0,
                C4H10 = 0,
                N2 = 0,
                CO2 = 0,
                O2 = 0,
                CO = 0,
                H2 = 0):
        self.CH4 = CH4
        self.C2H6 = C2H6
        self.C3H8 = C3H8
        self.C4H10 = C4H10
        self.N2 = N2
        self.CO2 = CO2
        self.O2 = O2
        self.CO = CO
        self.H2 = H2
        sum = CH4+C2H6+C3H8+C4H10+N2+CO2+O2+CO+H2
        if (sum>100 or sum<100):
          print("Общее процентное содержание должно быть 100 а задано:",sum)
          sys.exit()
    @property
    def Qnp (self):
      return 358.2*self.CH4+637.46*self.C2H6+860.05*self.C3H8+1185.8*self.C4H10
    @property
    def V0(self):
      fire = [self.CH4,self.C2H6,self.C3H8,self.C4H10]
      V0=[]
      for i in fire:
          V0.append((((fire.index (i))+1)+((2*((fire.index (i))+1)+2)/4)*i))   #  я не уверент в формуле 
      V0 = 0.0476 * sum(V0)
      return V0
    @property
    def V0RO2 (self):
       fire = [self.CH4,self.C2H6,self.C3H8,self.C4H10]
       V0RO2 = []
       for i in fire:
          V0RO2.append(fire.index (i)*i)
       V0RO2 = 0.01*(self.CO2 + sum(V0RO2))
       return V0RO2
a = gas(10,20,30,30,0,10)
print(a.V0RO2)







