import random
import matplotlib.pyplot as plt
from numpy.random import choice
import math
import numpy as np

class Exploradora:
  def __init__(self):
        self.posicao = []
        self.melhor_posicao = []
        self.trabalhadoras = []
        self.ciclos_sem_melhora = 0
        self.fitness = np.inf
        self.melhor_fitness = np.inf

class Trabalhadora:
  def __init__(self):
        self.posicao = []
        self.fitness = np.inf

def funcao_fitness(lista_solucao,funcao):
  if funcao=='ackley':
    return ackley(lista_solucao)
  elif funcao=='rastrigin':
    return rastrigin(lista_solucao)
  elif funcao=='rosenbrock':
    return rosenbrock(lista_solucao)
  elif funcao=='esfera':
    return esfera(lista_solucao)

def esfera(lista_solucao):
  total = 0
  return sum([coord ** 2 for coord in lista_solucao])

def ackley(lista_solucao):
	firstSum = 0.0
	secondSum = 0.0
	for c in lista_solucao:
		firstSum += c**2.0
		secondSum += math.cos(2.0*math.pi*c)
	n = float(len(lista_solucao))
	return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e

def rosenbrock(lista_solucao):
  sum_ = 0.0
  for i in range(1, len(lista_solucao) - 1):
      sum_ += 100 * (lista_solucao[i + 1] - lista_solucao[i] ** 2) ** 2 + (lista_solucao[i] - 1) ** 2
  return sum_

def rastrigin(lista_solucao):
  f_x = [xi ** 2 - 10 * math.cos(2 * math.pi * xi) + 10 for xi in lista_solucao]
  return sum(f_x)

def function_bounds(funcao):
  if funcao=='ackley':
    return [-32,32]
  elif funcao=='rastrigin':
    return [-5.12, 5.12]
  elif funcao=='rosenbrock':
    return [-30.0, 30.0]
  elif funcao=='esfera':
    return [-100.0, 100.0]

class ABC:
    def __init__(self,n,limite,cmax,dimensao,funcao,movimentacao):
        self.n = n
        self.nf = int(self.n / 2)
        self.limite = limite
        self.cmax = cmax
        self.exploradoras =[]
        self.trabalhadoras = []
        self.colonia = None
        self.dimensao = dimensao
        self.bound = function_bounds(funcao)
        self.funcao = funcao
        self.movimentacao = movimentacao

    def inicialization(self):
        for i in range(self.nf):
          exploradora = Exploradora()
          for i in range (self.dimensao):
            exploradora.posicao.append(random.uniform(self.bound[0], self.bound[1]))
          exploradora.melhor_posicao = exploradora.posicao.copy()
          exploradora.fitness = funcao_fitness(exploradora.posicao,self.funcao)
          exploradora.melhor_fitness = exploradora.fitness
          self.exploradoras.append(exploradora)
          trabalhadora = Trabalhadora()
          self.trabalhadoras.append(trabalhadora)

        self.colonia = self.exploradoras + self.trabalhadoras

    def alocar_abelhas(self,abelhas):
        # faz a soma dos fitness das abelhas exploradoras
        soma_fitness = 0
        for i in range(self.nf):
            if self.exploradoras[i].fitness >=0:
              soma_fitness += 1/(1+self.exploradoras[i].fitness)
            else:
              soma_fitness += 1+abs(self.exploradoras[i].fitness)
        # cria a distribuição de probabilidades de acordo com o fitness calculado
        distribuicao_probabilidade = []
        for i in range(self.nf):
            if self.exploradoras[i].fitness >=0:
              probabilidade_alocacao = (1/(1+self.exploradoras[i].fitness)) / soma_fitness
            else:
              probabilidade_alocacao = (1+abs(self.exploradoras[i].fitness)) / soma_fitness
            
            distribuicao_probabilidade.append(probabilidade_alocacao)
        # posiciona as oportunitas de acordo com a atratividade das exploradoras        
        for abelha in abelhas:
            exploradora = choice(self.exploradoras, 1, p=distribuicao_probabilidade)[0]
            abelha.posicao = exploradora.posicao.copy()
            abelha.fitness = funcao_fitness(abelha.posicao,self.funcao)
            exploradora.trabalhadoras.append(abelha)


    def exibir_colonia(self):
        for i in range(self.nf):
            abelha = self.exploradoras[i]
            x, y = zip(abelha.posicao)
            plt.plot(x, y, marker='x', color='k')

        plt.plot(0,0, marker='*', markersize=10, color='b')
        plt.axis([-100, 100, -100, 100])
        plt.show()

    def mov_tradicional(self,abelha):
      colonia_candidata = self.colonia.copy()
      colonia_candidata.remove(abelha)
      abelha_candidata = random.choice(colonia_candidata)
      return abelha_candidata

    def mov_dentro_colonia(self,abelha,trabalhadoras_colonia):
      colonia_candidata = trabalhadoras_colonia.copy()
      if len(colonia_candidata) == 1:
        return self.mov_tradicional(abelha)
      else:
        colonia_candidata.remove(abelha)
        abelha_candidata = random.choice(colonia_candidata)
        return abelha_candidata

    def mov_probabilidade_geral(self,abelha):
      colonia_candidata = self.colonia.copy()
      colonia_candidata.remove(abelha)
      # faz a soma dos fitness das abelhas exploradoras
      soma_fitness = 0
      for i in range(self.n-1):
          soma_fitness += colonia_candidata[i].fitness
      # cria a distribuição de probabilidades de acordo com o fitness calculado
      distribuicao_probabilidade = []
      for i in range(self.n-1):
          probabilidade_alocacao = colonia_candidata[i].fitness / soma_fitness
          distribuicao_probabilidade.append(probabilidade_alocacao)
      
      #escolhe a abelha canditada de acordo com o fitness
      abelha_candidata = choice(colonia_candidata, 1, p=distribuicao_probabilidade)[0]
      return abelha_candidata

    def movimenta_abelha(self,abelha,trabalhadoras_colonia):
      if self.movimentacao == 'tradicional':
        abelha_candidata = self.mov_tradicional(abelha)
      elif self.movimentacao == 'roleta':
        abelha_candidata = self.mov_probabilidade_geral(abelha)
      elif self.movimentacao == 'colmeia':
        abelha_candidata = self.mov_dentro_colonia(abelha,trabalhadoras_colonia)
      for i in range (self.dimensao):
        abelha.posicao[i] = abelha.posicao[i] + random.uniform(-1, 1) * (abelha.posicao[i] - abelha_candidata.posicao[i])        
        if abelha.posicao[i] < self.bound[0]:
            abelha.posicao[i] = self.bound[0]
        elif abelha.posicao[i] > self.bound[1]:
            abelha.posicao[i] = self.bound[1]

      abelha.fitness = funcao_fitness(abelha.posicao,self.funcao)

    def optimize(self):
        melhor_posicao = None
        melhor_fitness = np.inf
        track_fitness = []
        self.inicialization()
        self.alocar_abelhas(self.trabalhadoras)
        for i in range(self.cmax):
          # alocando as abelhas oportunistas como trabalhadoras em suas fontes
          for exploradora in self.exploradoras:
              # movimentando as trabalhadoras de cada fonte de comida
              for trabalhadora in exploradora.trabalhadoras:
                self.movimenta_abelha(trabalhadora,exploradora.trabalhadoras)
              # obtem a melhor posição da iteração
              melhor_posicao_inter = None
              melhor_fitness_inter = np.inf
              for trabalhadora in exploradora.trabalhadoras:
                  trabalhadora_fitness = funcao_fitness(trabalhadora.posicao,self.funcao)
                  if melhor_posicao_inter is None or trabalhadora_fitness < melhor_fitness_inter:
                      melhor_posicao_inter = trabalhadora.posicao.copy()
                      melhor_fitness_inter = trabalhadora_fitness
              
              # atualiza posição atual da abelha exploradora
              if melhor_posicao_inter is not None and melhor_fitness_inter < exploradora.fitness:
                  exploradora.posicao = melhor_posicao_inter.copy()
                  exploradora.fitness = melhor_fitness_inter
                  exploradora.ciclos_sem_melhora = 0
              else:
                  exploradora.ciclos_sem_melhora += 1
              
              
              # atualiza melhor posicao da exploradora
              if melhor_posicao_inter is not None and melhor_fitness_inter < exploradora.melhor_fitness:
                  exploradora.melhor_posicao = melhor_posicao_inter.copy()
                  exploradora.melhor_fitness = melhor_fitness_inter

              # desfaz a exploração da fonte de comida se necessário
              if exploradora.ciclos_sem_melhora >= self.limite:
                  for i in range (self.dimensao):
                    exploradora.posicao[i]= random.uniform(self.bound[0], self.bound[1])
                  exploradora.melhor_posicao = exploradora.posicao.copy()
                  exploradora.fitness = funcao_fitness(exploradora.posicao,self.funcao)
                  exploradora.melhor_fitness = exploradora.fitness
                  exploradora.ciclos_sem_melhora = 0
                  trabalhadoras_temp = exploradora.trabalhadoras.copy()
                  exploradora.trabalhadoras = []
                  self.alocar_abelhas(trabalhadoras_temp)                  
              # atuaiza melhor posicao global
              if melhor_posicao is None or exploradora.melhor_fitness < melhor_fitness:
                  melhor_posicao = exploradora.melhor_posicao.copy()
                  melhor_fitness = exploradora.melhor_fitness

          #self.exibir_colonia()
          track_fitness.append(melhor_fitness)
        return melhor_fitness,track_fitness


