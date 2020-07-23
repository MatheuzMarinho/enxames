import random
import numpy as np
import matplotlib.pyplot as plt
import math




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
  for i in range (dimensao):
      total += lista_solucao[i]**2
  return total

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

class Particula:
  def __init__(self):
        self.lista_posicao = []
        self.lista_velocidade = []
        self.fitness = np.inf
        self.fitness_pbest = np.inf
        self.lista_posicao_pbest = []

  def calculo_fitness(self,funcao):
    self.fitness = funcao_fitness(self.lista_posicao,funcao)
    if (self.fitness < self.fitness_pbest):
      self.fitness_pbest= self.fitness
      self.lista_posicao_pbest= list(self.lista_posicao)

  def atualizacao_velocidade(self, lista_best,dimensao,c1,c2,INERCIA,bound,w):
    for i in range(dimensao):
      e1 = random.random()
      e2 = random.random()
      velocidade_cognitiva = c1*e1* (self.lista_posicao_pbest[i] - self.lista_posicao[i])
      velocidade_social = c2*e2* (lista_best[i] - self.lista_posicao[i])
      if(INERCIA == 'CLERC'):
          v = w * (self.lista_velocidade[i] + velocidade_cognitiva + velocidade_social)
      else:
          v = w * self.lista_velocidade[i] + velocidade_cognitiva + velocidade_social
      if v>bound[1]:
        v = bound[1]
      elif v<bound[0]:
        v = bound[0]
      self.lista_velocidade[i] = v

  def atualiza_posicao(self, bound,dimensao):
    for i in range(dimensao):
        novo_valor = self.lista_posicao[i] + self.lista_velocidade[i]
        if novo_valor > bound[1]:
            novo_valor =  bound[1]
        if novo_valor < bound[0]:
            novo_valor = bound[0]
        self.lista_posicao[i] = novo_valor

class PSO:
  def __init__(self,c1,c2,velocidade_max,dimensao,numero_particulas,qtd_iteracoes,TOPOLOGIA,INERCIAS,funcao):
    self.w = np.nan
    self.c1 = c1
    self.c2 = c2
    self.bound= function_bounds(funcao)
    self.velocidade_max= velocidade_max
    self.dimensao = dimensao
    self.numero_particulas = numero_particulas
    self.qtd_iteracoes = qtd_iteracoes
    self.TOPOLOGIA = TOPOLOGIA
    self.INERCIAS = INERCIAS
    self.swarm = []
    self.funcao = funcao

  def inicializacao(self):
    for i in range(self.numero_particulas):
      p = Particula()
      for i in range (self.dimensao):
        p.lista_posicao.append(random.uniform(self.bound[0], self.bound[1]))
        p.lista_velocidade.append(random.uniform(self.velocidade_max[0], self.velocidade_max[1]))
      self.swarm.append(p)

  def melhor_local(self, particula_idx):
    vizinho_a = particula_idx-1
    vizinho_b = particula_idx+1
    if particula_idx == len(self.swarm) - 1 :
      vizinho_a = 0
      vizinho_b = particula_idx - 1
    
    if self.swarm[vizinho_a].fitness_pbest < self.swarm[vizinho_b].fitness_pbest:
      return vizinho_a
    else:
      return vizinho_b

  def decaimento_linear(self,iteracao_atual):
      w_max = 0.9
      w_min = 0.4
      return (w_max- w_min)*((self.qtd_iteracoes - iteracao_atual)/self.qtd_iteracoes)+w_min

  def curva_convergencia_ind(self,lista_valores_gbest,fitness_gbest,INERCIA,dir_name):
      fig = plt.figure()
      plt.plot(lista_valores_gbest)
      fit = round(fitness_gbest,2)
      plt.title(f"Curva de Convergencia PSO {self.TOPOLOGIA}-{INERCIA} Fitness: {fit}")
      plt.xlabel("Iterações")
      plt.ylabel("Melhor Fitness")
      #plt.tight_layout()
      #plt.show()
      print('Valor gbest:',fitness_gbest)
      import os
      # define the name of the directory to be created
      path = "resultados"
      if not os.path.exists(path):
          os.mkdir(path)
      path = path+'/'+dir_name
      if not os.path.exists(path):
        os.mkdir(path)
      name_figure = f"PSO {self.TOPOLOGIA}-{INERCIA}"
      plt.savefig(path+'/'+name_figure+'.png')
      plt.clf() 

  def curva_convergencia_geral(self,list_resultados_inercias,dir_name):
    for i, INERCIA in enumerate(self.INERCIAS):
      plt.plot(list_resultados_inercias[i])
    plt.title(f"Curva de Convergencia PSO {self.TOPOLOGIA}")
    plt.legend(self.INERCIAS)
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Fitness")
    name_figure = f"PSO {self.TOPOLOGIA}"
    import os
    # define the name of the directory to be created
    path = "resultados"
    if not os.path.exists(path):
        os.mkdir(path)
    path = path+'/'+dir_name
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path+'/'+name_figure+'.png')
    plt.clf() 



  def optimize(self,w,dir_name='',save_graph=False):
    list_resultados_inercias = []
    for INERCIA in self.INERCIAS:
      self.swarm = []
      self.inicializacao()
      self.w = w
      fitness_gbest = float('inf')
      lista_posicao_gbest = []
      lista_valores_gbest = []
      for i in range(self.qtd_iteracoes):
        for j in range(self.numero_particulas):
          self.swarm[j].calculo_fitness(self.funcao)
          if  self.swarm[j].fitness < fitness_gbest:
            fitness_gbest = self.swarm[j].fitness
            lista_posicao_gbest = list (self.swarm[j].lista_posicao)
        for j in range(self.numero_particulas):
          if(INERCIA=='LINEAR'):
              self.w = self.decaimento_linear(i)
          if(self.TOPOLOGIA == 'GLOBAL'):
            self.swarm[j].atualizacao_velocidade(lista_posicao_gbest,self.dimensao,self.c1,self.c2,INERCIA,self.bound,self.w)
          elif(self.TOPOLOGIA=='LOCAL'):
            lbest = self.melhor_local(j)
            self.swarm[j].atualizacao_velocidade(self.swarm[lbest].lista_posicao_pbest,self.dimensao,self.c1,self.c2,INERCIA,self.bound,self.w)
          self.swarm[j].atualiza_posicao(self.bound,self.dimensao)
        
        lista_valores_gbest.append(fitness_gbest)
      
      list_resultados_inercias.append(lista_valores_gbest)
      if save_graph:
        self.curva_convergencia_ind(lista_valores_gbest,fitness_gbest,INERCIA,dir_name)
    if save_graph:
      self.curva_convergencia_geral(list_resultados_inercias,dir_name)
    return list_resultados_inercias


  

