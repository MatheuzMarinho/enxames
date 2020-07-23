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
  for i in range (len(lista_solucao)):
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



class Fish:
    def __init__(self,dim):
        self.weight = np.inf
        self.position = []
        self.fitness = np.inf
        #Diferenca entre o fitss antigo e novo
        self.delta_cost = np.inf
        #Diferenca entre a posicao antiga e nova
        self.delta_pos = np.zeros((dim,), dtype=np.float)
        self.weight = np.nan


    def individual_moviment(self,current_step,dim,bound,function):
        n_moviment = np.zeros((dim,), dtype=np.float)
        for i in range(dim):
            n_moviment[i] = self.position[i] + random.uniform(-1, 1) * current_step
            if n_moviment[i] > bound[1]:
                n_moviment[i] = bound[1]
            if n_moviment[i] < bound[0]:
                n_moviment[i] = bound[0]
        n_fitness = funcao_fitness(n_moviment,function)
        if n_fitness < self.fitness:
            self.delta_cost = abs(n_fitness - self.fitness)
            self.fitness = n_fitness
            for idx in range(dim):
                self.delta_pos[idx] = n_moviment[idx] - self.position[idx]
            self.position = list(n_moviment)
        else:
            self.delta_pos = np.zeros((dim,), dtype=np.float)
            self.delta_cost = 0
    
    def feeding(self, max_weight):
        self.weight = self.weight + self.delta_cost/max_weight
    


class School:
    def __init__(self,population_size,dimension,iteration,step_individual_init,step_individual_final,step_volitive_init,step_volitive_final,min_w,function):
        self.population_size = population_size
        self.dimension = dimension
        self.iteration = iteration
        self.function = function
        self.bound = function_bounds(function)
        self.step_individual_init = step_individual_init
        self.step_individual_final = step_individual_final
        self.step_volitive_init = step_volitive_init
        self.step_volitive_final = step_volitive_final
        self.min_w = min_w
        self.current_step = step_individual_init * (self.bound[1] - self.bound[0])
        self.curr_step_volitive = step_volitive_init * (self.bound[1] - self.bound[0])
        self.w_scale = iteration / 2.0
        self.school = []
        self.prev_weight_school = 0
        self.curr_weight_school = 0
        self.barycenter = []
        self.list_best_fitness = [float('inf')]
    
    def inicialization(self):
        for i in range(self.population_size):
            f = Fish(self.dimension)
            for i in range (self.dimension):
                f.position.append(random.uniform(self.bound[0], self.bound[1]))
            f.fitness = funcao_fitness(f.position,self.function)
            f.weight = self.w_scale/2.0
            self.school.append(f)

    def calculate_barycenter(self):
        barycenter = np.zeros((self.dimension,), dtype=np.float)
        density = 0.0
        for fish in self.school:
            density += fish.weight
            for dim in range(self.dimension):
                barycenter[dim] += (fish.position[dim] * fish.weight)
        for dim in range(self.dimension):
            barycenter[dim] = barycenter[dim] / density

        self.barycenter = list(barycenter)

    def total_school_weight(self):
            self.prev_weight_school = self.curr_weight_school
            self.curr_weight_school = 0.0
            for fish in self.school:
                self.curr_weight_school += fish.weight
            

    def update_best_value(self):
        best_fitness = self.list_best_fitness[-1]
        for fish in self.school:
            if best_fitness > fish.fitness:
                best_fitness = fish.fitness
        self.list_best_fitness.append(best_fitness)

    def calculate_step(self):
        self.current_step = self.current_step - ((self.step_individual_init-self.step_individual_final)/self.iteration)
        self.curr_step_volitive = self.curr_step_volitive - ((self.step_volitive_init - self.step_volitive_final)/self.iteration)

    def max_delta_cost(self):
        max_value = 0
        for fish in self.school:
            if max_value < fish.delta_cost:
                max_value = fish.delta_cost
        return max_value

    def feeding(self,max_delta_value):
        for fish in self.school:
            if max_delta_value > 0:
                fish.weight = fish.weight + (fish.delta_cost / max_delta_value)
                if fish.weight > self.w_scale:
                    fish.weight = self.w_scale
                elif fish.weight < self.min_w:
                    fish.weight = self.min_w

    def collective_instinctive_movement(self):
        cost_eval_enhanced = np.zeros((self.dimension,), dtype=np.float)
        density = 0.0
        for fish in self.school:
            density += fish.delta_cost
            for dim in range(self.dimension):
                cost_eval_enhanced[dim] += (fish.delta_pos[dim] * fish.delta_cost)
        for dim in range(self.dimension):
            if density != 0:
                cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density
        for fish in self.school:
            new_pos = np.zeros((self.dimension,), dtype=np.float)
            for dim in range(self.dimension):
                new_pos[dim] = fish.position[dim] + cost_eval_enhanced[dim]
                if new_pos[dim] < self.bound[0]:
                    new_pos[dim] = self.bound[0]
                elif new_pos[dim] > self.bound[1]:
                    new_pos[dim] = self.bound[1]
            fish.position = new_pos

    def collective_volitive_movement(self):
        for fish in self.school:
            new_pos = np.zeros((self.dimension,), dtype=np.float)
            for dim in range(self.dimension):
                #Contração
                if self.curr_weight_school > self.prev_weight_school:
                    new_pos[dim] = fish.position[dim] - ((fish.position[dim] - self.barycenter[dim]) * self.curr_step_volitive *
                                                    np.random.uniform(0, 1))
                #Dilatação                                               
                else:
                    new_pos[dim] = fish.position[dim] + ((fish.position[dim] - self.barycenter[dim]) * self.curr_step_volitive *
                                                    np.random.uniform(0, 1))                                                
                if new_pos[dim] < self.bound[0]:
                    new_pos[dim] = self.bound[0]
                elif new_pos[dim] > self.bound[1]:
                    new_pos[dim] = self.bound[1]   
            fitness = funcao_fitness(new_pos,self.function)
            fish.fitness = fitness
            fish.position = new_pos

    def exibe_school(self):
        for i in range(self.population_size):
            f = self.school[i]
            x, y = zip(f.position)
            plt.plot(x,y, marker='o')
        #ponto ideal
        plt.plot(0,0, marker='*')
        plt.axis([-100, 100, -100, 100])
        plt.show()

    def otimization(self):
        self.inicialization()
        #self.exibe_school()
        self.total_school_weight()
        self.update_best_value()
        for i in range(self.iteration):
            for fish in self.school:
                fish.individual_moviment(self.current_step,self.dimension,self.bound,self.function)
            self.feeding(self.max_delta_cost())
            self.collective_instinctive_movement()
            self.calculate_barycenter()
            self.total_school_weight()
            self.collective_volitive_movement()
            #self.exibe_school() 
            self.calculate_step()    
            self.update_best_value()

        print(self.list_best_fitness[-1])
        return self.list_best_fitness[-1]
