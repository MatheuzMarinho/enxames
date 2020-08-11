import numpy as np
from scipy import spatial
import random
from numpy.random import choice
import copy
import os




class Ant(object):
    def __init__(self, graph,num_points):
        self.graph = graph
        self.total_cost = 0.0
        self.solution = [] 
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(num_points)]  # nodes which are allowed for the next selection
        self.heuristic_information  = np.array([[0 if i == j else 1 / graph[i][j] for j in range(num_points)] for i in
                    range(num_points)])  # heuristic information
        start = random.randint(0, num_points - 1)  # start from any node
        self.solution.append(start)
        self.current = start
        self.allowed.remove(start)

    def select_next(self,alpha,beta,pheromone):
        denominator = 0
        for i in self.allowed:
            denominator += pheromone[self.current][i] ** alpha * self.heuristic_information[self.current][i] ** beta

        probabilities = []
        for i in self.allowed:
            probabilities.append(pheromone[self.current][i] ** alpha * self.heuristic_information[self.current][i] ** beta / denominator)
        
        selected = choice(self.allowed, 1, p=probabilities)[0]
        self.allowed.remove(selected)
        self.solution.append(selected)
        self.total_cost += self.graph[self.current][selected]
        self.current = selected

    def update_pheromone_delta(self,num_points):
        #gera a matriz com 0's
        self.pheromone_delta = np.array([[0.0 for j in range(num_points)] for i in range(num_points)])
        for idx in range(1, len(self.solution)):
            i = self.solution[idx-1]
            j = self.solution[idx]
            self.pheromone_delta[i][j] = 1.0 / self.total_cost


def update_pheromone(pheromone,ants,p):
        for i, row in enumerate(pheromone):
            for j, col in enumerate(row):
                pheromone[i][j] *= (1-p)
                for ant in ants:
                    pheromone[i][j] += ant.pheromone_delta[i][j]

def read_txt(problem_name):
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    input = np.loadtxt(path+"/"+problem_name+".txt", dtype='i', delimiter=' ')
    return len(input), input


p = 0.2 #evaporate ratio
alfa= 0.5 # importance of the pheromone
beta = 0.8  #  importance of heuristic information
iterations = 500
ant_count = 30

# num_points = 3
# points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points

problem_name = 'att48_xy'
num_points,points_coordinate = read_txt(problem_name)
#calcula as distancias
graph = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
#inicializa a matriz de ferominios -> Mesmo valor pra todos


MIN_MAX = True

bests_results = []

for _round in range(30):
    pheromone = np.array([[1 / (num_points * num_points) for j in range(num_points)] for i in range(num_points)])
    best_cost = np.inf
    best_solution = []
    best_ant = None
    for iteraction in range(iterations):
        print('EXECUTION',_round,'ITERACTION:',iteraction)
        #Criação das formigas -> Não sei se fica aqui realmente!
        ants = [Ant(graph,num_points) for i in range(ant_count)]
        for ant in ants:
            for i in range(num_points - 1):
                ant.select_next(alfa,beta,pheromone)

            ant.update_pheromone_delta(num_points)
            if ant.total_cost < best_cost:
                best_cost = ant.total_cost
                best_solution = list(ant.solution)
                best_ant = copy.deepcopy(ant)

        if MIN_MAX:
            update_pheromone(pheromone,[best_ant],p)
        else:
            update_pheromone(pheromone,ants,p)

    bests_results.append(best_cost)

bests_results.sort()
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
if MIN_MAX:
    ax.set_title(f'ACO MIN MAX- {round(bests_results[0])}')
else:
    ax.set_title(f'ACO - {round(bests_results[0])}')
ax.boxplot(bests_results)
plt.tight_layout()
path = os.path.dirname(os.path.abspath(__file__))
path += "/boxplot"
if not os.path.exists(path):
    os.mkdir(path)
path = path+'/'+problem_name
if not os.path.exists(path):
    os.mkdir(path)
if MIN_MAX:
    name_figure = f'ACO MIN MAX - {problem_name}'
else:
    name_figure = f'ACO - {problem_name}'

plt.savefig(path+'/'+name_figure+'.png')