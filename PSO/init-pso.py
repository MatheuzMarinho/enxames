import pso
import os
def multiplas_rodadas():
    results = []
    for i in range(1,31):
        print('RODADA:',i)
        p = pso.PSO(c1,c2,velocidade_max,dimensao,numero_particulas,qtd_iteracoes,TOPOLOGIA,INERCIAS,funcao)
        results.append(p.optimize(w))
    bests_clerc = []
    bests_const = []
    bests_linear = []
    for result in results:
        bests_clerc.append(result[0][-1])
        bests_const.append(result[1][-1])
        bests_linear.append(result[2][-1])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title(f'PSO {TOPOLOGIA}-{funcao}')
    ax.boxplot([bests_clerc,bests_const,bests_linear])
    ax.set_xticklabels(INERCIAS)
    path = "boxplot"
    if not os.path.exists(path):
        os.mkdir(path)
    path = path+'/'+dir_name
    if not os.path.exists(path):
        os.mkdir(path)
    name_figure = f"PSO {TOPOLOGIA}-{funcao}"
    plt.savefig(path+'/'+name_figure+'.png')
    #plt.show()

def unica_rodada():
    p = pso.PSO(c1,c2,velocidade_max,dimensao,numero_particulas,qtd_iteracoes,TOPOLOGIA,INERCIAS,funcao)
    p.optimize(w,dir_name,True)

w = 0.8
c1 = 2.05
c2 = 2.05
velocidade_max=[-6,6]
dimensao = 30
numero_particulas = 30
qtd_iteracoes = 1000
swarm = []
TOPOLOGIA = 'GLOBAL'  #LOCAL #GLOBAL
INERCIAS = ['CLERC', 'CONSTANTE', 'LINEAR']
dir_name = 'atividade_2'
funcao = 'rosenbrock' #rastrigin ackley rosenbrock esfera

#unica_rodada()
multiplas_rodadas()