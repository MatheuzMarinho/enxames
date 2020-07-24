import pso
import os
import pandas as pd
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
    bests = [bests_clerc,bests_const,bests_linear]
    df = pd.DataFrame()
    import matplotlib.pyplot as plt
    for i in range(len(INERCIAS)):
        bests[i].sort()
        df[INERCIAS[i]] = bests[i]
        fig, ax = plt.subplots()
        ax.set_title(f'PSO {TOPOLOGIA}-{funcao}-{INERCIAS[i]}')
        ax.boxplot(bests[i])
        ax.yaxis.get_major_formatter().set_scientific(False)
        plt.tight_layout()
        path = os.path.dirname(os.path.abspath(__file__))
        path += "/boxplot"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+'/'+dir_name
        if not os.path.exists(path):
            os.mkdir(path)
        name_figure = f"PSO {TOPOLOGIA}-{funcao}-{INERCIAS[i]}"
        plt.savefig(path+'/'+name_figure+'.png')
        #plt.show()
    name_figure = f"PSO {TOPOLOGIA}-{funcao}.csv"
    df.to_csv(path+'/'+name_figure,index=False)


def unica_rodada():
    p = pso.PSO(c1,c2,velocidade_max,dimensao,numero_particulas,qtd_iteracoes,TOPOLOGIA,INERCIAS,funcao)
    p.optimize(w,dir_name,True)

w = 0.8
c1 = 2.05
c2 = 2.05
velocidade_max=[-6,6]
dimensao = 30
numero_particulas = 30
qtd_iteracoes = 10000
swarm = []
TOPOLOGIA = 'GLOBAL'  #LOCAL #GLOBAL
INERCIAS = ['CLERC', 'CONSTANTE', 'LINEAR']
dir_name = 'atividade_2'
funcao = 'rosenbrock' #rastrigin ackley rosenbrock esfera

#unica_rodada()
multiplas_rodadas()