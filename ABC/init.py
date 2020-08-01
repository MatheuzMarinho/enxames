import abc_swarm
import matplotlib.pyplot as plt
import numpy as np
import os
def analysis():
    for movimentacao in movimentacoes:
        print('MOVIMENTACAO',movimentacao)
        results = []
        tracks=[]
        for i in range(1,31):
            print('RODADA:',i)
            a = abc_swarm.ABC(n,limite,cmax,dimensao,funcao,movimentacao)
            fitness,track = a.optimize()
            results.append(fitness)
            tracks.append(track)
        fig, ax = plt.subplots()
        ax.set_title(f'ABC {movimentacao}-{funcao}\n Média: {round(np.mean(results),4)} (+/- {round(np.std(results),4)})')
        ax.boxplot(results)
        ax.yaxis.get_major_formatter().set_scientific(False)
        plt.tight_layout()
        path_root = os.path.dirname(os.path.abspath(__file__))
        path = path_root + "/boxplot"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+'/'+dir_name
        if not os.path.exists(path):
            os.mkdir(path)
        name_figure = f'ABC {movimentacao}-{funcao}'
        plt.savefig(path+'/'+name_figure+'.png')

        plt.subplots()
        plt.plot(np.mean(tracks, axis=0))
        plt.title(f"Curva de Convergencia ABC {funcao}\n{movimentacao}")
        plt.xlabel("Iterações")
        plt.ylabel("Melhor Fitness")
        name_figure = f'ABC {movimentacao}-{funcao}'
        path = path_root + "/convergencia"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+'/'+dir_name
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path+'/'+name_figure+'.png')
        

def single():
    a = abc_swarm.ABC(n,limite,cmax,dimensao,funcao,movimentacao)
    best, track = a.optimize()
    fig, ax = plt.subplots()
    plt.plot(track)
    plt.title(f"Curva de Convergencia ABC {movimentacao}")
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Fitness")
    plt.show()
    print(best)

n = 20
dimensao = 5
limite = dimensao*(n/2)
cmax = 1
funcao = 'esfera'  #rastrigin ackley rosenbrock esfera
movimentacoes = ['tradicional','roleta','colmeia']
movimentacao = 'tradicional'
dir_name = 'atividade_3'

#analysis()
single()


