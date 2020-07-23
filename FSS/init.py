import fss

population_size =30
dimension = 30
iteration = 10000
step_individual_init = 0.001
step_individual_final = 0.00001
step_volitive_init = 0.01
step_volitive_final = 0.0001
w_scale = iteration / 2.0
min_w = 1
function = 'rosenbrock' #rastrigin ackley rosenbrock esfera
dir_name = 'atividade-2'

results = []
for i in range(1,31):
    print('iteration:',i)
    f = fss.School(population_size,dimension,iteration,step_individual_init,step_individual_final,step_volitive_init,step_volitive_final,min_w,function)
    best = f.otimization()
    results.append(best)

import matplotlib.pyplot as plt
import os
fig, ax = plt.subplots()
ax.set_title(f'FSS - {function}')
ax.boxplot(results)
path = os.path.dirname(os.path.abspath(__file__))
path += "/boxplot"
if not os.path.exists(path):
    os.mkdir(path)
path = path+'/'+dir_name
if not os.path.exists(path):
    os.mkdir(path)
name_figure = f"FSS -{function}"
plt.savefig(path+'/'+name_figure+'.png')
#plt.show()