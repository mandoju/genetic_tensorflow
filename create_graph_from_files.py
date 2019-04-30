from graph import Graph
import matplotlib.pyplot as plt
import pickle


#files_pickle = ['./graphs/20.pckl','./graphs/40.pckl','./graphs/80.pckl','./graphs/160.pckl','./graphs/gradient_he.pckl']
#files_pickle = ['./graphs/20_he.pckl','./graphs/40_he.pckl','./graphs/80_he.pckl','./graphs/160_he.pckl','./graphs/gradient_he.pckl']
files_pickle = ['./graphs/20_he_menor.pckl','./graphs/40_he_menor.pckl','./graphs/80_he_menor.pckl','./graphs/160_he_menor.pckl','./graphs/gradient.pckl']

graphs = []

for file_pickle in files_pickle:
    file_pickle_opened = open(file_pickle, 'rb') 
    graphs.append(pickle.load(file_pickle_opened) )
    file_pickle_opened.close();
 
for graph in graphs:
    if(graph.performance[0] < 0):
        graph.performance = [x * -1 for x in graph.performance]

print(graphs[0].performance)
mutation_20 = plt.plot(graphs[0].tempo,graphs[0].performance , '-', label="Mutation_20")
mutation_40 = plt.plot(graphs[1].tempo,graphs[1].performance , '-', label="Mutation_40")
mutation_80 = plt.plot(graphs[2].tempo,graphs[2].performance , '-', label="Mutation_80")
mutation_160 = plt.plot(graphs[3].tempo,graphs[3].performance , '-', label="Mutation_160")
gradient = plt.plot(graphs[4].tempo,graphs[4].performance , '-', label="Gradient")
#plt.legend([mutation_20,mutation_40,mutation_80,mutation_160,gradient],['mutação 20','mutação 40','mutação 80','mutação 160'])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.grid(True);
plt.show()



