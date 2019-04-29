from graph import Graph
import matplotlib.pyplot as plt
import pickle


files_pickle = ['./graphs/20.pckl','./graphs/40.pckl','./graphs/80.pckl','./graphs/160.pckl','./graphs/gradient.pckl']
graphs = []
for file_pickle in files_pickle:
    file_pickle_opened = open(file_pickle, 'rb') 
    graphs.append(pickle.load(file_pickle_opened))
    file_pickle_opened.close();
 
plt.plot(graphs[0].tempo,graphs[0].performance, '-',graphs[1].tempo,graphs[1].performance, '-',graphs[2].tempo,graphs[2].performance, '-',graphs[3].tempo,graphs[3].performance, '-',graphs[4].tempo,graphs[4].performance, '-')
plt.show()


