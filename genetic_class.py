from neural_network import Neural_network
from create_population import create_population
from choose_best import choose_best_tensor
import time

class Population:
    
    def __init__(self,populationSize,layers,mutationRate):
        self.populationSize = populationSize
        self.layers = layers
        self.mutationRate = mutationRate
        self.neural_networks = Neural_network(create_population(layers,populationSize),layers, './log/' )
        self.current_epoch = 0

    def run_epoch(self):
        
        
        print("neural networks run:")
        start = time.time()
        self.neural_networks.run()
        print("neural network : ", time.time() - start)
        choose_best_tensor(self.neural_networks.neural_networks, self.neural_networks.accuracies)

        self.current_epoch += 1

    
    
