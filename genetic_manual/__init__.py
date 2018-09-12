import numpy as np

def create_population(populationSize):

    population = []
    for i in range(populationSize):
        w_1 = np.random.rand(5, 4).astype('f');
        w_2 = np.random.rand(4, 100).astype('f');
        w_3 = np.random.rand(100, 500).astype('f');
        population.append([w_1,w_2,w_3]);
    return population

def Crossover():


def Mutation():



def Selection():



def Fitness():




if __name__ == "__main__":
    genetic_pool_settings = {
        'populationSize': 30,
        'tournamentSize': 4,
        'memberDimensions': [4, 3, 2, 3, 4],
        'mutationRate': 0.05,
        'averagesCount': 1,
        'maxEpochs': 10,
 #       'ins': input_data,
 #       'outs': output_data
    };





