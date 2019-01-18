from genetic import ENN


if __name__ == "__main__":
    genetic_pool_settings = {
        'populationSize': 30,
        'tournamentSize': 4,
        'memberDimensions': [4, 3, 2, 3, 4],
        'mutationRate': 0.05,
        'averagesCount': 1,
        'maxEpochs': 10
    },

    geneticSettings = {
        'populationSize': 10,
        'epochs': 10,
        'layers': [785,10,9],
        'mutationRate': 0.20
    }

    ENN(geneticSettings)
