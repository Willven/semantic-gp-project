from gsgp import GSGP

class MPGSGP:
    """
    Class for performing MP-GSGP.
    """
    def __init__(self, mutation_step, input_vars, max_depth, operators, \
            pop_size, trunc_ratio=0.5):
        """
        Method for constructing a MPGSGP instance.

        Parameters:
        
        mutation_step (float) : The mutation step of the algorithm. 
            A smaller value means more exploitation, whilst a larger one 
            signifies more exploration.
        input_vars (list) : A list of strings, each corresponding to a 
            single value found within the data.
        max_depth (int) : A value specifiying the maximum depth of any of
                the trees generated.
        operators (iterable) : An iterable containing strings corresponding
            to the function names of all suitable operators.
        pop_size (int) : The size of the population to be maintained.
        trunc_ratio (float) : The percentage of the population to be 
            truncated and replaced each generation. Default 0.5.    
        """
        self.mutation_step = mutation_step
        self.input_vars = input_vars
        self.max_depth = max_depth
        self.operators = operators
        self.pop_size = pop_size
        self.trunc_ratio = trunc_ratio
        
    def run(self, inputs, targets, generations=10):
        """
        Method for running the MP-GSGP algorithm.

        Parameters:
        inputs (iterable) : An iterable (typically a NumPy array) 
            containing each item of training data, over all input.
        targets (iterable) : An iterable (typically a NumPy array) 
            containing each of the desired targets for the training data.
        generations (int) : The number of generations the algorithm should
            be run for. Default 10.
        """
        self.inputs = inputs
        self.targets = targets
        
        self.tree = [GSGP(mutation_step=self.mutation_step, input_vars=self.input_vars, \
                                targets=self.targets[:, i], inputs= self.inputs, \
                                max_depth=self.max_depth, operators=self.operators, pop_size=self.pop_size,\
                                trunc_ratio=self.trunc_ratio) for i in range(self.targets.shape[1])]
        
        return self._evolve(generations)
            
    def _evolve(self, generations):
        """
        Method used for running the algorithm over a number of generations.

        Parameters:
        generations (int) : The number of generations to evolve the algorithm over.

        Returns:
        tuple: (best_fitnesses, avg_fitnesses) containing both the best fitness achieved at 
            every generation as well the average fitness achived at each generation. 
        """
        bests = []
        avgs = []
        for sim in self.tree:
            _, avg, best = sim.evolve(generations)
            bests.append(best)
            avgs.append(avg)
        return bests, avgs
            
    def predict(self, Xte):
        """
        Method used to predict the output from the best functions obtained.

        Parameters:

        Xte (Iterable) : An iterable (typically a NumPy array) containing the data inputs
            from which to obtain a prediction.

        Returns:
        List: A list of len(Xte) containing the predictions for the input parameters over
            all the different targets previously trained on the algorithm.
        """
        return [[sim.best_fitness_func(*X) for sim in self.tree] for X in Xte]


if __name__ == '__main__':
    import pandas as pd
    from numpy import loadtxt, count_nonzero, array
    from numpy.random import permutation
    from util import *
    from util import default_operators as operators

    X = pd.read_csv('./data/reg_inputs.csv')
    y = loadtxt('./data/reg_targets.csv', delimiter=',', dtype=float).astype('object')

    X = X.values[:, 1:].astype('object')
    print(X.shape)

    input_vars = [f'x{i}' for i in range(X.shape[1])]

    # Training set
    I = permutation(len(y))
    Xtr = X[I[:250]]
    ttr = y[I[:250]]

    Xte = X[I[-500:]]
    tte = y[I[-500:]]

    mpgsgp = MPGSGP(mutation_step=0.2, input_vars=input_vars, max_depth=3, operators=operators,
                    pop_size=500, trunc_ratio=0.2)
    mpgsgp.run(Xtr, ttr, 10)

    prediction = array(mpgsgp.predict(Xte)).astype(int)

    print(f'Accuracy: {(1 - count_nonzero(array(tte) - prediction)/tte.size)*100:.2f}%.')