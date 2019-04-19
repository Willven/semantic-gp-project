from gsgp import GSGP

class MPGSGP:
    def __init__(self, mutation_rate, input_vars, max_depth, operators, \
                 pop_size, trunc_ratio=0.5, max_fitness=None):
        self.mutation_rate = mutation_rate
        self.input_vars = input_vars
        self.max_depth = max_depth
        self.max_fitness = max_fitness
        self.operators = operators
        self.pop_size = pop_size
        self.trunc_ratio = trunc_ratio
        
    def run(self, inputs, targets, generations):
        self.inputs = inputs
        self.targets = targets
        
        self.tree = [GSGP(mutation_rate=self.mutation_rate, input_vars=self.input_vars, \
                                targets=self.targets[:, i], inputs= self.inputs, \
                                max_depth=self.max_depth, operators=self.operators, pop_size=self.pop_size,\
                                trunc_ratio=self.trunc_ratio,
                                max_fitness=self.max_fitness) for i in range(self.targets.shape[1])]
        
        return self._evolve(generations)
            
    def _evolve(self, generations):
        bests = []
        avgs = []
        for sim in self.tree:
            _, acg, best = sim.evolve(generations)
            bests.append(best)
            avgs.append(avg)
        return bests, avgs
            
    def predict(self, Xte):
        return [[sim.best_fitness_func(*X) for sim in self.tree] for X in Xte]