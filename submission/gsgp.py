import random
from tqdm import tqdm
from util import *
from statistics import mean


def get_loop(generations):
    """
    Function used to obtain the appropriate range, using TQDM if installed to give progress updates,
    otherwise remains with a standard list.

    Parameters:
    generations (int) : The number of generations, and thus the range to generate.

    Returns:
    A range object.
    """
    try:
        import tqdm
        return tqdm.trange(generations)
    except ModuleNotFoundError:
        return range(generations)


def memoize(f):
    """
    Function used to cache a function to ensure the system can cope with the exponential function size.

    Parameters:
    f (function) : The decorated function to be cached.
    """
    f.cache = {}
    def decorated_function(*args):
        if args in f.cache:
            return f.cache[args]
        else:
            f.cache[args] = f(*args)
            return f.cache[args]
    return decorated_function

class GSGP:
    """
    Class used to handle a singular GSGP problem.
    """
    def __init__(self, mutation_step, input_vars, max_depth, inputs, targets, operators, \
                 pop_size, trunc_ratio=0.5, tournament_size=None, penalty_min=0., penalty_max=7.):
                 """
                 Function constructing a GSGP instance. 

                 Parameters:
                 
                 mutation_step (float) : The mutation step of the algorithm. 
                    A smaller value means more exploitation, whilst a larger one 
                    signifies more exploration.
                input_vars (list) : A list of strings, each corresponding to a 
                    single value found within the data.
                max_depth (int) : A value specifiying the maximum depth of any of
                     the trees generated.
                inputs (iterable) : An iterable (typically a NumPy array) 
                    containing each item of training data, over all input.
                targets (iterable) : An iterable (typically a NumPy array) 
                    containing each of the desired targets for the training data.
                operators (iterable) : An iterable containing strings corresponding
                    to the function names of all suitable operators.
                pop_size (int) : The size of the population to be maintained.
                trunc_ratio (float) : The percentage of the population to be 
                    truncated and replaced each generation. Default 0.5.
                tournament_size (int or None) If None, the selection mechanism will
                    be semi-stochastic. However, should an integer be 
                    provided then this value will given as the tournament size for
                    selection.
                penalty_min (float) : The lower threshold at which a penalty should 
                    be applied to the fitness.
                penalty_max (float) : The upper threshold at which a penalty should 
                    be applied to the fitness.
                 """
                 self.mutation_step = mutation_step
                 self.vars = input_vars
                 self.max_depth = max_depth
                 self.inputs = inputs
                 self.targets = targets
                 self.operators = operators
                 self.pop_size = pop_size
                 self.trunc_ratio = trunc_ratio
                 self.tournament_size = tournament_size
                 self.penalty_min = penalty_min
                 self.penalty_max = penalty_max
    
    def random_expression(self, depth):
        """
        Method for generating a random expression. 

        Parameters:

        depth (int) : The maximum depth of the expression to generate.

        Returns: 

        String: A random expression to be evaluated.

        """
        if depth == 1 or random.random() < 1/(2**depth-1):
            return random.choice((*self.vars, *[str(random.random()) for i in range(len(self.vars))]))
        else:
            return '(' + random.choice(self.operators) + '(' + \
                self.random_expression(depth - 1) + ',' + \
                self.random_expression(depth - 1) + '))'
        
    def random_function(self):
        """
        Method for generating random functions.

        Returns:

        Lambda: The random function generated. 
        """
        re = self.random_expression(self.max_depth)

        rf = eval('lambda ' + ', '.join(self.vars) + ': ' + re)
        rf = memoize(rf)
        rf.geno = lambda: re
        return rf
    
    def crossover(self, p1, p2):
        """
        Method for performing geometric semantic crossover on two parent individuals.

        Parameters:

        p1 (Lambda) : The first parent function.
        p2 (Lambda) : The second parent function.

        Returns:
        Lambda: The offspring parent function.
        """
        k = random.random()
        offspring = lambda *x: k * p1(*x) + (1-k) * p2(*x)
        offspring = memoize(offspring) # add cache
        offspring.geno = lambda: '(('+ str(k) + '*' + p1.geno() + ') + ((1-' + str(k) + ')' +  p2.geno() + '))'
        return offspring
    
    def mutation(self, p):
        """
        Method for performing geometric semantic mutation on a parent individual.

        Parameters: 

        p (Lambda) : The parent function.

        Returns:
        Lambda: The offspring.
        """
        rm = self.random_function()
        rn = self.random_function()
        
        offspring = lambda *x: p(*x) - (self.mutation_step * (rn(*x) - rm(*x)))
        offspring = memoize(offspring) # add cache

        offspring.geno = lambda: '((' + p.geno() - '(' + str(self.mutation_step) + '*(' + rm.geno() + '*' + rn.geno() + '))))'

        return offspring
    
    def _grade_pop(self):
        """
        Method for grading the population based upon the fitness.

        Returns:

        List: A list of tuples (fitness, indvidual) for each individual in the population.
        """
        return [ (self.fitness(ind), ind) for ind in self.pop ]
    
    def _sort_pop(self, graded_pop):
        """
        Method for sorting the population based upon the fitness.

        Returns:

        List: A list of tuples (fitness, indvidual) for each individual in the population, sorted by fitness.
        """
        return [ (ind[0], ind[1]) for ind in sorted(graded_pop, key=self._sorted_key) ]
    
    def _sorted_key(self, x):
        """
        Method used in sorting the population.

        Parameters:

        x (tuple) : A tuple (fitness, individual) 

        Returns:
        float: The fitness of the individual.
        """
        return x[0]
        
    def _get_parent_pop(self, parent_pop):
        """
        Method for extracting the individuals from a list of tuples (fitness, individual).

        Parameters:

        parent_pop (list) : The parent population, as a list of tuples (fitness, individual).

        Returns:
        List: A list of each individual within the parent population.
        """
        return [p[1] for p in parent_pop]
               
    def _get_initial_pop(self):
        """
        Method for obtaining an initial population.

        Returns:
        List: An initial population of random functions.
        """
        return [ self.random_function() for i in range(self.pop_size) ]
    

    def _get_penalty(self, prediction):
        """
        Method used to apply a penalty to the fitness, should it fall 
        outside a desired range.

        Parameters:

        prediction (float) : The fitness of the given individual.

        Returns:
        float: A rectified fitness should it have fallen outside the desired range.
        """
        if prediction < self.penalty_min or prediction > self.penalty_max:
            return 1000
        return prediction
        
        
    def fitness(self, individual, X=None, t=None):
        """
        Method used to obtain the fitness of an individual.

        Parameters:
        individual (Lambda) : The individual for which to obtain the fitness for.
        X (Iterable) : The (optional) training data (typically a NumPy array) to use to 
            train the prediction. If None, defaults to self.inputs.
        t (Iterable) : The (optional) training targets (typically a NumPy array) to use 
            to train the prediction. If None, defaults to self.targets.

        Returns:
        float: The fitness of the given individual.
        """
        if (X is None) or (t is None):
            X = self.inputs
            t = self.targets
            
        fit = 0
        for i, elements in enumerate(X):
            pred = individual(*elements)
            fit += abs(t[i] - self._get_penalty(pred))

        return fit

    def _tournament_selection(self, parents):
        """
        Method to perform tournament selection, used within the initial validation problem.

        Parameters:
        parents (list) : A list of parent individuals from which to perform the selection.

        Returns:
        List: A list of length 2 containing the two chosen individuals from two tournaments.
        """
        out = []
        parent_size = len(parents)
        for j in range(2):
            tournament = []
            for i in range(self.tournament_size):
                index = int(random.random() * parent_size)
                tournament.append(parents[index])
            
            out.append(max(tournament, key=lambda x: x[0])[1])
        return out
        


    def evolve(self, generations=10):
        """
        Method used to perform the evolution of the system.

        Parameters:
        generations (int) : The number of generations to operate over. Default 10.

        Returns:
        tuple: A tuple (best_fitness_func, avg_fitnesses, best_fitnesses) containing the following:
            - best_fitness_func which is the best fitting individual found during training.
            - avg_fitnesses which is a list containing the average fittnesses for the population at each generation.
            - best_fitnesses which is a list containing the best fitness in the population at each generation.
        """
        self.pop = self._get_initial_pop()
    
        desc_template = 'G: {:.0f}. M: {:.0f}, A: {:.0f}, BE: {:.0f}'
        loop = get_loop(generations)

        self.best_fitness = float('inf')
        self.best_fitness_func = None
        
        past_avg_fit = -1
        past_avg_fit_convergence_counter = 0
        
        avg_fitnesses = []
        best_fitnesses = []
        
        for gen in loop:
            graded_pop = self._grade_pop()
            sorted_pop = self._sort_pop(graded_pop)
            
            if (sorted_pop[0][0] < self.best_fitness):
                self.best_fitness = sorted_pop[0][0]
                self.best_fitness_func = sorted_pop[0][1]
            
            avg_fit = mean([i[0] for i in sorted_pop])
            avg_fitnesses.append(avg_fit)
            best_fitnesses.append(sorted_pop[0][0])
            
            loop.set_description(desc_template.format(gen, sorted_pop[0][0], avg_fit, self.best_fitness))
            tp = int(self.trunc_ratio * self.pop_size)

            if self.tournament_size is None:
                parent_pop = self._get_parent_pop(sorted_pop[:tp])
                self.pop = parent_pop.copy()
                for i in range(tp, self.pop_size):
                    p1, p2 = random.sample(parent_pop, 2)
                    self.pop.append(self.mutation(self.crossover(p1, p2)))
            else:
                self.pop.append(self.mutation(self.crossover(*self._tournament_selection(sorted_pop[:tp]))))
                
        return self.best_fitness_func, avg_fitnesses, best_fitnesses