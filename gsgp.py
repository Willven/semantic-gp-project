import random
from tqdm import tqdm
from util import *
from statistics import mean

inf = float('inf')

def memoize(f):
    f.cache = {}
    def decorated_function(*args):
        if args in f.cache:
            return f.cache[args]
        else:
            f.cache[args] = f(*args)
            return f.cache[args]
    return decorated_function

class GSGP:
    def __init__(self, mutation_rate, input_vars, max_depth, inputs, targets, operators, \
                 pop_size, trunc_ratio=0.5, tournament_size=5, max_fitness=None):
        self.mutation_rate = mutation_rate
        self.vars = input_vars
        self.max_depth = max_depth
        self.inputs = inputs
        self.targets = targets
        self.operators = operators
        self.pop_size = pop_size
        self.trunc_ratio = trunc_ratio
        self.max_fitness = max_fitness
        self.tournament_size = tournament_size
    
    def random_expression(self, depth):
        if depth == 1 or random.random() < 1/(2**depth-1):
            return random.choice((*self.vars, *[str(random.random())] * 10))
        else:
            return '(' + random.choice(self.operators) + '(' + \
                self.random_expression(depth - 1) + ',' + \
                self.random_expression(depth - 1) + '))'
        
    def random_function(self):
        re = self.random_expression(self.max_depth)

        rf = eval('lambda ' + ', '.join(self.vars) + ': ' + re)
        rf = memoize(rf)
        rf.geno = lambda: re
        return rf
    
    def crossover(self, p1, p2):
        k = random.random()
        offspring = lambda *x: k * p1(*x) + (1-k) * p2(*x)
        offspring = memoize(offspring) # add cache
        offspring.geno = lambda: '(('+ str(k) + '*' + p1.geno() + ') + ((1-' + str(k) + ')' +  p2.geno() + '))'
        return offspring
    
    def mutation(self, p):
        rm = self.random_function()
        rn = self.random_function()
        
        offspring = lambda *x: p(*x) - (self.mutation_rate * (rn(*x) - rm(*x)))
        # offspring = lambda *x: p(*x) + self.mutation_rate * (rm(*x) * p(*x) - rn(*x) * p(*x))
        offspring = memoize(offspring) # add cache

        offspring.geno = lambda: '((' + p.geno() - '(' + str(self.mutation_rate) + '*(' + rm.geno() + '*' + rn.geno() + '))))'

        # offspring.geno = lambda: '((' + p.geno() + str(self.mutation_rate) + '*(' + rm.geno() + '*' + p.geno() + \
                            # '-' + rn.geno() + '*' + p.geno() + ')))'
        return offspring
    
    def _grade_pop(self):
        return [ (self.fitness(ind), ind) for ind in self.pop ]
    
    def _sort_pop(self, graded_pop):
        return [ (ind[0], ind[1]) for ind in sorted(graded_pop, key=self._sorted_key) ]
    
    def _sorted_key(self, x):
        return x[0]
        
    def _get_parent_pop(self, parent_pop):
        return [p[1] for p in parent_pop]

    def _do_crossover_mutation(self, parent_pop):
        for i in range(self.pop_size - len(parent_pop)):
            parent = random.sample(parent_pop, 2)
            self.pop.append(self.mutation(self.crossover(parent[0][1], parent[1][1])))
               
    def _get_initial_pop(self):
        return [ self.random_function() for i in range(self.pop_size) ]
    

    def _get_value(self, individual, elements):
        val = individual(*elements)
        if val > 7:
            return 7
        if val < 0:
            return 0
        return val
        
    def fitness(self, individual, X=None, t=None):
        if (X is None) or (t is None):
            X = self.inputs
            t = self.targets
            
        fit = 0
        for i, elements in enumerate(X):
            if int(t[i]) != self._get_value(individual, elements):
                fit += 1


            # fit += abs(int(t[i]) - self._get_value(individual, elements))
        
        return fit
        # return min(fit / len(X), self.max_fitness)

    def _tournament_selection(self, parents):
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
        self.pop = self._get_initial_pop()

        if self.max_fitness is None:
            self.max_fitness = 7*len(self.inputs)
    
        desc_template = 'G: {:.0f}. M: {:.0f}, A: {:.0f}, BE: {:.0f}'
        loop = tqdm(range(generations))
        
        self.best_fitness = inf
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

            parent_pop = sorted_pop[:int(self.trunc_ratio * self.pop_size)]
            
            if avg_fit == past_avg_fit:
                past_avg_fit_convergence_counter += 1
                if past_avg_fit_convergence_counter >= 10:
                    print('Early convergence detected.')
                    break
            else:
                past_avg_fit = avg_fit
                past_avg_fit_convergence_counter = 0


            # for i in range(self.pop_size):
                # p1, p2 = self._tournament_selection(parent_pop)
                # self.pop[i] = self.mutation(self.crossover(p1, p2))

            parent_pop = self._get_parent_pop(parent_pop)
            for i in range(self.pop_size):
                p1, p2 = random.sample(parent_pop, 2)
                self.pop[i] = self.mutation(self.crossover(p1, p2))
            # self.pop = self._get_parent_pop(parent_pop)
            # self._do_crossover_mutation(parent_pop)

        return self.best_fitness_func, avg_fitnesses, best_fitnesses