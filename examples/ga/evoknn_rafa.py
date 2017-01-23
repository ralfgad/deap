#!/usr/bin/env python2.7
#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import csv
import random

import numpy as np


import scipy.io as sio
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

datos_matlab = sio.loadmat('./idea.mat')
training=datos_matlab.get('training1_inicial')
conjunto=training[0,0] 
entradasI=conjunto[0].transpose()[:1000]
salidasT=conjunto[3].transpose()[:1000]
IND_SIZE=entradasI.shape[1]
def evalClassifier(individual):
    loo = LeaveOneOut()
    individuo=np.array(individual)
    entradasI_filt=entradasI[:,individuo==1]
    idea=0
    j=0;
    for train, test in loo.split(entradasI_filt):
        #print("%s %s" % (train, test))
        X_train, X_test, y_train, y_test = entradasI_filt[train], entradasI_filt[test], salidasT[train], salidasT[test]
        knn = KNeighborsRegressor(1, weights='uniform')
        clf = knn.fit(X_train, y_train)
        y_ = clf.predict(X_test)
        idea+=mean_squared_error(y_test,y_)
        j+=1;
    resultado=idea/(j)    
    return resultado

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operator registering
toolbox.register("evaluate", evalClassifier)
toolbox.register("mate", tools.cxUniform, indpb=0.1)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)


def ga_knn():
    # random.seed(64)
    MU, LAMBDA = 100, 200
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                             cxpb=0.7, mutpb=0.3, ngen=40, 
                                             stats=stats, halloffame=hof)
    
    return pop, logbook, hof

if __name__ == "__main__":
    ga_knn()
