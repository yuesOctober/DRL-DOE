from DoE import HyperParameterRange, DoE, DoEOption
from workspace import WorkspaceEnvironment
from bandit import BanditEnvironment
from ppretty import ppretty
import sys
import time
import random
import numpy as np
import operator

# random.seed(300)
# np.random.seed(200)

seed = random.randint(0, 2**32 - 1)
# seed = 300
random.seed(seed)
np.random.seed(seed)
print 'Seed:', seed

#Test generating samples using LHS. Use this example to illustrate how to use

# initHyperParameterRange=initHPRange(sgd_lrRng=[1e-2,1e-2],rhoRng=[0.3, 0.33927419781684875],gammaRng=[0,1],betaRng=[0.5,0.5],kRng=[1, 4.6012172698974609],thetaRng=[1,3],mRng=[1,1],cRng=[0,0])
# print initHyperParameterRange
# mysamples=generateLHS(initHyperParameterRange,10)

# samplelist=mysamples.samples
# print samplelist
# for i in range(len(samplelist)):
#     print "sample number: ", i
#     hyperparameters=samplelist[i].X
#     print "sgd_lr is: ", hyperparameters.sgd_lr
#     print "rho is: ",hyperparameters.rho
#     print "gamma is: ",hyperparameters.gamma
#     print "beta is: ",hyperparameters.beta
#     print "k is: ",hyperparameters.k
#     print "theta is: ",hyperparameters.theta
#     print "m is: ",hyperparameters.m
#     print "c is: ",hyperparameters.c


#Test runsimulator function

# myEnv=initEnv(input_dimension=2,output_dimension=11,T=180*3)
# s0=[1,0]
# s0=np.array(s0)
# initHyperParameterRange=initHPRange(sgd_lrRng=[3e-2,3e-2],rhoRng=[0,1],gammaRng=[0.6,0.6],betaRng=[0.5,0.5],kRng=[1,5],thetaRng=[2,2],mRng=[1,1],cRng=[0,0])
# print initHyperParameterRange
# mySamples=generateLHS(initHyperParameterRange,10)
# #mySamples.samples = list(reversed(mySamples.samples))
# samplesList=mySamples.samples
# print "Before run the simulation, print the samples properties:"
# for i in range(len(samplesList)):
#     eachSample=samplesList[i]
#     print "Sample ",i
#     print "Hyper-Parameters are: ",vars(eachSample.X)
#     print "Numerical results are: ", eachSample.Y


# print "After run the simulation, print the samples properties:"
# mySamples=runSimulator(mySamples,s0,myEnv)
# samplesList=mySamples.samples
# for i in range(len(samplesList)):
#     eachSample=samplesList[i]
#     print "Sample :",i
#     print "Hyper-Parameters are: ",vars(eachSample.X)
#     print "Numerical results are: ",eachSample.Y

# print "TEST calPhi:"
# mySamples=calPhi(mySamples,3)
# samplesList=mySamples.samples
# for i in range(len(samplesList)):
#     eachSample=samplesList[i]
#     print "Sample :",i
#     print "Hyper-Parameters are: ",vars(eachSample.X)
#     print "Numerical results are: ",eachSample.Y
#     print "Phi are: ",eachSample.phi

# print "TEST calZ:"
# mySamples=calZ(mySamples,0.8)
# mySamples=calPhi(mySamples,3)
# samplesList=mySamples.samples
# for i in range(len(samplesList)):
#     eachSample=samplesList[i]
#     print "Sample :",i
#     print "Hyper-Parameters are: ",vars(eachSample.X)
#     print "Numerical results are: ",eachSample.Y
#     print "Z are: ",eachSample.Z

#Sequential CART test.

initHyperParamRange = HyperParameterRange(sgd_lr=3e-2, rho=[0, 1], gamma=0.6, beta=0.5, k=1, theta=2, m=1, c=0)
baselineHyperParamRange = HyperParameterRange(sgd_lr=3e-2, rho=.3, gamma=0.6, beta=0.5, k=1, theta=2, m=1, c=0)
# initHyperParamRange = HyperParameterRange(sgd_lr=3e-2, rho=[0, .2], gamma=1, a=.1, beta=0.5, m=1)
# baselineHyperParamRange = HyperParameterRange(sgd_lr=3e-2, rho=1, gamma=1, a=.1, beta=0.5, m=1)
myEnv = WorkspaceEnvironment(T=180*3)
# myEnv = BanditEnvironment(T=100)
# initHyperParamRange = HyperParameterRange(sgd_lr=3e-2, rho=[0, 1], gamma=1, beta=0.5, k=1, theta=2, m=1, c=1)
# myEnv = BanditEnvironment(T=100*3)

#t-test baselineHyperParamRange
#not test: goal baseline= None, goal 1. #goalOp: ge.
#min_sampes_leaf=3/5

option = DoEOption(
    representation='QTable', # QTable, NN, or RNN2
    nSamples=50,
    min_samples_leaf=10,
    min_samples_leaf_2=2,
    max_leaf_nodes=5,
    minParamRange=2e-2,
    minParamGap=5e-2,
    nRunPerSample=5,
    goal=0.05,
    goalOp=operator.le,
    convHigh=0.95,
    convTruncation=0.05,
    maxLevel=10,
    keepRatio=0.8,
    partitionCount=4)

print 'Baseline:', ppretty(baselineHyperParamRange, seq_length=100)
print 'Initial:', ppretty(initHyperParamRange, seq_length=100)
print 'DoEOption:', ppretty(option, indent='    ', seq_length=100)

start = time.time()
hyperParamRanges = DoE(myEnv, option).sequentialCART2(initHyperParamRange, baselineHyperParamRange)

print [[(k, hyperParamRange[k]) for k in initHyperParamRange.get_range_param_names()] for hyperParamRange in hyperParamRanges]
print 'Total time: {:f}s'.format(time.time()-start)
