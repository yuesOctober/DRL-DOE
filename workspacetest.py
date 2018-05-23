#Code test:

from  workspace import WorkspaceEnvironment
from simulator import Environment, HyperParameter,Result,Simulator
import numpy as np
#NN TEST:

# s0=[0,1]
# s0=np.array(s0)
# replay=[]
# bufferSize=80
# batchSize=40
# myEnv=initEnv(input_dimension=2,output_dimension=11,T=180*3)
# myHyperParameters=initHP(sgd_lr=3e-2, rho=1,gamma=0.6,beta=0.5,k=1,theta=2,m=1,c=1)
# myResult = simulateNN(s0,myHyperParameters,myEnv,replay,bufferSize,batchSize)
# print evaluateEach(myResult)	
# print myResult.qval
#plotSeq(myResult) 



#NN2 TEST:

# s0=[0,1]
# s0=np.array(s0)
# replay=[]
# bufferSize=80
# batchSize=40
# myEnv=initEnv(input_dimension=2,output_dimension=11,T=180*3)
# myHyperParameters=initHP(sgd_lr=3e-2, rho=1,gamma=0.6,beta=0.5,k=1,theta=2,m=1,c=1)
# timestep=10
# myResult = simulateRNN2(s0,myHyperParameters,myEnv,timestep)
# print evaluateEach(myResult)	
# print myResult.qval
#plotSeq(myResult) 

#RNN TEST:
#rho=0.1,gamma=0.6,beta=0.5,k=1,theta=2,m=1,c=0
s0=[1,0]
s0=np.array(s0)
timestep=10
env= WorkspaceEnvironment(T=2000)
hyperParam=HyperParameter(rho=1,m=1,k=1,beta=0.5,gamma=0.6,theta=2,c=0.5)
result=Simulator(env).simulateQTable(s0, hyperParam, replay=None, bufferSize=80, batchSize=40)
#print result.resultActState 
print result.qval 
#Q-TABLE TEST:
# s0=[0,1]
# s0=np.array(s0)
# replay=[]
# bufferSize=80
# batchSize=40
# myHyperParameters=initHP(sgd_lr=3e-2, rho=1,gamma=0.6,beta=0.5,k=1,theta=2,m=1,c=1)
# myEnv=initEnv(input_dimension=2,output_dimension=11,T=180*3)
# myResult=simulateQTable(s0,myHyperParameters,myEnv,replay,bufferSize,batchSize)
# print evaluateEach(myResult)	
# print myResult.qval
# plotSeq(myResult) 


#timestep=1, RNN test

# s0=[1,0]
# s0=np.array(s0)
# replay=[]
# bufferSize=80
# batchSize=40
# myEnv=initEnv(input_dimension=2,output_dimension=11,T=180*3)
# myHyperParameters=initHP(sgd_lr=1e-1, rho=0.1,gamma=0.6,beta=0.5,k=1,theta=2,m=1,c=0)
# myResult=simulateRNN(s0,myHyperParameters,myEnv,replay,bufferSize,batchSize)
# print evaluateEach(myResult)	
# print myResult.qvalss0
# plotSeq(myResult)            



#For debug purpose

# def run(p1):
#     global p, result, replay, s0
#     p=initp(T,p1)
#     random.setstate(rs)
#     np.random.set_state(nrs)
#     result=[]
#     replay=[]
#     result,model=simulate(s0)
#     evaluation(result)

# for _ in range(10):
#     seed = random.randint(0, 836537583)
#     #seed = 323196643
#     random.seed(seed)
#     np.random.seed(seed)
#     print "SEED:", seed

#     rs = random.getstate()
#     nrs = np.random.get_state()

#     run(1)
#     run(2)	     