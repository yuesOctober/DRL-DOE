######################## Simulator Code ###############################
# Author: Yue Shi
# Email: yueshi@usc.edu
#######################################################################

import abc
import math
import copy
import random
import numpy as np
from dotmap import DotMap
from nnmodel import gennnmodel, genrnnmodel
from keras.optimizers import SGD

class Simulator:
    def __init__(self, env):
        self.env = env

    def initNN(self, networkType, hyperParam):
        '''
        Initialize the neural network. change the architecture here(*). For RNN, change the timestep.
        Args:
            networkType: 'NN' or 'RNN'
            hyperParam: the hyper parameters
        Output:
            model: compiled model
        '''

        sgd_lr = hyperParam.sgd_lr
        sgd_decay = 1e-5
        sgd_mom = 0
        sgd_Nesterov = False

        # Generate Model
        if networkType == 'RNN':
            model = genrnnmodel(num_units=[self.env.input_dimension, 32, 32, self.env.output_dimension], actfn='tanh', last_act='linear', timestep=1)
        else:
            model = gennnmodel(num_units=[self.env.input_dimension, 5, 5, self.env.output_dimension], actfn='relu', last_act='linear', reg_coeff=0.0)

        sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_mom, nesterov=sgd_Nesterov)
        model.compile(loss='mse', optimizer=sgd)
        return model

    def initNN2(self, networkType, hyperParam):
        '''
        Initialize the neural network, used for simulateNN2 and simulateRNN2
        Args:
            S0: states for initialization
            A0: actions for initialization
        Output:
            model: initialized model
        '''

        sgd_lr = hyperParam.sgd_lr
        sgd_decay = 1e-5
        sgd_mom = 0
        sgd_Nesterov = False

        # Generate Model
        if networkType == 'RNN':
            model = genrnnmodel(num_units=[self.env.input_dimension,32,self.env.output_dimension], actfn='tanh', last_act='linear', timestep=10)
        else:
            model = gennnmodel(num_units=[self.env.input_dimension*self.env.output_dimension,100,50,1], actfn='relu', last_act='linear', reg_coeff=0.0)

        sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_mom, nesterov=sgd_Nesterov)
        model.compile(loss='mse', optimizer=sgd)
        return model

    @staticmethod
    def selectAct(qval, beta):
        '''
        Given the qval, select an action
        Args:
            qval: Q values all actions given a state
        Output: a, select an action according to softmax distribution
        '''

        threshold = []
        qvalarr = qval.reshape(len(qval[0]),)
        accsum = np.sum(np.exp(qvalarr/beta))
        prob = np.exp(qvalarr/beta) / accsum
        start = 0.0
        for m in prob:
            start = start + m
            threshold.append(start)
        threshold = np.array(threshold)
        randnum = random.random()
        for i in range(len(threshold)):
            if randnum <= threshold[i]:
                return i

    @staticmethod
    def gensample(replay, batchSize, curpos):
        '''
        Get the samples from replay memory
        Args:
            replay: replay memory
            batchSize: number of samples to be generated
        Output:
            samples: samples generated from certain replay rule
        '''
        # if len(replay)<batchSize:
        #   samples=random.sample(replay, len(replay))
        # else:
        #   samples=random.sample(replay, batchSize)
        samples = [replay[curpos]]
        return samples

    @staticmethod
    def replace(curpos, state, action, reward, new_state, replay, bufferSize, batchSize):
        '''
        Set the rule to replace the memory. For NN only
        Args:
            state: current state
            action: action taken
            reward: reward obtained by taking the action
            new_state: the next state
            curpos: current memory position
        Output:
            curpos: current memory position
        '''
        if len(replay) < bufferSize:
            replay.append((state, action, reward, new_state))
            curpos += 1
        else:
            if curpos < (bufferSize - 1):
                curpos += 1
            else:
                curpos = 0
            replay[curpos] = (state, action, reward, new_state)

        return curpos

    def simulateNN(self, s0, hyperParam, replay=None, bufferSize=80, batchSize=40):
        '''
        simulate using NN
        Args:
            s0: the initial state
            hyperParam: the hyper parameters setting.
            bufferSize: replay buffer size.
        Output:
            result: Result class instance(individual Q values, and (state,action) sequence)
        '''

        t = 0
        replay = replay if replay is not None else []
        resultActState = []
        model = self.initNN('NN', hyperParam)
        state = s0
        curpos = -1
        a = hyperParam.a if 'a' in hyperParam else 1
        m = self.env.init_m(hyperParam.m)
        rho = self.env.init_rho(hyperParam.rho)

        while t < self.env.T:
            qval = model.predict(state.reshape(1, self.env.input_dimension), batch_size=1)
            action = self.selectAct(qval, hyperParam.beta)
            new_state = self.env.transition(state, action, hyperParam)
            # print new_state
            # get the immediate reward for the new_state
            reward = self.env.get_reward(state, action, new_state, hyperParam, t)
            # append the state, action pair to the result
            resultActState.append([state, action, reward])
            curpos = self.replace(curpos, state, action, reward, new_state, replay, bufferSize, batchSize)
            minibatch = self.gensample(replay, batchSize, curpos)
            X_train = []
            y_train = []
            for memory in minibatch:
                replay_old_state, replay_action, replay_reward, replay_new_state = memory
                old_qval = model.predict(replay_old_state.reshape(1, self.env.input_dimension), batch_size=1)
                newQ = model.predict(replay_new_state.reshape(1, self.env.input_dimension), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1, self.env.output_dimension))
                y[:] = old_qval[:]
                update = m[t] * y[0][replay_action] + a * ( rho[t] * replay_reward + (hyperParam.gamma * maxQ) - y[0][replay_action] )
                y[0][replay_action] = update
                X_train.append(replay_old_state.reshape(self.env.input_dimension,))
                y_train.append(y.reshape(self.env.output_dimension,))
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            model.fit(X_train, y_train, batch_size=len(minibatch), epochs=1, verbose=0)
            state = new_state
            t += 1

        qvals = []
        for i in range(self.env.input_dimension):
            state = np.array([int(count == i) for count in range(self.env.input_dimension)])
            qvalEach = model.predict(state.reshape(1, self.env.input_dimension), batch_size=1)
            qvals.append(qvalEach)
        result = Result(qvals, resultActState)
        return result

    def simulateQTable(self, s0, hyperParam, replay=None, bufferSize=80, batchSize=40):
        '''
        simulation with QTable
        Args:
            s0: the initial state
            hyperParam: the hyperParamter setting.
        Output:
            result: Result class instance(individual Q values, and (state,action) sequence)
        '''
        t = 0
        replay = replay if replay is not None else []
        resultActState = []
        state = s0
        curpos =- 1
        a = hyperParam.a if 'a' in hyperParam else 0.1
        m = self.env.init_m(hyperParam.m)
        rho = self.env.init_rho(hyperParam.rho)
        resultqvals=[]
        qvals = np.array([np.array([[0.0 for count in range(self.env.output_dimension)]]) for _ in range(self.env.input_dimension)])
        # print qvals
        while t < self.env.T:
            for i in range(len(state)):
                if state[i] == 1:
                    qval = qvals[i]
                    break

            action = self.selectAct(qval, hyperParam.beta)
            new_state = self.env.transition(state, action, hyperParam)
            reward = self.env.get_reward(state, action, new_state, hyperParam, t)
            qvals_copy=copy.deepcopy(qvals)
            #print "copy:",qvals_copy
            resultqvals.append(qvals_copy)
            #print "qval:",qvals
            #print "list is:",resultqvals
            resultActState.append([state, action, reward])
            curpos = self.replace(curpos, state, action, reward, new_state, replay, bufferSize, batchSize)

            minibatch = self.gensample(replay, batchSize, curpos)
            y_train = []
            X_train = []
            for memory in minibatch:
                replay_old_state, replay_action, replay_reward, replay_new_state = memory
                for i in range(len(replay_old_state)):
                    if replay_old_state[i] == 1:
                        old_qval = qvals[i]
                        break
                for i in range(len(replay_new_state)):
                    if replay_new_state[i] == 1:
                        newQ = qvals[i]
                        break

                maxQ = np.max(newQ.reshape(self.env.output_dimension,))
                update = m[t] * old_qval[0][replay_action] + a * ( rho[t] * replay_reward + (hyperParam.gamma * maxQ) - old_qval[0][replay_action] )
                old_qval[0][replay_action] = update

            state = new_state
            t += 1
            
            #print "qvals:",qvals
            # if np.array_equal(qvals,qvals_copy):
            #     print t
            #     result = Result(resultqvals, resultActState)
            #     return result

        result = Result(resultqvals, resultActState)
        return result

    def simulateRNN2(self, s0, hyperParam, replay=None, timestep=10):
        '''
        Run the simulation with RNN, multiple timesteps, use buffersize as the timestep.
        Args:
            s0: the initial state
            hyperParam: the hyperParamter setting.
            timestep: the timesteps for sequence.
        Output:
            result: Result class instance(individual Q values, and (state,action) sequence)
        '''

        t = 0
        replay = replay if replay is not None else []
        resultActState = []
        model = self.initNN2('RNN', hyperParam)
        state = s0
        curpos = -1
        a = hyperParam.a if 'a' in hyperParam else 1
        m = self.env.init_m(hyperParam.m)
        rho = self.env.init_rho(hyperParam.rho)

        bufferSize = 1
        batchSize = 1
        result = self.getTrain('QTable', timestep=timestep, s0=s0, hyperParam=hyperParam, replay=replay, bufferSize=bufferSize, batchSize=batchSize)
        resultActState = result.resultActState
        states = self.getInitStates(result)

        while t < self.env.T - timestep:
            state = states[-1]
            old_qval = model.predict(states.reshape(1, timestep, self.env.input_dimension), batch_size=1)
            last_old_qval = np.array([old_qval[0][timestep-1]])
            action = self.selectAct(last_old_qval, hyperParam.beta)
            new_state = self.env.transition(state, action, hyperParam)
            reward = self.env.get_reward(state, action, new_state, hyperParam, t)
            resultActState.append([state, action, reward])
            popstates = np.delete(states, 0, axis=0)
            newstates = np.append(popstates, np.array([new_state]), axis=0)
            newQ = model.predict(newstates.reshape(1, timestep, self.env.input_dimension), batch_size=1)
            newQ = np.array([newQ[0][timestep-1]])
            maxQ = np.max(newQ)
            y = np.zeros((1, timestep, self.env.output_dimension))
            y[:] = old_qval[:]
            update = m[t] * y[0][timestep-1][action] + a * ( rho[t] * reward + (hyperParam.gamma * maxQ) - y[0][timestep-1][action] )
            y[0][timestep-1][action] = update
            model.fit(states.reshape(1, timestep, self.env.input_dimension), y, batch_size=1, epochs=1, verbose=0)
            states = newstates
            t += 1

        qval = model.predict(states.reshape(1, timestep, self.env.input_dimension), batch_size=1)
        result = Result(qval, resultActState)
        return result

    def getTrain(self, representation, timestep=None, **kwargs):
        oldT = self.env.T
        self.env.T = timestep or oldT

        if representation == 'QTable':
            result = self.simulateQTable(**kwargs)
        elif representation == 'NN':
            result = self.simulateNN(**kwargs)
        elif representation == 'RNN2':
            result = self.simulateRNN2(**kwargs)

        self.env.T = oldT
        return result

    def getInitStates(self, result):
        resultActState = result.resultActState
        states = np.array([result[0] for result in resultActState])
        return states


class Result:
    def __init__(self, qval, resultActState):
        self.qval = qval # qval for state 0.ss
        self.resultActState = resultActState # all the (state,action) pairs of the result

class HyperParameter(DotMap):
    # def __init__(self):
    #     self.sgd_lr = sgd_lr  #1e-3  # sgd_lr
    #     self.rho = rho  #1.0 # reward sensitivity
    #     self.gamma = gamma #0.6# gamma
    #     self.beta = beta #0.5 #temperature
    #     self.k = k   #1 # related to fire rate
    #     self.theta = theta   #2  # related to employment rate
    #     self.m = m   #1 #related to memory factor
    #     self.c = c  #0 barrier to find the job

    def to_array(self):
        return [self[k] for k in self.get_param_names()]

    def from_array(self, values):
        return HyperParameter({ k:v for k, v in zip(self.get_param_names(), values) })

    def get_param_names(self):
        return sorted(self.keys())

class Environment:
    __metaclass__  = abc.ABCMeta

    def __init__(self, input_dimension, output_dimension, s0, T):
        self.input_dimension = input_dimension # 2 # input dimension
        self.output_dimension = output_dimension  # 11 # output dimension
        self.s0 = s0
        self.T = T   # 180*3 # time setting.


    @abc.abstractmethod
    def init_m(self, m0):
        return np.array([m0 for i in range(self.T)])

    @abc.abstractmethod
    def init_rho(self, rho0):
        return np.array([rho0 for i in range(self.T)])

    def reward_gen(self, r, prob):
        '''
        Generate reward with certain prob
        Args:
            r: reward
            prob: probability
        Output: reward r according to prob distribution
        '''

        if random.random() < prob:
            return r
        else:
            return 0

    @abc.abstractmethod
    def transition(self, s, a, hyperParam):
        '''
        MDP transition.
        Args:
            s: current state
            a: selected action
        Output:
            s_next: next action
        '''
        # [0,1] is 1, [1,0] is 0

    @abc.abstractmethod
    def get_reward(self, s, a, s_next, hyperParam, t):
        '''
        Get the immediate reward.
        Args:
            s: state
            a: selected action
            s_next: next state
            t:time
        Output:
            r: immediate reward
        '''

    @abc.abstractmethod
    def evaluateEach(self, result):
        '''
        Get the interested goal of each individual behavior. for DoE.
        Args:
            result: the Result class
        Output:
            goal: How to define it depends. This should give the Y value in DoE. The goal could be avg_act,std_act, avg_state
            std_state, qvalss0, qvalss1 ,etc.  Will test with avg_act for the time being.

        '''

