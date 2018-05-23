from simulator import Environment
import numpy as np

class WorkspaceEnvironment(Environment):
    def __init__(self, T):
        Environment.__init__(self,
            input_dimension=2,
            output_dimension=3,
            s0=np.array([0,1]),
            T=T)
    def init_m(self, m0):
        return np.array([m0 for i in range(self.T)])
    def init_rho(self, rho0):
        return np.array([rho0 for i in range(self.T)])

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
        #rho=1,m=1,k=1,beta=0.5,gamma=0.6,theta=2,c=0.5
        if s[0] == 0:
            prob = 1.0 / (1 + hyperParam.k * np.exp(0-a))
        else:
            prob = 1.0 / (hyperParam.theta + 100 * np.exp(0-a))

        x = self.reward_gen(1, prob)

        if x == 1:
            s_next = [0,1]
        else:
            s_next = [1,0]

        s_next = np.array(s_next)
        return s_next

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
        # [0,1] is 1, [1,0] is 0
        if s[0] == 0 and a == 0:
            r = 0
        elif s[0] == 0 and a > 0:
            r = 1
        elif s[0] == 1:
            r = -a * hyperParam.c
        return r

    def evaluateEach(self, result):
        '''
        Get the interested goal of each individual behavior. for DoE.
        Args:
            result: the Result class
        Output:
            goal: How to define it depends. This should give the Y value in DoE. The goal could be avg_act,std_act, avg_state
            std_state, qvalss0, qvalss1 ,etc.  Will test with avg_act for the time being.

        '''

        state0 = 0
        state1 = 0
        actions0 = []
        actions1 = []
        payment=0.0
        result = result.resultActState
        for i in range(len(result)):
            if result[i][0][0] == 0:
                if result[i][1] >0 :
                    payment = payment + 1
                state1 += 1
                actions1.append(result[i][1])
            elif result[i][0][0] == 1:
                state0 += 1
                actions0.append(result[i][1])
        #payrate=float(payment)/len(result)
        avg_act_state1 = np.std(actions1, axis=0)
        return avg_act_state1
        #return payrate
