import numpy as np
import math
t=0
s0=np.array([0,1])
state=s0
curpos=-1
qvals0=np.array([0.0,0.0])
qvals1=np.array([0.0,0.0])
s0=[1,0]
s0=np.array(s0)
result=[]
beta=1
from testinit import *
# no modification needed    
    
def gensample(replay,batchSize,curpos):
    '''
    Get the samples from replay memory
    Args:
        replay: replay memory
        batchSize: number of samples to be generated
    Output:
        samples: samples generated from certain replay rule
    '''


    # if len(replay)<batchSize:
    # 	samples=random.sample(replay, len(replay))
    # else:
    # 	samples=random.sample(replay, batchSize)
    samples=[replay[curpos]]

    return samples



def replace(curpos,state,action,reward,new_state):
    '''
    Set the rule to replace the memory
    Args:
		state: current state
		action: action taken
		reward: reward obtained by taking the action
		new_state: the next state
        curpos: current memory position
    Output:
        
        curpos: current memory position
    '''
    if(len(replay)<bufferSize):
        replay.append((state,action,reward,new_state))
        curpos += 1
    else:
        if (curpos < (bufferSize-1)):
            curpos += 1
        else:
            curpos = 0
        replay[curpos] = (state, action, reward, new_state)
    return curpos


def selectAct(qval,beta):
	'''
	Given the qval, select an action
	Args: state ,  and qval for all actions given that state
	Output: select an action according to softmax distribution
	'''

	threshold=[]

	#qval=model.predict(state.reshape(1,input_dimension),batch_size=1)
	qvalarr=qval.reshape(output_dimension,)
	#print "qvalarr ", qvalarr
	accsum=np.sum(np.exp(qvalarr/beta))

	#print "accsum: ", accsum
	prob=np.exp(qvalarr/beta)/accsum
	#print "prob: ", prob
	start=0
	for m in prob:
		start=start+m
		threshold.append(start)
	threshold=np.array(threshold)
	#print "threshold: ", threshold
	randnum=random.random()

	for i in range(len(threshold)):
		if randnum <= threshold[i]:
			#print i
			return i

while(t<T):
	if state[0]==0:
		qval=qvals0
	else:
		qval=qvals1


	action=selectAct(qval,beta)
	#print action
	#carry out the action 
	new_state=transition(state,action)
	#get the immediate reward for the new_state
	reward=getReward(state,action,new_state)



	#append the state,action pair to the result
	result.append((state,action))

	curpos=replace(curpos,state,action,reward,new_state)


	#print "state,action,reward,new_state is", state,action,reward,new_state
	#randomly sample our experience replay memory
	minibatch = gensample(replay,batchSize,curpos)
	X_train = []
	y_train = []
	for memory in minibatch:
	    #Get max_Q(S',a)
	    replay_old_state, replay_action, replay_reward, replay_new_state = memory

	    #for debug
	    #print "state,action,reward,new_state is", state,action,reward,new_state
	    print "replay_old_state, replay_action, replay_reward, replay_new_state is",  replay_old_state, replay_action, replay_reward, replay_new_state
	    
	    old_qval = qval
	    #newQ = model.predict(replay_new_state.reshape(1,input_dimension), batch_size=1)

	    #for debug:
	    #print "newQ is: " ,newQ

	    #maxQ = np.max(newQ)
	    #for debug
	    #print "maxQ is: ", maxQ


	    #y = np.zeros((output_dimension,))
	    y = old_qval
	    update = m[t] * y[replay_action] + a * (  p[t] * replay_reward  - y[replay_action] )
	    #update = p[t] * replay_reward
	    print "update is: " , update
	    print "replay_action is: ", replay_action
	    y[replay_action] = update
	    print "y[replay_action] is", y[replay_action]
	    #for debug
	    print "y is ", y
	    print "qvals0 is: ", qvals0
	    print "qvals1 is: ", qvals1


	    




	state = new_state
	t=t+1




#for debug
# ss0=np.array([0,1])
# ss1=np.array([1,0])
# print model.predict(ss0.reshape(1,input_dimension),batch_size=1)
# print model.predict(ss1.reshape(1,input_dimension),batch_size=1)

print result


'''
evaluations
'''
def evaluation(result):
	rr=0
	rl=0
	lr=0
	ll=0
	for i in range(len(result)):
		if result[i][0][0]==result[i][1] and result[i][0][0]==1:
			rr=rr+1
		elif result[i][0][0]==result[i][1] and result[i][0][0]==0:
			ll=ll+1
		elif result[i][0][0]!=result[i][1] and result[i][0][0]==1:
			rl=rl+1
		elif result[i][0][0]!=result[i][1] and result[i][0][0]==0:
			lr=lr+1
	if rr==0 or lr==0:
		print "domian error"
	elif rl==0 or ll==0:
		print "divide by zero error"
	else:
		print 0.5*math.log(float(rr*lr)/float(rl*ll))


evaluation(result)	

