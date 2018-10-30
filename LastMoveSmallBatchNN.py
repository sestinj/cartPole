import random, gym
from math import *
from sklearn.svm import SVC

rewardSum = 0
num = 0
average = 0
X = []
y = []
agent = SVC(gamma=0.0001, max_iter=10)
#There need to be 2 classes in the inputs to fit
class1 = False
class0 = False
env = gym.make('CartPole-v0')

for i_episode in range(200):
    observation = env.reset()
    exX = 0
    exY = 0
    tempX = []
    tempY = []
    totalReward = 0
    for t in range(250):
        #Render
        env.render()

        #Action
        if not(class1 and class0):
            action = env.action_space.sample()
        else:
            action = agent.predict([[float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])]])[0]
            tempX.append([float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])])
        exX = [float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])]
        if action == 1:
            exY = 1
            tempY.append(1)
        else:
            exY = 0
            tempY.append(0)

        #Observe
        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            #Update NN
            print(totalReward)
            rewardSum += totalReward
            num += 1
            average = (rewardSum+totalReward)/num
            diff = totalReward-average
            relativeDiff = diff/average
            print("average = "+str(average))
            if relativeDiff > 0.75:
                print("BIg step:" +str(average))
                for weight in range(int(10*relativeDiff)):
                    for newX in tempX:
                        X.append(newX)
                    for newY in tempY:
                        y.append(newY)
            
            #X.append(exX)
            exY = abs(exY-1)
            #y.append(exY)

            if not(class1 and class0):
                for classnum in y:
                    if classnum == 0:
                        class0 = True
                    else:
                        class1 = True
    
            if class1 and class0:
                agent = SVC(gamma=0.0001, max_iter=10)
                agent.fit(X, y)
            break
