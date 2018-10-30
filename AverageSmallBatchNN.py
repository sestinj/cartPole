import random, gym
from math import *
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC


rewardSum = 0
num = 0
average = 0
X = []
y = []
agent = SVC(gamma=0.001, max_iter=10)

env = gym.make('CartPole-v0')

for i_episode in range(200):
    observation = env.reset()
    tempX = []
    tempY = []
    totalReward = 0
    for t in range(250):
        #Render
        env.render()

        #Action
        if i_episode < 10:
            action = env.action_space.sample()
        else:
            action = agent.predict([[float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])]])[0]
            
        tempX.append([float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])])
        if action == 1:
            tempY.append(1)
        else:
            tempY.append(0)

        #Observe
        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            #Update NN
            print(totalReward)
            rewardSum += totalReward
            num += 1
            average = (rewardSum + totalReward)/num

            diff = totalReward-average
            for weight in range(abs(int(diff))):
                for newX in tempX:
                    X.append(newX)
                for newY in tempY:
                    if diff >= 0:
                        y.append(newY)
                    else:
                        y.append(abs(newY-1))

            #Forgetfullnes
            for i in range(min(len(X), 10)):
                X.pop(0)
                y.pop(0)

            agent = SVC(gamma=0.001, max_iter=10)
            agent.fit(X, y)
            break

