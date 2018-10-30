import random, gym
from math import *
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC


five = 10
topFiveX = []
topFivey = []
topFiveRewards = []
smallestReward = 0
X = []
y = []
agent = SVC(gamma=0.001, max_iter=10)
#agent = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 6, 2), random_state=1)

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
        if smallestReward < 30:
            action = env.action_space.sample()
        else:
            #print(action)
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
            if len(topFiveRewards) < five:
                for weight in range(int(totalReward)):
                    for temp in tempX:
                        X.append(temp)
                    for temp in tempY:
                        y.append(temp)

                topFiveX.append(tempX)
                topFivey.append(tempY)
                topFiveRewards.append(totalReward)
                smallestReward = min(topFiveRewards)
                    
            elif totalReward > smallestReward:
                index = topFiveRewards.index(min(topFiveRewards))
                topFiveRewards[index] = totalReward
                topFiveX[index] = tempX
                topFivey[index] = tempY

                X = []
                y = []
                for i in range(len(topFiveX)):
                    for weight in range(int(topFiveRewards[i])):
                        for new in topFiveX[i]:
                            X.append(new)
                for i in range(len(topFivey)):
                    for weight in range(int(topFiveRewards[i])):
                        for new in topFivey[i]:
                            y.append(new)
                print(topFiveRewards)

            agent = SVC(gamma=0.001, max_iter=10)
            agent.fit(X, y)
            smallestReward = min(topFiveRewards)
            break
