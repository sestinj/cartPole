#nn = Neural Network
#arch = Architecture (This is the raw definition of any nn, each item represents a layer and its numbrer of nodes) - The input and output layers are not included
initialArch = [8, 8, 4]
#mu = Percentage chance of mutating (expressed as double between [0, 1])
mu = 0.75
#mu2 = Magnitude of each mutation that occurs
mu2 = 15
#g = Number of generations.
g = 20
#n = Number of children in each generation
n = 8
#s = Number of survivors from each generation(the s top performing nn's move on, while n-s new nn's are created as offspring of the survivors)
s = 4
#initialAlpha = Initial value for regularization
initialAlpha = 1e-3
#probType = 'classification' or 'regression'
probType = "classification"

import random, gym
from math import *
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC

#Get data
X = []
y = []
env = gym.make('CartPole-v0')
for i_episode in range(100):
    observation = env.reset()
    tempX = []
    tempY = []
    totalReward = 0
    for t in range(200):
        env.render()
        action = env.action_space.sample()
        tempX.append([float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])])
        if action == 1:
            tempY.append(1)
        else:
            tempY.append(0)
        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            print(totalReward)
            if totalReward > 45:
                for temp in tempX:
                    X.append(temp)
                for temp in tempY:
                    y.append(temp)
            break

testX = X[0:6]
testY = y[0:6]
X = X[6:]
y = y[6:]








###COST FUNCTIONS:
costFunctions = ["score", "sumOfSquares"]
def scoreFunc(clf, testX, testY):
    theCost = 1.0-clf.score(testX, testY)
    return theCost
def sumOfSquaresFunc(clf, testX, testY):
    cost = 0
    for i in range(len(testX)):
        cost += float(testY[i] - clf.predict(testX)[i])**2.0
    return cost


#Solvers:
solvers = ['lbfgs', 'sgd', 'adam']


class NN:
    #Hidden layers
    arch = []
    #Regularization
    alpha = 1e-5
    solver = 'lbfgs'
    costFunc = 'score'
    isSVM = False
    cost = 0
    clf = 0

    #displays information about the nn. Each param is a bool, saying whether or not the attribute will be displayed
    def disp(self, arch, alpha, solver, costFunc, isSVM, cost):
        archstr = ""
        if arch:
            archstr = "Arch:" + str(self.arch)
        alphastr = ''
        if alpha:
            alphastr = "Alpha: " + str(self.alpha)
        solverstr = ''
        if solver:
            solverstr = "Solver: " + self.solver
        costFuncstr = ''
        if costFunc:
            costFuncstr = "Cost Func: " + self.costFunc
        isSVMstr = ''
        if isSVM:
            isSVMstr = "SVM: " + str(self.isSVM)
        coststr = ''
        if cost:
            coststr = "Cost: " + str(self.cost)

        return archstr + "\n    " + alphastr + "\n    " + solverstr + "\n    " + costFuncstr + "\n    " + coststr + "\n    " + isSVMstr
    
    #run returns the cost of the NN, given the inputs X and y.
    def run(self, X, y):
        cost = 0

        #Using scikit-learn to fit neural network
        if probType == "classification":
            #Add in the ability to use Support Vector Machines, because they can take the same input, but give drastically more accurate output for certain problems
            if self.isSVM:
                print("                                                   SSSSSSSSSSSVVVVVVVVVVVVVVVVVVVMMMMMMMMMMMM")
                clf = SVC(gamma=0.001,max_iter=10)
            else:
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        else:
            clf = MLPRegressor(solver=self.solver, alpha=self.alpha, hidden_layer_sizes=self.arch, random_state=1)

        clf.fit(X, y)
        self.clf = clf
        
        #Determine cost (try to implement and mutate various cost functions)
        cost = scoreFunc(clf, testX, testY)
        
        
        
        return cost

    def mutate(self, mu, mu2):
        #Based on the mutation rate mu, possibly change the number of nodes in each layer
        for i in range(len(self.arch)):
            if random.random() < mu:
                #The max mutation is adding or subtracting mu2 nodes to a single layer
                self.arch[i] += random.randint(-mu2, mu2)

        #There is a mu(as a percentage) chance of adding/subtracting a layer. It is randomly placed anywhere after the input and before the output. The size is in the interval [1, mu2].
        if random.random() < mu:
            if random.random() <= 0.5:
                #Only remove a layer if there will be at least one hidden layer left
                if len(self.arch) > 3:
                    self.arch.pop(random.randint(0, len(self.arch)-1))
            else:
                self.arch.insert(random.randint(0, len(self.arch)-1), random.randint(1, mu2))

        #If any layers have n<1 nodes, replace that number with mu2
        for i in range(len(self.arch)):
            if self.arch[i] < 1:
                self.arch[i] = 1

        #Mutate alpha
        if random.random() < mu:
            if random.random() <= 0.5:
                self.alpha = self.alpha * 10.0
            else:
                self.alpha = self.alpha/10.0

        #Mutate cost function
        if random.random() < mu:
            self.costFunc = costFunctions[random.randint(0, len(costFunctions)-1)]

        #Mutate solver
        if random.random() < mu:
            self.solver = solvers[random.randint(0, len(solvers)-1)]
        
        #Mutate isSVM (this shouldn't happen as often)
        if random.random() < mu/10.0:
            self.isSVM = not self.isSVM
            
        return None

    #Returns a matrix of theta values(weights)
    def __init__(self, arch, alpha, solver, costFunc, isSVM):
        self.arch = arch
        self.alpha = alpha
        self.solver = solver
        self.costFunc = costFunc
        self.isSVM = isSVM

#Creates an offspring architecture as a genetic combination of the two given architectures
def breedArch(nn1, nn2):
    arch1 = nn1.arch
    arch2 = nn2.arch

    newArch = []
    for i in range(min(len(arch1), len(arch2))):
        if random.random() >= 0.5:
            newArch.append(arch1[i])
        else:
            newArch.append(arch2[i])
            

    #Take average of parents' alpha values
    alpha = (nn1.alpha + nn2.alpha)/2.0

    #Randomly choose one of the parents' solvers and costFuncs
    solver = nn1.solver
    if random.random() <= 0.5:
        solver = nn2.solver

    costFunc = nn1.costFunc
    if random.random() <= 0.5:
        costFunc = nn2.costFunc

    isSVM = nn1.isSVM
    if random.random() <= 0.5:
        isSVM = nn2.isSVM

    return NN(newArch, alpha, solver, costFunc, isSVM)

#nns = Array of all current Neural Network objects
nns = []
#Initialize primary nn's
lowestCost = 9999999999
bestCLF = 0
for i in range(n):
    nns.append(NN(initialArch, initialAlpha, 'lbfgs', 'score', False))



for i in range(g):
    #print("-------------------------------------------------------------------------------------------------------------------------------------")
    #print("Generation #" + str(i+1) + ":")
    #Get the cost of each nn
    costs = []
    for j in range(n):
        cost = nns[j].run(X, y)
        nns[j].cost = cost
        costs.append(cost)
        if cost < lowestCost:
            lowestCost = cost
            bestCLF = nns[j].clf
        #print(str(j+1) + ".  " + nns[j].disp(True, True, True, True, True, True))
        
    survivors = []
    #Decide which nn's survive
    for j in range(s):
        survivors.append(nns[costs.index(min(costs))])
        
    offspring = []
    #Breed new nn's
    for j in range(n-s):
        nn1 = survivors[random.randint(0, len(survivors)-1)]
        nn2 = survivors[random.randint(0, len(survivors)-1)]
        offspring.append(breedArch(nn1, nn2))

    #Mutate only the offspring, not the survivors from the last generation
    for nn in offspring:
        nn.mutate(mu, mu2)
    
    #Replace nns with survivors and offspring
    nns = survivors
    for i in range(len(offspring)):
        nns.append(offspring.pop(0))
        
print("LowCost:" + str(lowestCost))


##for i_episode in range(25):
##    observation = env.reset()
##    totalReward = 0
##    for t in range(100):
##        env.render()
##        action = bestCLF.predict([[float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])]])
##        observation, reward, done, info = env.step(action[0])
##        totalReward += reward
##        if done:
##            print(str(totalReward))
##            break

#Get data
X = []
y = []
for i_episode in range(100):
    observation = env.reset()
    tempX = []
    tempY = []
    totalReward = 0
    for t in range(200):
        env.render()
        action = bestCLF.predict([[float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])]])
        tempX.append([float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])])
        if action == 1:
            tempY.append(1)
        else:
            tempY.append(0)
        observation, reward, done, info = env.step(action[0])
        totalReward += reward
        if done:
            print(totalReward)
            if totalReward > 70:
                for temp in tempX:
                    X.append(temp)
                for temp in tempY:
                    y.append(temp)
            break

testX = X[0:6]
testY = y[0:6]
X = X[6:]
y = y[6:]




#nns = Array of all current Neural Network objects
nns = []
#Initialize primary nn's
lowestCost = 9999999999
bestCLF = 0
for i in range(n):
    nns.append(NN(initialArch, initialAlpha, 'lbfgs', 'score', False))



for i in range(g):
    #print("-------------------------------------------------------------------------------------------------------------------------------------")
    #print("Generation #" + str(i+1) + ":")
    #Get the cost of each nn
    costs = []
    for j in range(n):
        cost = nns[j].run(X, y)
        nns[j].cost = cost
        costs.append(cost)
        if cost < lowestCost:
            lowestCost = cost
            bestCLF = nns[j].clf
        #print(str(j+1) + ".  " + nns[j].disp(True, True, True, True, True, True))
        
    survivors = []
    #Decide which nn's survive
    for j in range(s):
        survivors.append(nns[costs.index(min(costs))])
        
    offspring = []
    #Breed new nn's
    for j in range(n-s):
        nn1 = survivors[random.randint(0, len(survivors)-1)]
        nn2 = survivors[random.randint(0, len(survivors)-1)]
        offspring.append(breedArch(nn1, nn2))

    #Mutate only the offspring, not the survivors from the last generation
    for nn in offspring:
        nn.mutate(mu, mu2)
    
    #Replace nns with survivors and offspring
    nns = survivors
    for i in range(len(offspring)):
        nns.append(offspring.pop(0))
        
print("LowCost:" + str(lowestCost))

##for i_episode in range(25):
##    observation = env.reset()
##    totalReward = 0
##    for t in range(100):
##        env.render()
##        action = bestCLF.predict([[float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])]])
##        observation, reward, done, info = env.step(action[0])
##        totalReward += reward
##        if done:
##            print(str(totalReward))
##            break

#Get data
X = []
y = []
for i_episode in range(200):
    observation = env.reset()
    tempX = []
    tempY = []
    totalReward = 0
    for t in range(200):
        env.render()
        action = bestCLF.predict([[float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])]])
        tempX.append([float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])])
        if action == 1:
            tempY.append(1)
        else:
            tempY.append(0)
        observation, reward, done, info = env.step(action[0])
        totalReward += reward
        if done:
            print(totalReward)
            if totalReward > 150:
                for temp in tempX:
                    X.append(temp)
                for temp in tempY:
                    y.append(temp)
            break

testX = X[0:6]
testY = y[0:6]
X = X[6:]
y = y[6:]


#nns = Array of all current Neural Network objects
nns = []
#Initialize primary nn's
lowestCost = 9999999999
bestCLF = 0
for i in range(n):
    nns.append(NN(initialArch, initialAlpha, 'lbfgs', 'score', False))

for i in range(g):
    #print("-------------------------------------------------------------------------------------------------------------------------------------")
    #print("Generation #" + str(i+1) + ":")
    #Get the cost of each nn
    costs = []
    for j in range(n):
        cost = nns[j].run(X, y)
        nns[j].cost = cost
        costs.append(cost)
        if cost < lowestCost:
            lowestCost = cost
            bestCLF = nns[j].clf
        #print(str(j+1) + ".  " + nns[j].disp(True, True, True, True, True, True))
        
    survivors = []
    #Decide which nn's survive
    for j in range(s):
        survivors.append(nns[costs.index(min(costs))])
        
    offspring = []
    #Breed new nn's
    for j in range(n-s):
        nn1 = survivors[random.randint(0, len(survivors)-1)]
        nn2 = survivors[random.randint(0, len(survivors)-1)]
        offspring.append(breedArch(nn1, nn2))

    #Mutate only the offspring, not the survivors from the last generation
    for nn in offspring:
        nn.mutate(mu, mu2)
    
    #Replace nns with survivors and offspring
    nns = survivors
    for i in range(len(offspring)):
        nns.append(offspring.pop(0))
        
print("LowCost:" + str(lowestCost))

for i_episode in range(25):
    observation = env.reset()
    totalReward = 0
    for t in range(200):
        env.render()
        action = bestCLF.predict([[float(observation[0]), float(observation[1]), float(observation[2]), float(observation[3])]])
        observation, reward, done, info = env.step(action[0])
        totalReward += reward
        if done:
            print(str(totalReward))
            break
