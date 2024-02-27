import random
import math
import time
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
popSize = 300
elitismRate = 0.2
mutationRate = 1
crossOverRate = 0.6
generationNo = 200

fileName = "att48_d.txt"
chromosomeRollNo = 0
plotFlag = 1  # Assign 1 for plotting, 0 for not plotting

# Function to read TSP data from file
def readTSPData(fileName):
    with open(fileName, "r") as sourceFile:
        rawData = sourceFile.readlines()

    formattedData = []
    for line in rawData:
        data = list(map(float, line.split()))
        formattedData.append(data)
    return formattedData


dataMatrix = readTSPData(fileName)
cityList = [i for i in range(len(dataMatrix))]  # City List


class chromosomes:
    def __init__(self, route, chromosomeRollNo, parents=["NA", "NA"]):
        self.chromosomeRollNo = chromosomeRollNo
        self.route = route
        self.distance = 0.0
        self.fitnessScore = 0.0
        self.parents = parents

    def __repr__(self):
        return " " + str(self.chromosomeRollNo) + ") " + str(self.route) + " Cost = " + str(
            self.distance) + " Parents= " + str(self.parents) + "\n"


def generateInitialPopulation(popSize, initialPopulation, cityList):
    global chromosomeRollNo
    count = popSize

    while count > 0:
        chromosome = chromosomes(random.sample(cityList, len(cityList)), chromosomeRollNo)
        if chromosome not in initialPopulation:
            chromosomeRollNo += 1
            initialPopulation.append(chromosome)
            count -= 1


def fitnessOperator(chromosome):
    route = chromosome.route
    totalDistance = 0
    fromCity = 0
    toCity = 0
    for i in range(len(route) - 1):
        fromCity = int(route[i])
        toCity = int(route[i + 1])
        totalDistance += dataMatrix[fromCity][toCity]
    fromCity = toCity
    toCity = int(route[0])
    totalDistance += dataMatrix[fromCity][toCity]
    return totalDistance


def assignFitness(population):
    for i in population:
        i.distance = fitnessOperator(i)
        i.fitnessScore = 1 / i.distance


def elitism(population, eliteChromosomes, elitismRate):
    eliteSize = int(len(population) * elitismRate)
    sortedPopulation = sorted(population, key=lambda x: x.fitnessScore, reverse=True)
    for i in range(eliteSize):
        eliteChromosomes.append(sortedPopulation[i])


def rwSelection(population):
    totalFitness = sum(chromosome.fitnessScore for chromosome in population)
    P = random.random()
    cumulative_prob = 0
    for chromosome in population:
        cumulative_prob += chromosome.fitnessScore / totalFitness
        if cumulative_prob >= P:
            return chromosome


def selectParents(population, matingPool, eliteChromosomes, numberOfParents):
    count = numberOfParents
    while count > 0:
        selectedParent = rwSelection(population)
        if selectedParent not in matingPool:
            matingPool.append(selectedParent)
            count -= 1


def orderedCrossOver(parent1, parent2):
    randomPoint1 = random.randint(0, len(parent1) - 1)
    randomPoint2 = random.randint(0, len(parent1) - 0)

    startGene = min(randomPoint1, randomPoint2)
    endGene = max(randomPoint1, randomPoint2)

    # child 1
    child1 = parent1[startGene:endGene]
    parent2subset = [item for item in parent2 if item not in child1]
    child1 += parent2subset

    # child 2
    child2 = parent2[startGene:endGene]
    parent1subset = [item for item in parent1 if item not in child2]
    child2 += parent1subset

    return child1, child2


def generateChildren(matingPool, children):
    matingPool.sort(key=lambda x: x.distance)
    length = len(matingPool) - 1
    for i in range(0, length, 2):
        parent1 = matingPool[i]
        parent2 = matingPool[i + 1]

        child1, child2 = orderedCrossOver(parent1.route, parent2.route)

        parents1 = [parent1.chromosomeRollNo, parent2.chromosomeRollNo]
        childChromosome1 = chromosomes(child1, chromosomeRollNo, parents1)
        children.append(childChromosome1)

        parents2 = [parent2.chromosomeRollNo, parent1.chromosomeRollNo]
        childChromosome2 = chromosomes(child2, chromosomeRollNo + 1, parents2)
        children.append(childChromosome2)


def mutate(chromosome):
    route = chromosome.route
    routeLength = len(route)
    position1 = random.randint(0, routeLength - 1)
    position2 = random.randint(0, routeLength - 1)
    route[position1], route[position2] = route[position2], route[position1]


def mutateChildren(children, mutationRate):
    for chromosome in children:
        if random.random() < mutationRate:
            mutate(chromosome)


def createNextGeneration(population, eliteChromosomes, children):
    nextGeneration = eliteChromosomes + children
    remainingLength = len(population) - len(nextGeneration)
    nextGeneration += random.sample(population, remainingLength)
    return nextGeneration


def geneticAlgorithm():
    costList = []

    initialPopulation = []
    generateInitialPopulation(popSize, initialPopulation, cityList)
    population = initialPopulation.copy()

    for generation in range(generationNo):
        print("\n/////////////////////////////////////////////////////////")
        print("/////_____________ GENERATION: ", generation + 1, "_______________//")
        print("/////////////////////////////////////////////////////////\n")

        assignFitness(population)
        eliteChromosomes = []
        elitism(population, eliteChromosomes, elitismRate)

        matingPool = []
        selectParents(population, matingPool, eliteChromosomes, numberOfParents=popSize * crossOverRate)

        children = []
        generateChildren(matingPool, children)
        mutateChildren(children, mutationRate)

        population = createNextGeneration(population, eliteChromosomes, children)

        best_chromosome = min(population, key=lambda x: x.distance)
        print("Best Chromosome: ", best_chromosome)

        costList.append(best_chromosome.distance)

    return costList


takenTime = time.time()
costList = geneticAlgorithm()
takenTime = time.time() - takenTime
print("\nTime Taken: ", takenTime)

# Plotting
if plotFlag == 1:
    plt.title("TSP USING GENETIC ALGORITHM\n MIN COST = " + str(costList[-1]))
    plt.xlabel("Generations")
   
