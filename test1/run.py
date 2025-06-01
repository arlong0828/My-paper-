import time
import random
import argparse
import matplotlib.pyplot as plt
# from plotRoute import plot_route
import pandas as pd 
from CW import Vrp
random.seed(0) 
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="./test1/Input_Data.json", required=False,
                        help="Enter the input Json file name")
    parser.add_argument('--pop_size', type=int, default=29, required=False,
                        help="Enter the population size")
    parser.add_argument('--mute_prob', type=float, default=0.8, required=False,
                        help="Mutation Probabilty")
    parser.add_argument('--iterations', type=int, default=5000, required=False,
                        help="Number of iterations to run")

    return parser.parse_args()

def load(file):
    file_path = file  # 请替换为实际文件路径
    with open(file_path, 'r') as file:
        data0 = file.read()

    # 将数据转换为DataFrame
    data = []
    for line in data0.strip().split('\n'):
        data.append(line.split())

    columns = ["CUST", "XCOORD.", 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df = pd.DataFrame(data[9:], columns=columns)

    # 将字符型列转换为数字
    numeric_cols = ['CUST', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY', 'DUE', 'SERVICE']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['sum'] = df['DUE'] - df['READY']
    # print(np.mean(df['sum']))
    return df

def load_instances():
    json_instance = []
    json_instances = []
    se_vehicle_capacity = 0
    dataset = [
        "./test1/r1/r101.txt",
        "./test1/r1/r102.txt",
        "./test1/r1/r103.txt",
        "./test1/r1/r104.txt",
        "./test1/r1/r105.txt",
        "./test1/r1/r106.txt",
        "./test1/r1/r107.txt",
        "./test1/r1/r108.txt",
        "./test1/r1/r109.txt",
        "./test1/r1/r110.txt",
        "./test1/r1/r111.txt",
        "./test1/r1/r112.txt",
        "./test1/r2/r201.txt",
        "./test1/r2/r202.txt",
        "./test1/r2/r203.txt",
        "./test1/r2/r204.txt",
        "./test1/r2/r205.txt",
        "./test1/r2/r206.txt",
        "./test1/r2/r207.txt",
        "./test1/r2/r208.txt",
        "./test1/r2/r209.txt",
        "./test1/r2/r210.txt",
        "./test1/r2/r211.txt",
        "./test1/rc1/rc101.txt",
        "./test1/rc1/rc102.txt",
        "./test1/rc1/rc103.txt",
        "./test1/rc1/rc104.txt",
        "./test1/rc1/rc105.txt",
        "./test1/rc1/rc106.txt",
        "./test1/rc1/rc107.txt",
        "./test1/rc1/rc108.txt",
        "./test1/rc2/rc201.txt",
        "./test1/rc2/rc202.txt",
        "./test1/rc2/rc203.txt",
        "./test1/rc2/rc204.txt",
        "./test1/rc2/rc205.txt",
        "./test1/rc2/rc206.txt",
        "./test1/rc2/rc207.txt",
        "./test1/rc2/rc208.txt"
    ]
                                                                                       #| 客戶20           | #|客戶30          |
    test_num = [26, 14, 23, 19, 25, 5, 11, 2, 17, 36, 4, 1, 8, 6, 33, 0, 37, 16, 21, 35, 22, 24, 15, 10 ,3, 28, 12, 34, 38, 31]
    data_num_20 = [
        [38, 13, 51, 52, 88, 58, 54, 100, 62, 6, 49, 85, 61, 90, 74, 76, 37, 20, 66, 25],
        [84, 74, 1, 19, 26, 85, 98, 41, 82, 55, 25, 57, 16, 54, 43, 88, 91, 47, 66, 21],
        [34, 8, 50, 64, 41, 1, 98, 2, 58, 46, 71, 56, 76, 40, 9, 24, 94, 52, 60, 69],
        [57, 16, 11, 52, 34, 10, 45, 64, 84, 51, 22, 31, 86, 6, 14, 70, 80, 63, 98, 28],
        [91, 6, 23, 100, 85, 22, 58, 71, 9, 8, 28, 76, 34, 47, 35, 37, 60, 53, 80, 68],
        [55, 85, 100, 80, 59, 97, 4, 18, 91, 75, 57, 14, 52, 16, 71, 2, 72, 68, 73, 92],
        [17, 30, 72, 29, 3, 34, 38, 89, 100, 63, 39, 35, 16, 46, 73, 40, 6, 54, 33, 49],
        [66, 80, 33, 65, 31, 39, 45, 79, 62, 1, 84, 15, 91, 32, 6, 90, 49, 93, 52, 64],
        [61, 66, 49, 70, 92, 26, 20, 44, 41, 82, 85, 73, 7, 25, 27, 98, 76, 56, 91, 1],
        [32, 18, 50, 56, 89, 51, 24, 62, 23, 49, 58, 8, 74, 97, 64, 16, 9, 5, 68, 42],


        [49, 97, 18, 46, 7, 98, 29, 81, 41, 36, 70, 11, 74, 90, 3, 27, 57, 76, 62, 78],
        [83, 79, 34, 68, 67, 69, 33, 18, 28, 85, 81, 35, 64, 58, 48, 40, 27, 7, 41, 21],
        [9, 25, 5, 40, 91, 21, 28, 54, 27, 95, 3, 79, 84, 41, 2, 38, 43, 76, 97, 37],
        [72, 36, 94, 68, 17, 12, 91, 63, 9, 52, 79, 93, 78, 30, 1, 70, 50, 83, 59, 49],
        [44, 24, 56, 53, 26, 80, 91, 18, 77, 57, 86, 8, 17, 59, 19, 14, 99, 94, 45, 51],
    ]
    data_num_30 = [
        [27, 52, 10, 19, 81, 99, 55, 13, 1, 16, 42, 33, 47, 38, 14, 45, 64, 91, 21, 88, 92, 97, 17, 8, 39, 86, 57, 77, 87, 54],
        [65, 0, 57, 87, 53, 41, 23, 36, 22, 93, 38, 46, 96, 79, 99, 92, 43, 95, 77, 67, 59, 72, 90, 70, 60, 42, 56, 11, 78, 91],
        [87, 26, 57, 76, 65, 52, 8, 96, 1, 59, 75, 98, 30, 95, 0, 41, 47, 28, 69, 32, 61, 14, 91, 74, 78, 60, 5, 20, 2, 90],
        [21, 88, 31, 68, 83, 57, 76, 29, 79, 52, 28, 84, 85, 34, 42, 35, 3, 72, 100, 77, 91, 94, 60, 32, 1, 89, 67, 74, 96, 13],
        [41, 2, 28, 1, 77, 37, 25, 49, 76, 18, 87, 57, 66, 42, 71, 5, 54, 79, 68, 97, 56, 40, 14, 46, 15, 26, 55, 17, 62, 100],
        [15, 83, 42, 75, 17, 34, 92, 64, 84, 87, 62, 63, 9, 76, 30, 72, 19, 73, 47, 58, 37, 78, 4, 97, 85, 32, 38, 96, 36, 49],
        [6, 89, 15, 78, 79, 83, 43, 85, 24, 18, 48, 90, 57, 31, 22, 65, 74, 16, 38, 51, 86, 5, 50, 71, 53, 58, 98, 8, 2, 28],
        [28, 45, 90, 5, 65, 32, 81, 37, 64, 34, 27, 7, 39, 0, 68, 95, 1, 18, 11, 89, 93, 70, 75, 48, 12, 55, 10, 62, 43, 57],
        [38, 95, 6, 23, 30, 99, 77, 37, 16, 79, 82, 28, 12, 76, 78, 0, 42, 74, 36, 92, 46, 33, 93, 64, 7, 57, 53, 49, 2, 81],
        [25, 27, 48, 47, 11, 49, 51, 97, 58, 67, 38, 81, 76, 57, 32, 88, 43, 42, 62, 74, 7, 50, 70, 31, 30, 41, 98, 24, 14, 69],


        [62, 46, 13, 48, 70, 37, 31, 81, 2, 74, 55, 63, 49, 61, 1, 82, 83, 71, 58, 90, 28, 40, 35, 100, 77, 7, 14, 88, 18, 38],
        [90, 84, 47, 34, 21, 51, 87, 98, 96, 29, 23, 15, 53, 100, 89, 17, 13, 20, 75, 85, 6, 74, 24, 92, 26, 40, 71, 30, 18, 64],
        [31, 7, 9, 95, 37, 6, 57, 50, 56, 80, 32, 12, 22, 52, 51, 53, 47, 68, 17, 77, 63, 79, 30, 82, 81, 48, 5, 85, 27, 4],
        [96, 14, 75, 15, 9, 98, 27, 7, 28, 86, 31, 16, 22, 12, 39, 97, 84, 87, 85, 52, 46, 18, 30, 59, 10, 68, 19, 100, 56, 93],
        [28, 95, 22, 67, 65, 11, 16, 58, 19, 89, 96, 23, 31, 75, 83, 5, 59, 10, 81, 86, 48, 13, 39, 29, 4, 14, 32, 51, 71, 73],
    ]
    tpr = 29

    num = test_num[tpr]
    # print(num)
    se_vehicle_capacity = 200
    if num >= 12 and num <= 22:
        se_vehicle_capacity = 1000
    if num >= 31:
        se_vehicle_capacity = 1000
    df = load(dataset[num])
    r = data_num_30[tpr-15]
    # print(r)
    
    instance_id = r
    for q in range(len(r)):
        json_instance.append([q+1 ,df['XCOORD.'][r[q]] , df['YCOORD.'][r[q]] , df['DEMAND'][r[q]] ])
    r.insert(0 , 0)
    for q in range(len(r)):
        json_instances.append([q ,df['XCOORD.'][r[q]] , df['YCOORD.'][r[q]] , df['DEMAND'][r[q]] ])
    return json_instance , se_vehicle_capacity , instance_id , json_instances


def initialize_population(n_customers, n_population):
    population = []
    while len(population) < n_population:
        chromosome = random.sample([i for i in range(1, n_customers+1)], n_customers)
        if chromosome not in population:
            population.append(chromosome)
    return population


def evaluate(chromosome, distance_matrix, demand, cap_vehicle, return_subroute=False):
    total_distance = 0
    cur_load = 0
    n_vehicle = 0
    route = []
    sub_route = []
    for customer in chromosome:
        cur_load += demand[customer]
        if cur_load > cap_vehicle:
            if return_subroute:
                sub_route.append(route[:])
            total_distance += calculate_distance(route, distance_matrix)
            n_vehicle += 1
            cur_load = demand[customer]
            route = [customer]
        else:
            route.append(customer)

    total_distance += calculate_distance(route, distance_matrix)
    n_vehicle += 1
    if return_subroute:
        sub_route.append(route[:])
        return sub_route
    return total_distance + n_vehicle


def calculate_distance(route, distance_matrix):
    distance = 0
    distance += distance_matrix[0][route[0]]
    distance += distance_matrix[route[-1]][0]
    for i in range(0, len(route)-1):
        distance += distance_matrix[route[i]][route[i+1]]
    return distance


def get_chromosome(population, func, *params, reverse=False, k=1):
    scores = []
    for chromosome in population:
        scores.append([func(chromosome, *params), chromosome])
    scores.sort(reverse=reverse)
    if k == 1:
        return scores[0]
    elif k > 1:
        return scores[:k]
    else:
        raise Exception("invalid k")


def ordered_crossover(chromo1, chromo2):
    # Modifying this to suit our needs
    #  If the sequence does not contain 0, this throws error
    #  So we will modify inputs here itself and then
    #       modify the outputs too

    ind1 = [x-1 for x in chromo1]
    ind2 = [x-1 for x in chromo2]
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    # Finally adding 1 again to reclaim original input
    ind1 = [x+1 for x in ind1]
    ind2 = [x+1 for x in ind2]
    return ind1, ind2


def mutate(chromosome, probability):
    if random.random() < probability:
        index1, index2 = random.sample(range(len(chromosome)), 2)
        chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
        index1, index2 = sorted(random.sample(range(len(chromosome)), 2))
        mutated = chromosome[:index1] + list(reversed(chromosome[index1:index2+1]))
        if index2 < len(chromosome) - 1:
            mutated += chromosome[index2+1:]
        return mutated
    return chromosome


def replace(population, chromo_in, chromo_out):
    population[population.index(chromo_out)] = chromo_in


def check_validity(chromosome, length):
    for i in range(1, length+1):
        if i not in chromosome:
            raise Exception("invalid chromosome")



if __name__ == '__main__':
    avg = []
    avg_time = []
    for i in range(10):
 
        start_time = time.time()
        distance_matrix = []
        distance_matrixs = []
        args = get_parser()
        json_instance , se_vehicle_capacity , instance_id , json_instances = load_instances()

        n_customers = len(json_instance)
        demand = {}
        demands = []
        for i in range(1 , n_customers+1):
            demand[i] = json_instance[i - 1][3]
        for i in range(n_customers+1):
            demands.append(json_instances[i][3])
        # print(demand)
        for i in range(len(json_instances)):
            for j in range(len(json_instances)):
                distance_matrixs.append(((json_instances[i][1] - json_instances[j][1])**2 + (json_instances[i][2] - json_instances[j][2])**2)**0.5)
            distance_matrix.append(distance_matrixs)
            distance_matrixs = []


        cap_vehicle = se_vehicle_capacity
        # print(cap_vehicle)
        depart = json_instances[0]

        n_population = args.pop_size
        iteration = args.iterations
        cur_iter = 1
        mutate_prob = args.mute_prob
        # print(demands)
        # cw = Vrp(n_customers , cap_vehicle ,demands , distance_matrix)
        # R = cw.start()
        population = initialize_population(n_customers, n_population)
        # population.append(R)
        prev_score, chromosome = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle)

        score_history = [prev_score]

        while cur_iter <= iteration:
            chromosomes = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle, k=2)
            chromosome1 = chromosomes[0][1]
            chromosome2 = chromosomes[1][1]
            offspring1, offspring2 = ordered_crossover(chromosome1, chromosome2)
            offspring1 = mutate(offspring1, mutate_prob)
            offspring2 = mutate(offspring2, mutate_prob)
            score1 = evaluate(offspring1, distance_matrix, demand, cap_vehicle)
            score2 = evaluate(offspring2, distance_matrix, demand, cap_vehicle)
            score, chromosome = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle, reverse=True)

            if score1 < score:
                replace(population, chromo_in=offspring1, chromo_out=chromosome)

            score, chromosome = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle, reverse=True)

            if score2 < score:
                replace(population, chromo_in=offspring2, chromo_out=chromosome)

            score, chromosome = get_chromosome(population, evaluate, distance_matrix, demand, cap_vehicle)
            score_history.append(score)
            prev_score = score
            cur_iter += 1
        avg.append(score)
        avg_time.append(time.time() - start_time)
        print(score, chromosome)
        print("總共花費時間：" ,time.time() -  start_time)
    
    print("總時間：" , sum(avg_time))
    print("最小分數"    , min(avg))
    print("平均分數：" , sum(avg) / len(avg))
