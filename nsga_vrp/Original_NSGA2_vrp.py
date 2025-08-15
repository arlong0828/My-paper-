# 標準函式庫
import random
import copy
import time

# 外部函式庫
import openpyxl
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from operator import itemgetter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point

# 距離計算函式庫
from haversine_python import haversine

# 自定義模組
from FE_gurobipy import FE_gurobi
# 匯入資料
random.seed(1)

# 使用kmeans分組客戶
def calculate_satellite_coordinates(instance , number_satellite):
    coordinates = []
    for i in range(len(instance)):
        coordinates.append([instance[i][1] , instance[i][2]])
    model = KMeans(n_clusters= number_satellite , n_init='auto')
    model_fit = model.fit(coordinates)
    customer_group = model.predict(coordinates)
    centers = model_fit.cluster_centers_
    return centers , customer_group

# 區分不同時期的顧客
def distinguish_between_periods(instance , periods = 720):
    morning_customer = []
    morning_customer_id = []
    afternoon_customer = []
    afternoon_customer_id = []
    for i in instance:
        if i[4] <= periods:
            morning_customer.append(i)
            morning_customer_id.append([i[0]])
        else:
            afternoon_customer.append(i)
            afternoon_customer_id.append([i[0]])
    return morning_customer , morning_customer_id , afternoon_customer , afternoon_customer_id

# 計算距離
def calculate_distance(latitude1, longitude1, latitude2, longitude2):
    H = haversine()
    return H.getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2)

def calculate_time(time_morning_vehicles , instace , distance , speed , cost):
    time_morning_vehicles += (distance / speed) * 60
    cost += (distance ) * 0.05 + 0.2 * max(instace[3] - time_morning_vehicles , 0) + 0.2 * max(time_morning_vehicles - instace[4], 0)
    return time_morning_vehicles , cost

def get_distance( morning_customer , afternoon_customer):
    morning_distances = []
    afternoon_distances = []
    for i in range(len(morning_customer)):
        distance = []
        for j in range(len(morning_customer)):
            if i == j:
                distance.append(0)
            else:
                if i == 0:
                    distance.append(calculate_distance(morning_customer[i][1] , morning_customer[i][0] , morning_customer[j][2] , morning_customer[j][1]))
                elif j == 0:
                    distance.append(calculate_distance(morning_customer[i][2] , morning_customer[i][1] , morning_customer[j][1] , morning_customer[j][0]))
                else:
                    distance.append(calculate_distance(morning_customer[i][2] , morning_customer[i][1] , morning_customer[j][2] , morning_customer[j][1]))
        morning_distances.append(distance)

    for i in range(len(afternoon_customer)):
        distance = []
        for j in range(len(afternoon_customer)):
            if i == j:
                distance.append(0)
            else:
                if i == 0:
                    distance.append(calculate_distance(afternoon_customer[i][1] , afternoon_customer[i][0] , afternoon_customer[j][2] , afternoon_customer[j][1]))
                elif j == 0:
                    distance.append(calculate_distance(afternoon_customer[i][2] , afternoon_customer[i][1] , afternoon_customer[j][1] , afternoon_customer[j][0]))
                else:
                    distance.append(calculate_distance(afternoon_customer[i][2] , afternoon_customer[i][1] , afternoon_customer[j][2] , afternoon_customer[j][1]))
        afternoon_distances.append(distance)
    return morning_distances , afternoon_distances

def savingsAlgorithms(morning_distances , afternoon_distances , morning_customer_id , afternoon_customer_id):
    morning_savings = []
    afternoon_savings = []
    for i in range(1, len(morning_customer_id) + 1):                                               
        for j in range(1, len(morning_customer_id) + 1):
            if i == j:
                pass
            else:
                morning_saving = (morning_distances[i][0] + morning_distances[0][j]) - morning_distances[i][j]
                morning_savings.append([morning_customer_id[i-1][0], morning_customer_id[j-1][0], morning_saving])                                          
    morning_savings = sorted(morning_savings, key=itemgetter(2), reverse=True)
    for i in range(1, len(afternoon_customer_id) + 1):                                                 
        for j in range(1, len(afternoon_customer_id) + 1):
            if i == j:
                pass
            else:
                afternoon_saving = (afternoon_distances[i][0] + afternoon_distances[0][j]) - afternoon_distances[i][j]
                afternoon_savings.append([afternoon_customer_id[i-1][0], afternoon_customer_id[j-1][0], afternoon_saving])                                      

    afternoon_savings = sorted(afternoon_savings, key=itemgetter(2), reverse=True)
    return morning_savings , afternoon_savings

def getRoute(instance , morning_savings , afternoon_savings , morning_customer_id , afternoon_customer_id , se_vehicle_capacity):
    for i in range(len(morning_savings)):
        startRoute = []
        endRoute = []
        routeDemand = 0
        for j in range(len(morning_customer_id)):
            if (morning_savings[i][0] == morning_customer_id[j][-1]):
                endRoute = morning_customer_id[j]
            elif (morning_savings[i][1] == morning_customer_id[j][0]):
                startRoute = morning_customer_id[j]
            if ((len(startRoute) != 0) and (len(endRoute) != 0)):
                for k in range(len(startRoute)):
                    for ins in instance:
                        if ins[0] == startRoute[k]:
                            routeDemand += ins[5] + ins[6]
                for k in range(len(endRoute)):
                    for ins in instance:
                        if ins[0] == endRoute[k]:
                            routeDemand += ins[5] + ins[6]
                if (routeDemand <= se_vehicle_capacity):
                    morning_customer_id.remove(endRoute)
                    morning_customer_id.remove(startRoute)
                    morning_customer_id.append(endRoute + startRoute)
                break

    for i in range(len(afternoon_savings)):
        startRoute = []
        endRoute = []
        routeDemand = 0
        for j in range(len(afternoon_customer_id)):
            if (afternoon_savings[i][0] == afternoon_customer_id[j][-1]):
                endRoute = afternoon_customer_id[j]
            elif (afternoon_savings[i][1] == afternoon_customer_id[j][0]):
                startRoute = afternoon_customer_id[j]
            if ((len(startRoute) != 0) and (len(endRoute) != 0)):
                for k in range(len(startRoute)):
                    for ins in instance:
                        if ins[0] == startRoute[k]:
                            routeDemand += ins[5] + ins[6]
                for k in range(len(endRoute)):
                    for ins in instance:
                        if ins[0] == endRoute[k]:
                            routeDemand += ins[5] + ins[6]
                if (routeDemand <= se_vehicle_capacity):
                    afternoon_customer_id.remove(endRoute)
                    afternoon_customer_id.remove(startRoute)
                    afternoon_customer_id.append(endRoute + startRoute)
                break
    return morning_customer_id , afternoon_customer_id

def eval_indvidual_fitness(se_vehicle_speed , morning_customer_id , afternoon_customer_id, instace , SatelliteCoordinates):
    number_morning_vehicles = 0
    morning_total_cost = 0
    number_afternoon_vehicles = 0
    afternoon_total_cost = 0
    n = 0
    for i in morning_customer_id:
        time_morning_vehicles = 540
        cost = 0
        for j in range(0 , len(i)):
            if j == 0:
                for z in range(len(instace)):
                    if instace[z][0] == i[j]:
                        time_morning_vehicles , cost = calculate_time(time_morning_vehicles , instace[z] , calculate_distance(SatelliteCoordinates[1] , SatelliteCoordinates[0] ,instace[z][2] , instace[z][1]) , se_vehicle_speed , cost)
            else:
                for z in range(len(instace)):
                    if instace[z][0] == i[j]:
                        n = z
                for z in range(len(instace)):
                    if instace[z][0] == i[j-1]:

                        time_morning_vehicles , cost = calculate_time(time_morning_vehicles , instace[n] , calculate_distance(instace[n][2] , instace[n][1] , instace[z][2] , instace[z][1]) , se_vehicle_speed , cost)
        for x in range(len(instace)):
            if instace[x][0] == i[-1]:
                cost += calculate_distance(instace[x][2] , instace[x][1] , SatelliteCoordinates[1] , SatelliteCoordinates[0])
        morning_total_cost += cost
        number_morning_vehicles += 1

    for i in afternoon_customer_id:
        time_afternoon_vehicles = 720
        cost = 0
        for j in range(0 , len(i)):
            if j == 0:
                for z in range(len(instace)):
                    if instace[z][0] == i[j]:
                        time_afternoon_vehicles , cost = calculate_time(time_afternoon_vehicles , instace[z] , calculate_distance(SatelliteCoordinates[1] , SatelliteCoordinates[0] ,instace[z][2] , instace[z][1]) , se_vehicle_speed , cost)
            else:
                for z in range(len(instace)):
                    if instace[z][0] == i[j]:
                        n = z
                for z in range(len(instace)):
                    if instace[z][0] == i[j-1]:
                        time_afternoon_vehicles , cost = calculate_time(time_afternoon_vehicles , instace[n] , calculate_distance(instace[n][2] , instace[n][1] , instace[z][2] , instace[z][1]) , se_vehicle_speed , cost)
        for x in range(len(instace)):
            if instace[x][0] == i[-1]:
                cost += calculate_distance(instace[x][2] , instace[x][1] , SatelliteCoordinates[1] , SatelliteCoordinates[0])
        afternoon_total_cost += cost
        number_afternoon_vehicles += 1
    return [number_morning_vehicles , round(morning_total_cost, 0)] , [number_afternoon_vehicles , round(afternoon_total_cost , 0)]


# def single_exchange(ind1 , cross_prob):
#     if random.random() <= cross_prob:
#         ind2 = copy.deepcopy(ind1)
#         a , b= random.sample(range(len(ind2)), 2)
#         ind2[b] , ind2[a] = ind2[a] , ind2[b]
#         return ind2
#     return ind1

def cxOrderedVrp(input_ind1, input_ind2):
    # Modifying this to suit our needs
    #  If the sequence does not contain 0, this throws error
    #  So we will modify inputs here itself and then 
    #       modify the outputs too
    if len(input_ind1) == 0 or len(input_ind2) == 0:
        return input_ind1, input_ind2
    
    ind1 = copy.deepcopy(input_ind1)
    ind2 = copy.deepcopy(input_ind2)
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    # print("size：" , size)
    # print(a, b)
    if a > b:
        a, b = b, a
    # print(a , b)

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2.index(ind2[i])] = False
            holes2[ind1.index(ind1[i])] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[input_ind2.index(temp1[(i + b + 1) % size])]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[input_ind1.index(temp2[(i + b + 1) % size])]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1
    # print(ind1 , ind2)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def mutation(ind1 , mut_prob):
    if len(ind1) > 1 and random.random() <= mut_prob:
        ind2 = copy.deepcopy(ind1)
        a , b= sorted(random.sample(range(len(ind2)), 2))
        mutated = ind2[:a] + list(reversed(ind2[a:b + 1]))
        if b < len(ind2) - 1:
            mutated += ind2[b + 1 :]
        return mutated
    return ind1

def exchange(ind1, se_vehicle_capacity, instace):

    update_ind1 = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instace)

    # 防止空路徑列表
    if not route or len(route) == 0:
        return ind1

    # 隨機選擇非空且長度大於 1 的子路徑
    valid_routes = [i for i in range(len(route)) if len(route[i]) > 1]
    if not valid_routes:
        return ind1

    r = random.choice(valid_routes)  # 安全選擇可用的路徑
    a, b = random.sample(range(len(route[r])), 2)
    route[r][a], route[r][b] = route[r][b], route[r][a]

    for i in route:
        update_ind1 += i
    return update_ind1

def opt2(ind1, se_vehicle_capacity, instance):
    update_ind1 = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instance)

    if len(route) == 0:
        return ind  # 若路徑為空，回傳原始個體

    r = random.randrange(0, len(route), 1)

    if len(route[r]) > 1:
        a, b = sorted(random.sample(range(len(route[r])), 2))
        re = route[r][:a] + list(reversed(route[r][a:b + 1]))
        if b < len(route[r]) - 1:
            re += route[r][b + 1:]
    else:
        re = route[r]

    for i in range(len(route)):
        if i == r:
            update_ind1 += re
        else:
            update_ind1 += route[i]

    return update_ind1

def relocate(ind1, se_vehicle_capacity, instance):
    update_ind1 = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instance)

    # 防呆：route 為空時，直接回傳原始個體
    if len(route) == 0:
        # print("⚠️ relocate：route 為空，無法操作，回傳原個體")
        return ind

    r = random.randrange(0, len(route))

    # 如果該路徑有多於1個節點才進行 relocate
    if len(route[r]) > 1:
        a, b = random.sample(range(len(route[r])), 2)
        # 建立新路徑：把第 b 個節點移到第 a 個位置
        temp_route = [route[r][i] for i in range(len(route[r])) if i != b]
        temp_route.insert(a, route[r][b])
    else:
        temp_route = route[r]  # 單節點的路徑不進行更動

    # 組裝新個體
    for i in range(len(route)):
        if i == r:
            update_ind1 += temp_route
        else:
            update_ind1 += route[i]

    return update_ind1


def update_route(ind1 , se_vehicle_capacity , instace):
    capacity = 0
    total_route = []
    route = []
    for i in ind1:
        for j in instace:
            if j[0] == i:
                capacity += j[5] + j[6]
                if capacity >= se_vehicle_capacity:
                    total_route.append(route)
                    capacity = 0
                    route = [i]
                else:
                    route.append(i)
    if route != []:
        total_route.append(route)
    return total_route

def satellite_calculate_fitness(ind1 , se_vehicle_speed , se_vehicle_capacity , instace, time_vehicles , satellite):
    # print(instace)
    num_vehicle = 0
    route = update_route(ind1 , se_vehicle_capacity , instace)
    num_vehicle = len(route)
    n = 0
    total_cost = 0
    for i in route:
        cost = 0
        time_vehicles2 = time_vehicles
        for j in range(len(i)):
            if j == 0:
                for ins in instace:
                    if ins[0] == i[j]:
                        # print(time_vehicles2 , ins , cost)
                        # print(satellite[1] , satellite[0] , ins[2] , ins[1] , calculate_distance(satellite[1] , satellite[0] , ins[2] , ins[1]))
                        time_vehicles2 , cost = calculate_time(time_vehicles2 , ins , calculate_distance(satellite[1] , satellite[0] , ins[2] , ins[1]) , se_vehicle_speed , cost)
                        # print(time_vehicles2 , cost)
            else:
                for z in range(len(instace)):
                    if instace[z][0] == i[j]:
                        n = z
                for z in range(len(instace)):
                    if instace[z][0] == i[j - 1]:
                        time_vehicles2 , cost = calculate_time(time_vehicles2 , instace[n] , calculate_distance(instace[z][2] , instace[z][1] , instace[n][2] , instace[n][1]) , se_vehicle_speed , cost)
        for x in range(len(instace)):
            if instace[x][0] == i[-1]:
                cost += calculate_distance(instace[x][2] , instace[x][1] , satellite[1] , satellite[0])
        total_cost += cost
    return [num_vehicle , round(total_cost , 0 )]

def find_smae(all_ind , ind):
    for i in all_ind:
        if i[0] == ind:
            return False
    return True

def non_dominated_sorting(chroms_obj_record):
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    for p in range(len(chroms_obj_record)):
        s[p]=[]
        n[p]=0
        for q in range(len(chroms_obj_record)):
            
            if ((chroms_obj_record[p][1][0]<chroms_obj_record[q][1][0] and chroms_obj_record[p][1][1]<chroms_obj_record[q][1][1]) or (chroms_obj_record[p][1][0]<=chroms_obj_record[q][1][0] and chroms_obj_record[p][1][1]<chroms_obj_record[q][1][1])
            or (chroms_obj_record[p][1][0]<chroms_obj_record[q][1][0] and chroms_obj_record[p][1][1]<=chroms_obj_record[q][1][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((chroms_obj_record[p][1][0]>chroms_obj_record[q][1][0] and chroms_obj_record[p][1][1]>chroms_obj_record[q][1][1]) or (chroms_obj_record[p][1][0]>=chroms_obj_record[q][1][0] and chroms_obj_record[p][1][1]>chroms_obj_record[q][1][1])
            or (chroms_obj_record[p][1][0]>chroms_obj_record[q][1][0] and chroms_obj_record[p][1][1]>=chroms_obj_record[q][1][1])):
                n[p]=n[p]+1
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    i=0
    while (front[i]!=[]):
        Q=[]
        for p in front[i]:
            for q in s[p]:
                n[q]=n[q]-1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i=i+1
        front[i]=Q
                
    del front[len(front)-1]
    return front

def calculate_crowding_distance(front,chroms_obj_record):
    
    distance={m:0 for m in front}
    for o in range(2):
        obj={m:chroms_obj_record[m][1][o] for m in front}
        sorted_keys=sorted(obj, key=obj.get)
        distance[sorted_keys[0]]=distance[sorted_keys[len(front)-1]]=999999999999
        for i in range(1,len(front)-1):
            if len(set(obj.values()))==1:
                distance[sorted_keys[i]]=distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]]=distance[sorted_keys[i]]+(obj[sorted_keys[i+1]]-obj[sorted_keys[i-1]])/(obj[sorted_keys[len(front)-1]]-obj[sorted_keys[0]])
            
    return distance     

def selection(population_size,front,chroms_obj_record,best = False):
    if best:
        B = []
        for i in front:
            B.append(chroms_obj_record[i])
        return B
    N=0
    new_pop=[]
    while N < population_size:
        for i in range(len(front)):
            N=N+len(front[i])
            if N > population_size:
                distance=calculate_crowding_distance(front[i],chroms_obj_record)
                sorted_cdf=sorted(distance, key=distance.get)
                sorted_cdf.reverse()
                for j in sorted_cdf:
                    if len(new_pop)==population_size:
                        break                
                    new_pop.append(j)              
                break
            else:
                new_pop.extend(front[i])
    
    population_list=[]
    for n in new_pop:
        population_list.append(chroms_obj_record[n])
    
    return population_list,new_pop

def calculate_satellite_demand(instance , customer_group):
    satellite_demand = 0
    for j in range(len(customer_group)):
        for i in range(len(instance)):
            if instance[i][0] == customer_group[j]:
                if instance[i][5] != None and instance[i][6] != None:
                    satellite_demand += instance[i][5] + instance[i][6]
    return satellite_demand

class NSGAAlgorithm(object):


    def __init__(self):
        self.start_customer_number = 0
        self.end_customer_number = 99
        self.json_instance = ""
        self.crossover_probability = 0.85
        self.mut_prob = 0.1
        self.num_gen = 1000
        self.pop_size = 20
        self.best_pop = 1
        self.fe_vehicle_capacity = 4000
        self.se_vehicle_capacity = 400
        self.se_vehicle_speed = 50
        self.depots_id = [["d1" , 120.373731 , 36.185609] , ["d2" , 118.054927 , 36.813487] , ["d3" , 116.897877 , 36.611274]]
        self.depot = [[120.373731 , 36.185609] , [ 118.054927 , 36.813487] , [116.897877 , 36.611274]]
        self.number_satellite = 2
        self.centers , self.customer_group = "" , ""
        self.all_morning_customer_id_and_fitness1 = []
        self.all_afternoon_customer_id_and_fitness1 = []
        self.all_morning_customer_id_and_fitness2 = []
        self.all_afternoon_customer_id_and_fitness2 = []
        self.all_morning_customer_id_and_fitness3 = []
        self.all_afternoon_customer_id_and_fitness3 = []
        self.all_morning_customer_id_and_fitness4 = []
        self.all_afternoon_customer_id_and_fitness4 = []
        self.start_time = time.time()
        self.day = 0
        self.all_number_vehicles = 0
        self.all_time = 0
        self.FE_all_cost = 0
        self.SE_all_cost = 0
        self.satellite_demand = []

    def load_instance(self):
        H = haversine()
        wb = openpyxl.load_workbook("./data/real_data.xlsx")
        s1 = wb["Sheet1"]
        arr = []
        for row in s1:
            arr2 = []
            for col in row:
                if col.value != None:
                    arr2.append(col.value)
            arr.append(arr2)
                    
        del arr[0]
        self.json_instance = arr[self.start_customer_number:self.end_customer_number]
        for customer in self.json_instance:
            customer_lon = customer[1]
            customer_lat = customer[2]
            min_dist = float("inf")
            assigned_depot = None
            for depot in self.depots_id:
                depot_lon = depot[1]
                depot_lat = depot[2]
                dist = H.getDistanceBetweenPointsNew(customer_lat , customer_lon, depot_lat , depot_lon )
                if dist < min_dist:
                    min_dist = dist
                    assigned_depot = depot[0]  # 倉庫名稱，例如 "d1"
            customer.append(assigned_depot)  # 加入最近倉庫名稱到資料末尾

    def initial_solution(self):
        for i in range(len(self.depot)):
            group_ = []
            all_satellite_morning_id = []
            all_satellite_afternoon_id = []
            for j in range(len(self.json_instance)):
                if self.depots_id[i][0] == self.json_instance[j][8]:
                    group_.append(self.json_instance[j])
            morning_customer , morning_customer_id , afternoon_customer , afternoon_customer_id= distinguish_between_periods(group_)

            for z in morning_customer_id:
                all_satellite_morning_id += z
            for z in afternoon_customer_id:
                all_satellite_afternoon_id += z
            if i == 0:
                # for _ in range(self.pop_size):
                if len(all_satellite_morning_id) >= 4:
                    while(len(self.all_morning_customer_id_and_fitness1) < self.pop_size):
                        # print(all_satellite_morning_id)
                        new = random.sample(all_satellite_morning_id, k=len(all_satellite_morning_id))
                        # print(new)
                        if find_smae(self.all_morning_customer_id_and_fitness1 , new):
                            fitness = satellite_calculate_fitness(new , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[i])
                            self.all_morning_customer_id_and_fitness1.append([new , fitness])
                elif len(all_satellite_morning_id) < 4:
                    fitness = satellite_calculate_fitness(all_satellite_morning_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[i])
                    self.all_morning_customer_id_and_fitness1.append([all_satellite_morning_id , fitness])
                if len(all_satellite_afternoon_id) >= 4:
                    while(len(self.all_afternoon_customer_id_and_fitness1) < self.pop_size):
                        new2 = random.sample(all_satellite_afternoon_id, k=len(all_satellite_afternoon_id))
                        if find_smae(self.all_afternoon_customer_id_and_fitness1 , new2):
                            fitness2 = satellite_calculate_fitness(new2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[i])
                            self.all_afternoon_customer_id_and_fitness1.append([new2 , fitness2])
                elif len(all_satellite_afternoon_id) < 4:
                    fitness2 = satellite_calculate_fitness(all_satellite_afternoon_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[i])
                    self.all_afternoon_customer_id_and_fitness1.append([all_satellite_afternoon_id , fitness2])
            elif i == 1:
                if len(all_satellite_morning_id) >= 4:
                    while(len(self.all_morning_customer_id_and_fitness2) < self.pop_size):
                        # print(all_satellite_morning_id)
                        new = random.sample(all_satellite_morning_id, k=len(all_satellite_morning_id))
                        if find_smae(self.all_morning_customer_id_and_fitness2 , new):
                            fitness = satellite_calculate_fitness(new , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[i])
                            self.all_morning_customer_id_and_fitness2.append([new , fitness])
                elif len(all_satellite_morning_id) < 4:
                    fitness = satellite_calculate_fitness(all_satellite_morning_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[i])
                    self.all_morning_customer_id_and_fitness2.append([all_satellite_morning_id , fitness])
                if len(all_satellite_afternoon_id) >= 4:
                    while(len(self.all_afternoon_customer_id_and_fitness2) < self.pop_size):
                        new2 = random.sample(all_satellite_afternoon_id, k=len(all_satellite_afternoon_id))
                        if find_smae(self.all_afternoon_customer_id_and_fitness2 , new2):
                            fitness2 = satellite_calculate_fitness(new2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[i])
                            self.all_afternoon_customer_id_and_fitness2.append([new2 , fitness2])
                elif len(all_satellite_afternoon_id) < 4:
                    fitness2 = satellite_calculate_fitness(all_satellite_afternoon_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[i])
                    self.all_afternoon_customer_id_and_fitness2.append([all_satellite_afternoon_id , fitness2])
            elif i == 2:
                if len(all_satellite_morning_id) >= 4:
                    while(len(self.all_morning_customer_id_and_fitness3) < self.pop_size):
                        # print(all_satellite_morning_id)
                        new = random.sample(all_satellite_morning_id, k=len(all_satellite_morning_id))
                        if find_smae(self.all_morning_customer_id_and_fitness3 , new):
                            fitness = satellite_calculate_fitness(new , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[i])
                            self.all_morning_customer_id_and_fitness3.append([new , fitness])
                elif len(all_satellite_morning_id) < 4: 
                    fitness = satellite_calculate_fitness(all_satellite_morning_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[i])
                    self.all_morning_customer_id_and_fitness3.append([all_satellite_morning_id , fitness])
                if len(all_satellite_afternoon_id) >= 4:
                    while(len(self.all_afternoon_customer_id_and_fitness3) < self.pop_size):
                        new2 = random.sample(all_satellite_afternoon_id, k=len(all_satellite_afternoon_id))
                        if find_smae(self.all_afternoon_customer_id_and_fitness3 , new2):
                            fitness2 = satellite_calculate_fitness(new2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[i])
                            self.all_afternoon_customer_id_and_fitness3.append([new2 , fitness2])
                elif len(all_satellite_afternoon_id) < 4:
                    fitness2 = satellite_calculate_fitness(all_satellite_afternoon_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[i])
                    self.all_afternoon_customer_id_and_fitness3.append([all_satellite_afternoon_id , fitness2])



    def initial_population(self):
        if len(self.all_morning_customer_id_and_fitness1[0][0]) > 5:
            while(len(self.all_morning_customer_id_and_fitness1) < 20):
                for i in range(len(self.all_morning_customer_id_and_fitness1)):

                    exchange_ind = exchange(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness1 , exchange_ind):
                        fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                        self.all_morning_customer_id_and_fitness1.append([exchange_ind , fitmess2])

                    opt2_ind = opt2(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness1 , opt2_ind):
                        fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                        self.all_morning_customer_id_and_fitness1.append([opt2_ind , fitmess2])
    
                    relocate_ind = relocate(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness1 , relocate_ind):
                        fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                        self.all_morning_customer_id_and_fitness1.append([relocate_ind , fitmess2])

        if self.all_morning_customer_id_and_fitness2:
            if len(self.all_morning_customer_id_and_fitness2[0][0]) > 5:
                while(len(self.all_morning_customer_id_and_fitness2) < 20):
                    for i in range(len(self.all_morning_customer_id_and_fitness2)):

                        exchange_ind = exchange(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_morning_customer_id_and_fitness2 , exchange_ind):
                            fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                            self.all_morning_customer_id_and_fitness2.append([exchange_ind , fitmess2])

                        opt2_ind = opt2(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_morning_customer_id_and_fitness2 , opt2_ind):
                            fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                            self.all_morning_customer_id_and_fitness2.append([opt2_ind , fitmess2])
        
                        relocate_ind = relocate(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_morning_customer_id_and_fitness2 , relocate_ind):
                            fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                            self.all_morning_customer_id_and_fitness2.append([relocate_ind , fitmess2])
        if self.all_morning_customer_id_and_fitness3:
            if len(self.all_morning_customer_id_and_fitness3[0][0]) > 5:
                while(len(self.all_morning_customer_id_and_fitness3) < 20):
                    for i in range(len(self.all_morning_customer_id_and_fitness3)):

                        exchange_ind = exchange(self.all_morning_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_morning_customer_id_and_fitness3 , exchange_ind):
                            fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                            self.all_morning_customer_id_and_fitness3.append([exchange_ind , fitmess2])

                        opt2_ind = opt2(self.all_morning_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_morning_customer_id_and_fitness3 , opt2_ind):
                            fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                            self.all_morning_customer_id_and_fitness3.append([opt2_ind , fitmess2])
        
                        relocate_ind = relocate(self.all_morning_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_morning_customer_id_and_fitness3 , relocate_ind):
                            fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                            self.all_morning_customer_id_and_fitness3.append([relocate_ind , fitmess2])
     

        if len(self.all_afternoon_customer_id_and_fitness1[0][0]) > 5:
            while(len(self.all_afternoon_customer_id_and_fitness1) < 20):
                for i in range(len(self.all_afternoon_customer_id_and_fitness1)):

                    exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness1 , exchange_ind):
                        fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
                        self.all_afternoon_customer_id_and_fitness1.append([exchange_ind , fitmess2])

                    opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness1 , opt2_ind):
                        fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
                        self.all_afternoon_customer_id_and_fitness1.append([opt2_ind , fitmess2])
    
                    relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness1 , relocate_ind):
                        fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
                        self.all_afternoon_customer_id_and_fitness1.append([relocate_ind , fitmess2])
        if self.all_afternoon_customer_id_and_fitness2:
            if len(self.all_afternoon_customer_id_and_fitness2[0][0]) > 5:
                while(len(self.all_afternoon_customer_id_and_fitness2) < 20):
                    for i in range(len(self.all_afternoon_customer_id_and_fitness2)):

                        exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_afternoon_customer_id_and_fitness2 , exchange_ind):
                            fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                            self.all_afternoon_customer_id_and_fitness2.append([exchange_ind , fitmess2])

                        opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_afternoon_customer_id_and_fitness2 , opt2_ind):
                            fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                            self.all_afternoon_customer_id_and_fitness2.append([opt2_ind , fitmess2])
        
                        relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_afternoon_customer_id_and_fitness2 , relocate_ind):
                            fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                            self.all_afternoon_customer_id_and_fitness2.append([relocate_ind , fitmess2])
        if self.all_afternoon_customer_id_and_fitness3:
            if len(self.all_afternoon_customer_id_and_fitness3[0][0]) > 5:
                while(len(self.all_afternoon_customer_id_and_fitness3) < 20):
                    for i in range(len(self.all_afternoon_customer_id_and_fitness3)):

                        exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_afternoon_customer_id_and_fitness3 , exchange_ind):
                            fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                            self.all_afternoon_customer_id_and_fitness3.append([exchange_ind , fitmess2])

                        opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_afternoon_customer_id_and_fitness3 , opt2_ind):
                            fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                            self.all_afternoon_customer_id_and_fitness3.append([opt2_ind , fitmess2])
        
                        relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                        if find_smae(self.all_afternoon_customer_id_and_fitness3 , relocate_ind):
                            fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                            self.all_afternoon_customer_id_and_fitness3.append([relocate_ind , fitmess2])

    def runGenerations(self):
        for gen in range(self.num_gen):
            # print(f"{20*'#'} Currently Evaluating {gen} Generation {20*'#'}")
            # 上午
            # local search 
            for i in range(len(self.all_morning_customer_id_and_fitness1)):
                fitness1 = satellite_calculate_fitness(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                
                exchange_ind = exchange(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness1 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                    if fitness1[1] > fitmess2[1]:
                        self.all_morning_customer_id_and_fitness1.append([exchange_ind , fitmess2])
                        del self.all_morning_customer_id_and_fitness1[i]
                        break

                opt2_ind = opt2(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness1 , opt2_ind):
                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                    if fitness1[1] > fitmess3[1]:
                        self.all_morning_customer_id_and_fitness1.append([opt2_ind , fitmess3])
                        del self.all_morning_customer_id_and_fitness1[i]
                        break

                relocate_ind = relocate(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness1 , relocate_ind):
                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                    if fitness1[1] > fitmess4[1]:
                        self.all_morning_customer_id_and_fitness1.append([relocate_ind , fitmess4])
                        del self.all_morning_customer_id_and_fitness1[i]
                        break

            for i in range(len(self.all_morning_customer_id_and_fitness2)):
                fitness1 = satellite_calculate_fitness(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                
                exchange_ind = exchange(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness2 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                    if fitness1[1] > fitmess2[1]:
                        self.all_morning_customer_id_and_fitness2.append([exchange_ind , fitmess2])
                        del self.all_morning_customer_id_and_fitness2[i]
                        break
                
                opt2_ind = opt2(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness2 , opt2_ind):

                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                    if fitness1[1] > fitmess3[1]:
                        self.all_morning_customer_id_and_fitness2.append([opt2_ind , fitmess3])
                        del self.all_morning_customer_id_and_fitness2[i]
                        break

                relocate_ind = relocate(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness2 , relocate_ind):

                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                    if fitness1[1] > fitmess4[1]:
                        self.all_morning_customer_id_and_fitness2.append([relocate_ind , fitmess4])
                        del self.all_morning_customer_id_and_fitness2[i]
                        break

            for i in range(len(self.all_morning_customer_id_and_fitness3)):
                fitness1 = satellite_calculate_fitness(self.all_morning_customer_id_and_fitness3[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                
                exchange_ind = exchange(self.all_morning_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness3 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                    if fitness1[1] > fitmess2[1]:
                        self.all_morning_customer_id_and_fitness3.append([exchange_ind , fitmess2])
                        del self.all_morning_customer_id_and_fitness3[i]
                        break
                
                opt2_ind = opt2(self.all_morning_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness3 , opt2_ind):
                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                    if fitness1[1] > fitmess3[1]:
                        self.all_morning_customer_id_and_fitness3.append([opt2_ind , fitmess3])
                        del self.all_morning_customer_id_and_fitness3[i]
                        break

                relocate_ind = relocate(self.all_morning_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness3 , relocate_ind):
                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                    if fitness1[1] > fitmess4[1]:
                        self.all_morning_customer_id_and_fitness3.append([relocate_ind , fitmess4])
                        del self.all_morning_customer_id_and_fitness3[i]
                        break
          
            # 下午
            # local search
            for i in range(len(self.all_afternoon_customer_id_and_fitness1)):
                fitness1 = satellite_calculate_fitness(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
            
                exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness1 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
                    if fitness1[1] > fitmess2[1]:
                        self.all_afternoon_customer_id_and_fitness1.append([exchange_ind , fitmess2])
                        del self.all_afternoon_customer_id_and_fitness1[i]
                        break
                
                opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness1 , opt2_ind):
                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
                    if fitness1[1] > fitmess3[1]:
                        self.all_afternoon_customer_id_and_fitness1.append([opt2_ind , fitmess3])
                        del self.all_afternoon_customer_id_and_fitness1[i]
                        break
                
                relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness1 , relocate_ind):
                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
                    if fitness1[1] > fitmess4[1]:
                        self.all_afternoon_customer_id_and_fitness1.append([relocate_ind , fitmess4])
                        del self.all_afternoon_customer_id_and_fitness1[i]
                        break

            for i in range(len(self.all_afternoon_customer_id_and_fitness2)):

                fitness1 = satellite_calculate_fitness(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness2 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                    if fitness1[1] > fitmess2[1]:
                        self.all_afternoon_customer_id_and_fitness2.append([exchange_ind , fitmess2])
                        del self.all_afternoon_customer_id_and_fitness2[i]
                        break
                opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness2 , opt2_ind):
                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                    if fitness1[1] > fitmess3[1]:
                        self.all_afternoon_customer_id_and_fitness2.append([opt2_ind , fitmess3])
                        del self.all_afternoon_customer_id_and_fitness2[i]
                        break
                relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness2 , relocate_ind):
                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                    if fitness1[1] > fitmess4[1]:
                        self.all_afternoon_customer_id_and_fitness2.append([relocate_ind , fitmess4])
                        del self.all_afternoon_customer_id_and_fitness2[i]
                        break
            
            for i in range(len(self.all_afternoon_customer_id_and_fitness3)):
                fitness1 = satellite_calculate_fitness(self.all_afternoon_customer_id_and_fitness3[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                
                exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness3 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                    if fitness1[1] > fitmess2[1]:
                        self.all_afternoon_customer_id_and_fitness3.append([exchange_ind , fitmess2])
                        del self.all_afternoon_customer_id_and_fitness3[i]
                        break
                
                opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness3 , opt2_ind):
                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                    if fitness1[1] > fitmess3[1]:
                        self.all_afternoon_customer_id_and_fitness3.append([opt2_ind , fitmess3])
                        del self.all_afternoon_customer_id_and_fitness3[i]
                        break
                
                relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness3[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness3 , relocate_ind):
                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                    if fitness1[1] > fitmess4[1]:
                        self.all_afternoon_customer_id_and_fitness3.append([relocate_ind , fitmess4])
                        del self.all_afternoon_customer_id_and_fitness3[i]
                        break
            # 交配
            if len(self.all_morning_customer_id_and_fitness1[0][0]) > 5 and self.crossover_probability >= random.random():
                while(len(self.all_morning_customer_id_and_fitness1) < 40):
                    for i in range(0 , len(self.all_morning_customer_id_and_fitness1) , 2):
                        if i + 1 < len(self.all_morning_customer_id_and_fitness1):
                            ind1, ind2  = cxOrderedVrp(self.all_morning_customer_id_and_fitness1[i][0] , self.all_morning_customer_id_and_fitness1[i + 1][0])
                            if find_smae(self.all_morning_customer_id_and_fitness1 , ind1):
                                fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                                self.all_morning_customer_id_and_fitness1.append([ind1 , fitness1])
                            if find_smae(self.all_morning_customer_id_and_fitness1 , ind2):
                                fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                                self.all_morning_customer_id_and_fitness1.append([ind2 , fitness2])
            if self.all_morning_customer_id_and_fitness2:
                if len(self.all_morning_customer_id_and_fitness2[0][0]) > 5 and self.crossover_probability >= random.random():
                    while(len(self.all_morning_customer_id_and_fitness2) < 40):
                        for i in range(0 , len(self.all_morning_customer_id_and_fitness2) , 2):
                            if i + 1 < len(self.all_morning_customer_id_and_fitness2):
                                ind1, ind2  = cxOrderedVrp(self.all_morning_customer_id_and_fitness2[i][0] , self.all_morning_customer_id_and_fitness2[i + 1][0])
                                if find_smae(self.all_morning_customer_id_and_fitness2 , ind1):
                                    fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                                    self.all_morning_customer_id_and_fitness2.append([ind1 , fitness1])
                                if find_smae(self.all_morning_customer_id_and_fitness2 , ind2):
                                    fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                                    self.all_morning_customer_id_and_fitness2.append([ind2 , fitness2])
            if self.all_morning_customer_id_and_fitness3:
                if len(self.all_morning_customer_id_and_fitness3[0][0]) > 5 and self.crossover_probability >= random.random():
                    while(len(self.all_morning_customer_id_and_fitness3) < 40):
                        for i in range(0 , len(self.all_morning_customer_id_and_fitness3) , 2):
                            if i + 1 < len(self.all_morning_customer_id_and_fitness3):
                                ind1, ind2  = cxOrderedVrp(self.all_morning_customer_id_and_fitness3[i][0] , self.all_morning_customer_id_and_fitness3[i + 1][0])
                                if find_smae(self.all_morning_customer_id_and_fitness3 , ind1):
                                    fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                                    self.all_morning_customer_id_and_fitness3.append([ind1 , fitness1])
                                if find_smae(self.all_morning_customer_id_and_fitness3 , ind2):
                                    fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])
                                    self.all_morning_customer_id_and_fitness3.append([ind2 , fitness2])


            if len(self.all_afternoon_customer_id_and_fitness1[0][0]) > 5 and self.crossover_probability >= random.random():
                while(len(self.all_afternoon_customer_id_and_fitness1) < 40):
                    for i in range(0 , len(self.all_afternoon_customer_id_and_fitness1) , 2):
                        if i + 1 < len(self.all_afternoon_customer_id_and_fitness1):
                        
                            ind1, ind2  = cxOrderedVrp(self.all_afternoon_customer_id_and_fitness1[i][0] , self.all_afternoon_customer_id_and_fitness1[i + 1][0])
                            if find_smae(self.all_afternoon_customer_id_and_fitness1 , ind1):
                                fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
                                self.all_afternoon_customer_id_and_fitness1.append([ind1 , fitness1])
                            if find_smae(self.all_afternoon_customer_id_and_fitness1 , ind2):
                                fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])
                                self.all_afternoon_customer_id_and_fitness1.append([ind2 , fitness2])
            if self.all_afternoon_customer_id_and_fitness2:
                if len(self.all_afternoon_customer_id_and_fitness2[0][0]) > 5 and self.crossover_probability >= random.random():
                    while(len(self.all_afternoon_customer_id_and_fitness2) < 40):
                        for i in range(0 , len(self.all_afternoon_customer_id_and_fitness2) , 2):
                            if i + 1 < len(self.all_afternoon_customer_id_and_fitness2):
                                ind1, ind2  = cxOrderedVrp(self.all_afternoon_customer_id_and_fitness2[i][0] , self.all_afternoon_customer_id_and_fitness2[i + 1][0])
                                if find_smae(self.all_afternoon_customer_id_and_fitness2 , ind1):
                                    fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                                    self.all_afternoon_customer_id_and_fitness2.append([ind1 , fitness1])
                                if find_smae(self.all_afternoon_customer_id_and_fitness2 , ind2):
                                    fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])
                                    self.all_afternoon_customer_id_and_fitness2.append([ind2 , fitness2])
            if self.all_afternoon_customer_id_and_fitness3:
                if len(self.all_afternoon_customer_id_and_fitness3[0][0]) > 5 and self.crossover_probability >= random.random():
                    while(len(self.all_afternoon_customer_id_and_fitness3) < 40):
                        for i in range(0 , len(self.all_afternoon_customer_id_and_fitness3) , 2):
                            if i + 1 < len(self.all_afternoon_customer_id_and_fitness3):
                                ind1, ind2  = cxOrderedVrp(self.all_afternoon_customer_id_and_fitness3[i][0] , self.all_afternoon_customer_id_and_fitness3[i + 1][0])
                                if find_smae(self.all_afternoon_customer_id_and_fitness3 , ind1):
                                    fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                                    self.all_afternoon_customer_id_and_fitness3.append([ind1 , fitness1])
                                if find_smae(self.all_afternoon_customer_id_and_fitness3 , ind2):
                                    fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])
                                    self.all_afternoon_customer_id_and_fitness3.append([ind2 , fitness2])


            # 突變
            if len(self.all_morning_customer_id_and_fitness1[0][0]) > 4:
                for i in self.all_morning_customer_id_and_fitness1:
                    ind2 = mutation(i[0] , self.mut_prob)
                    if find_smae(self.all_morning_customer_id_and_fitness1 , ind2):
                        fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[0])
                        self.all_morning_customer_id_and_fitness1.append([ind2 , fitness])
                        break
            if self.all_morning_customer_id_and_fitness2:
                if len(self.all_morning_customer_id_and_fitness2[0][0]) > 4:
                    for i in self.all_morning_customer_id_and_fitness2:
                        ind2 = mutation(i[0] , self.mut_prob)
                        if find_smae(self.all_morning_customer_id_and_fitness2 , ind2):
                            fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[1])
                            self.all_morning_customer_id_and_fitness2.append([ind2 , fitness]) 
                            break
            if self.all_morning_customer_id_and_fitness3:
                if len(self.all_morning_customer_id_and_fitness3[0][0]) > 4:
                    for i in self.all_morning_customer_id_and_fitness3:
                        ind2 = mutation(i[0] , self.mut_prob)
                        if find_smae(self.all_morning_customer_id_and_fitness3 , ind2):
                            fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.depot[2])  
                            self.all_morning_customer_id_and_fitness3.append([ind2 , fitness]) 
                            break

            if len(self.all_afternoon_customer_id_and_fitness1[0][0]) > 4:
                for i in self.all_afternoon_customer_id_and_fitness1:
                    ind2 = mutation(i[0] , self.mut_prob)
                    if find_smae(self.all_afternoon_customer_id_and_fitness1 , ind2):
                        fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[0])  
                        self.all_afternoon_customer_id_and_fitness1.append([ind2 , fitness]) 
                        break
            if self.all_afternoon_customer_id_and_fitness2:
                if len(self.all_afternoon_customer_id_and_fitness2[0][0]) > 4:
                    for i in self.all_afternoon_customer_id_and_fitness2:
                        ind2 = mutation(i[0] , self.mut_prob)
                        if find_smae(self.all_afternoon_customer_id_and_fitness2 , ind2):
                            fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[1])         
                            self.all_afternoon_customer_id_and_fitness2.append([ind2 , fitness])  
                            break
            if self.all_afternoon_customer_id_and_fitness3:
                if len(self.all_afternoon_customer_id_and_fitness3[0][0]) > 4:
                    for i in self.all_afternoon_customer_id_and_fitness3:
                        ind2 = mutation(i[0] , self.mut_prob)
                        if find_smae(self.all_afternoon_customer_id_and_fitness3 , ind2):
                            fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.depot[2])  
                            self.all_afternoon_customer_id_and_fitness3.append([ind2 , fitness]) 
                            break


            # non-dominated sorting
            front = non_dominated_sorting(self.all_morning_customer_id_and_fitness1)
            # selection
            population_list,new_pop=selection(self.pop_size,front,self.all_morning_customer_id_and_fitness1)
            self.all_morning_customer_id_and_fitness1 = population_list

            if self.all_morning_customer_id_and_fitness2:
                # non-dominated sorting
                front = non_dominated_sorting(self.all_morning_customer_id_and_fitness2)
                # selection
                population_list,new_pop=selection(self.pop_size,front,self.all_morning_customer_id_and_fitness2)
                self.all_morning_customer_id_and_fitness2 = population_list

            if self.all_morning_customer_id_and_fitness3:
                front = non_dominated_sorting(self.all_morning_customer_id_and_fitness3)
                population_list,new_pop=selection(self.pop_size,front,self.all_morning_customer_id_and_fitness3)
                self.all_morning_customer_id_and_fitness3 = population_list

            
            # non-dominated sorting
            front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness1)
            # selection
            population_list,new_pop=selection(self.pop_size,front,self.all_afternoon_customer_id_and_fitness1)
            self.all_afternoon_customer_id_and_fitness1 = population_list
            if self.all_afternoon_customer_id_and_fitness2:
                # non-dominated sorting
                front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness2)
                # selection
                population_list,new_pop=selection(self.pop_size,front,self.all_afternoon_customer_id_and_fitness2)
                self.all_afternoon_customer_id_and_fitness2 = population_list
            if self.all_afternoon_customer_id_and_fitness3:
                front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness3)
                population_list,new_pop=selection(self.pop_size,front,self.all_afternoon_customer_id_and_fitness3)
                self.all_afternoon_customer_id_and_fitness3 = population_list




    def result(self):
        
        front = non_dominated_sorting(self.all_morning_customer_id_and_fitness1)
        best_all_morning_customer_id_and_fitness1=selection(self.pop_size,front[0],self.all_morning_customer_id_and_fitness1 , best=True)
        self.satellite_demand.append(calculate_satellite_demand(self.json_instance , best_all_morning_customer_id_and_fitness1[0][0]))

        if self.all_morning_customer_id_and_fitness2:
            front = non_dominated_sorting(self.all_morning_customer_id_and_fitness2)
            best_all_morning_customer_id_and_fitness2=selection(self.pop_size,front[0],self.all_morning_customer_id_and_fitness2 , best=True)
            self.satellite_demand.append(calculate_satellite_demand(self.json_instance , best_all_morning_customer_id_and_fitness2[0][0]))
        if self.all_morning_customer_id_and_fitness3:
            front = non_dominated_sorting(self.all_morning_customer_id_and_fitness3)
            best_all_morning_customer_id_and_fitness3=selection(self.pop_size,front[0],self.all_morning_customer_id_and_fitness3 , best=True)
            self.satellite_demand.append(calculate_satellite_demand(self.json_instance , best_all_morning_customer_id_and_fitness3[0][0]))

        front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness1)
        best_all_afternoon_customer_id_and_fitness1=selection(self.pop_size,front[0],self.all_afternoon_customer_id_and_fitness1 , best=True)
        self.satellite_demand[0] += calculate_satellite_demand(self.json_instance , best_all_afternoon_customer_id_and_fitness1[0][0])
        if self.all_afternoon_customer_id_and_fitness2:
            front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness2)
            best_all_afternoon_customer_id_and_fitness2=selection(self.pop_size,front[0],self.all_afternoon_customer_id_and_fitness2 , best=True)
            self.satellite_demand[1] += calculate_satellite_demand(self.json_instance , best_all_afternoon_customer_id_and_fitness2[0][0])
        if self.all_afternoon_customer_id_and_fitness3:
            front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness3)
            best_all_afternoon_customer_id_and_fitness3=selection(self.pop_size,front[0],self.all_afternoon_customer_id_and_fitness3 , best=True)
            self.satellite_demand[2] += calculate_satellite_demand(self.json_instance , best_all_afternoon_customer_id_and_fitness3[0][0])



        self.all_number_vehicles = sum([best_all_morning_customer_id_and_fitness1[0][1][0] , best_all_afternoon_customer_id_and_fitness1[0][1][0] , best_all_morning_customer_id_and_fitness2[0][1][0]  , best_all_afternoon_customer_id_and_fitness2[0][1][0], best_all_afternoon_customer_id_and_fitness3[0][1][0] , best_all_morning_customer_id_and_fitness3[0][1][0]])
        self.SE_all_cost = sum([best_all_morning_customer_id_and_fitness1[0][1][1] , best_all_afternoon_customer_id_and_fitness1[0][1][1] , best_all_morning_customer_id_and_fitness2[0][1][1]  , best_all_afternoon_customer_id_and_fitness2[0][1][1] , best_all_afternoon_customer_id_and_fitness3[0][1][1] , best_all_morning_customer_id_and_fitness3[0][1][1]])
        
        
        self.SE_all_cost += self.all_number_vehicles * 50


    def Computation_time(self):
        self.all_time = time.time() -  self.start_time

    def runMain(self):
        self.load_instance()
        # self.filter_customers_in_china()
        # self.show_customer_on_china_map()
        self.initial_solution()
        self.initial_population()
        self.runGenerations()
        self.result()

        # self.Computation_time()

    def runMain(self):
        self.load_instance()
        # self.filter_customers_in_china()
        # self.show_customer_on_china_map()
        self.initial_solution()
        self.initial_population()
        self.runGenerations()
        self.result()
        # self.First_route()
        self.Computation_time()

if __name__ == "__main__":
    print("Running file directly, Executing nsga2vrp")

    C = [0.85 , 0.85]
    M = [0.6 , 1.0]
    select = 0
    start , end = 2200 , 2299
    # 開啟輸出檔案
    with open("OR_test_result2.txt", "w", encoding="utf-8") as f:
        for i in range(23,31):
            model = NSGAAlgorithm()
            model.start_customer_number = start
            model.end_customer_number = end
            model.crossover_probability = 0.85
            model.mut_prob = 0.5
            model.runMain()
            total = model.SE_all_cost
            result_line = f"第{i}天 , 開始:{start} ~ 結束:{end} ,交配率: {model.crossover_probability} , 突變率:{model.mut_prob} → Avg Total Cost: {total:.2f}"
            print(result_line)
            f.write(result_line + "\n")
            start, end = start + 100, end + 100