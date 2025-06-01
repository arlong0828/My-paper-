import random , openpyxl , copy , time
from haversine_python import haversine
from sklearn.cluster import KMeans
from operator import itemgetter
import matplotlib.pyplot as plt

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
    cost += (distance ) * 0.05 + 0.1 * max(instace[3] - time_morning_vehicles , 0) + 0.1 * max(time_morning_vehicles - instace[4], 0)
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

def cxOrderedVrp(input_ind1, input_ind2):
    # Modifying this to suit our needs
    #  If the sequence does not contain 0, this throws error
    #  So we will modify inputs here itself and then 
    #       modify the outputs too

    ind1 = copy.deepcopy(input_ind1)
    ind2 = copy.deepcopy(input_ind2)
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
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


def exchange(ind1, se_vehicle_capacity , instace):
    update_ind1 = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind , se_vehicle_capacity , instace)
    if len(route) >= 1:
        r = random.randrange(0 , len(route) , 1)
        if len(route[r]) > 1:
            a , b = random.sample(range(len(route[r])) , 2)
            route[r][a] , route[r][b] = route[r][b] , route[r][a]
        for i in route:
            update_ind1 += i   
        return update_ind1
    return ind
def opt2(ind1, se_vehicle_capacity , instace):
    update_ind1 = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind , se_vehicle_capacity , instace)
    if len(route) >= 1:
        r = random.randrange(0 , len(route) , 1)
        if len(route[r]) > 1:
            a , b = sorted(random.sample(range(len(route[r])) , 2))
            re = route[r][:a] + list(reversed(route[r][a:b + 1]))
            if b < len(route[r]) - 1:
                re += route[r][b + 1 :]
        else:
            re = route[r]
        for i in range(len(route)):
            if i == r:
                update_ind1 += re
            else:
                update_ind1 += route[i]
        return update_ind1
    return ind

def relocate(ind1, se_vehicle_capacity , instace):
    update_ind1 = []
    asd = []
    ind = copy.deepcopy(ind1)
    route = update_route(ind , se_vehicle_capacity , instace)
    if len(route) >= 1:
        r = random.randrange(0 , len(route) , 1)
        if len(route[r]) > 1:
            a , b = random.sample(range(len(route[r])) , 2)
            for i in range(len(route[r])):
                if i == b:
                    continue
                else:
                    asd.append(route[r][i])
            asd.insert(a , route[r][b])
        else:
            asd = route[r]
        for i in range(len(route)):
            if i == r:
                update_ind1 += asd
            else:
                update_ind1 += route[i]
        return update_ind1
    return ind


def s_exchange(ind1, se_vehicle_capacity , instace):
    update_ind1 = []
    ind = copy.deepcopy(ind1)
    # print(ind)
    route = update_route(ind , se_vehicle_capacity , instace)
    if len(route) >1:

        r1 , r2 = sorted(random.sample(range(len(route)) , 2))
        if len(route[r1]) > 1 and len(route[r2]) > 1:
            a = random.sample(range(len(route[r1])) , 1)
            b = random.sample(range(len(route[r2])) , 1)
            # print(a , b)
            route[r1][a[0]] , route[r2][b[0]] = route[r2][b[0]] , route[r1][a[0]]
        for i in route:
            update_ind1 += i   
        return update_ind1
    return ind

def all_s_exchange(ind1, se_vehicle_capacity , instace):
    update_ind1 = []
    ind = copy.deepcopy(ind1)
    # print(ind)
    route = update_route(ind , se_vehicle_capacity , instace)
    if len(route) >1:

        r1 , r2 = sorted(random.sample(range(len(route)) , 2))
        if len(route[r1]) > 1 and len(route[r2]) > 1:
            route[r1] , route[r2] = route[r2] , route[r1]
        for i in route:
            update_ind1 += i   
        return update_ind1
    return ind

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

class nsgaAlgo(object):
    def __init__(self):
        self.start_customer_number = 9001
        self.end_customer_number = 9300
        self.json_instance = ""
        self.mut_prob = 0.2
        self.num_gen = 1000
        self.pop_size = 1000
        self.best_pop = 1
        self.fe_vehicle_capacity = 20000
        self.se_vehicle_capacity = 300
        self.se_vehicle_speed = 50
        self.depot = [[120.373731 , 36.185609] , [118.054927 , 36.813487] , [116.897877 , 36.611274]]
        self.number_satellite = 2
        self.centers , self.customer_group = "" , ""
        self.all_morning_customer_id_and_fitness1 = []
        self.all_afternoon_customer_id_and_fitness1 = []
        self.all_morning_customer_id_and_fitness2 = []
        self.all_afternoon_customer_id_and_fitness2 = []
        self.start_time = time.time()
        self.score_history = []
        self.day = 0
        self.VNS_method = 3
        self.all = []
        self.all_cost = 0
        self.all_number_vehicles = 0
        self.all_time = 0
        self.all_time = [540,540,720,720]
        self.all_s = [0,0,1,1]

    def load_instance(self):
        # wb = openpyxl.load_workbook("./data/customer_coordinates.xlsx")
        # s1 = wb["Sheet1"]
        # arr = []
        # for row in s1:
        #     arr2 = []
        #     for col in row:
        #         if col.value != None:
        #             arr2.append(col.value)
        #     if arr2 != []:
        #         t = random.randrange(540,1020,30)
        #         arr2.append(t)
        #         arr2.append(t + random.randrange(30,60))
        #         arr2.append(random.randrange(10,40))
        #         arr2.append(10)
        #         arr.append(arr2)
        #         # print(arr)
        # del arr[0]
        self.json_instance = [[9001, 12.5115, 36.229725, 870, 907, 35, 10], [9002, 117.75116, 39.91611, 540, 574, 36, 10], [9003, 121.381658, 31.317871, 540, 574, 13, 10], [9004, 119.92175, 31.89745, 840, 899, 21, 10], [9005, 15.72215, 3.52647, 810, 866, 36, 10], [9006, 117.87118, 39.59875, 900, 935, 36, 10], [9007, 16.39284, 38.462584, 810, 846, 18, 10], [9008, 117.33393, 39.234561, 630, 684, 29, 10], [9009, 116.5817, 38.86681, 780, 825, 35, 10], [9010, 122.19487, 3.629889, 990, 1024, 27, 10], [9011, 121.27679, 31.174628, 840, 880, 38, 10], [9012, 117.891275, 39.5394, 810, 842, 16, 10], [9013, 115.659466, 37.992967, 960, 1011, 39, 10], [9014, 12.325351, 36.1412, 600, 642, 38, 10], [9015, 12.34674, 35.996898, 630, 677, 25, 10], [9016, 121.276952, 31.167895, 600, 644, 14, 10], [9017, 117.974781, 39.18151, 630, 661, 26, 10], [9018, 12.53528, 36.144943, 930, 987, 37, 10], [9019, 116.484499, 39.9522, 750, 805, 37, 10], [9020, 121.843314, 31.23788, 690, 736, 26, 10], [9021, 117.896112, 39.62648, 990, 1041, 15, 10], [9022, 121.721963, 31.321578, 960, 1012, 32, 10], [9023, 19.165447, 35.96627, 630, 668, 22, 10], [9024, 117.45836, 39.269944, 630, 663, 39, 10], [9025, 121.27847, 31.1685, 930, 962, 28, 10], [9026, 121.41936, 3.94731, 870, 906, 23, 10], [9027, 12.45335, 36.86338, 960, 1010, 28, 10], [9028, 121.888759, 31.143557, 780, 824, 11, 10], [9029, 118.863915, 36.6464712, 960, 1000, 11, 10], [9030, 118.146, 36.82353, 930, 983, 38, 10], [9031, 12.319235, 35.988591, 720, 770, 30, 10], [9032, 121.779724, 31.81158, 870, 920, 26, 10], [9033, 116.422885, 36.849466, 840, 897, 33, 10], [9034, 17.315958, 34.33896, 750, 804, 19, 10], [9035, 121.98231, 39.246, 900, 947, 29, 10], [9036, 121.99545, 3.887516, 930, 965, 27, 10], [9037, 12.272672, 36.8226, 540, 589, 15, 10], [9038, 119.5351, 35.99644, 570, 629, 28, 10], [9039, 117.887539, 38.986766, 570, 621, 16, 10], [9040, 117.485, 39.216821, 660, 699, 18, 10], [9041, 116.95822, 39.628454, 570, 601, 18, 10], [9042, 116.882767, 36.784164, 600, 647, 35, 10], [9043, 12.542117535138, 36.37697234494, 570, 610, 16, 10], [9044, 123.688416, 41.74997, 600, 645, 11, 10], [9045, 118.17966, 36.84195, 840, 878, 28, 10], [9046, 117.331972, 39.232946, 630, 681, 24, 10], [9047, 114.592282, 37.74579, 840, 875, 38, 10], [9048, 118.86286, 36.64841, 750, 786, 12, 10], [9049, 121.88873, 3.899466, 750, 802, 16, 10], [9050, 117.75719, 38.997376, 630, 664, 25, 10], [9051, 12.541236, 36.1623, 720, 757, 21, 10], [9052, 121.343543, 31.347, 600, 653, 25, 10], [9053, 122.19985, 3.619219, 630, 664, 29, 10], [9054, 116.72332, 39.761861, 660, 700, 28, 10], [9055, 12.55894, 36.164444, 930, 963, 12, 10], [9056, 117.87498, 39.58661, 990, 1034, 22, 10], [9057, 117.865116, 39.18626, 630, 673, 26, 10], [9058, 117.537929, 37.37595, 990, 1022, 17, 10], [9059, 121.27679, 31.174628, 870, 923, 33, 10], [9060, 121.22789, 31.196482, 630, 685, 23, 10], [9061, 116.567468, 35.9457, 960, 994, 33, 10], [9062, 121.343543, 31.347, 690, 736, 38, 10], [9063, 119.325258, 35.749664, 660, 697, 22, 10], [9064, 121.23559, 31.481811, 990, 1032, 22, 10], [9065, 121.766775, 42.111827, 900, 941, 15, 10], [9066, 126.91569, 45.733679, 960, 990, 37, 10], [9067, 121.461759, 31.38321, 660, 690, 24, 10], [9068, 121.311279, 3.99967, 720, 768, 21, 10], [9069, 117.233337, 36.22446, 930, 981, 25, 10], [9070, 117.47535, 39.323966, 720, 751, 30, 10], [9071, 116.89641, 36.662936, 660, 695, 26, 10], [9072, 12.358138, 36.539, 570, 622, 11, 10], [9073, 121.418775, 31.9555, 930, 988, 15, 10], [9074, 12.3169, 35.977213, 600, 637, 28, 10], [9075, 12.3169, 35.977213, 630, 667, 28, 10], [9076, 123.31976, 41.646697, 750, 789, 17, 10], [9077, 12.55432, 36.167169, 600, 649, 21, 10], [9078, 16.468381, 29.423664, 960, 999, 35, 10], [9079, 121.421717, 31.31658, 690, 721, 21, 10], [9080, 118.19794, 36.772175, 840, 882, 21, 10], [9081, 116.8388, 39.59289, 720, 750, 12, 10], [9082, 12.36296, 36.822, 930, 974, 27, 10], [9083, 12.358127, 36.5287, 720, 760, 29, 10], [9084, 113.866262, 23.46237, 960, 991, 14, 10], [9085, 124.866, 43.883913, 750, 780, 37, 10], [9086, 117.774173, 39.32846, 660, 717, 30, 10], [9087, 12.67278, 3.53343, 570, 604, 38, 10], [9088, 122.2141, 3.61178, 720, 764, 10, 10], [9089, 117.47555, 39.229642, 660, 702, 10, 10], [9090, 117.876928, 39.17964, 780, 829, 12, 10], [9091, 12.538269141611, 36.348844995644, 630, 688, 34, 10], [9092, 118.271, 36.751816, 780, 812, 27, 10], [9093, 121.96786, 39.196, 840, 897, 23, 10], [9094, 12.387592, 36.285461, 600, 657, 39, 10], [9095, 19.196717, 34.4993, 900, 944, 15, 10], [9096, 117.33185, 39.232181, 540, 581, 23, 10], [9097, 12.243435, 36.843, 840, 891, 13, 10], [9098, 123.87771, 42.274591, 720, 770, 33, 10], [9099, 12.2336, 36.94355, 930, 968, 11, 10]] 
        self.centers , self.customer_group = calculate_satellite_coordinates(self.json_instance , self.number_satellite)  

    def initial_solution(self):
        for i in range(self.number_satellite):
            group_ = []
            all_satellite_morning_id = []
            all_satellite_afternoon_id = []
            for j in range(len(self.customer_group)):
                if int(i) == int(self.customer_group[j]):
                    group_.append(self.json_instance[j])
            morning_customer , morning_customer_id , afternoon_customer , afternoon_customer_id= distinguish_between_periods(group_)

            for z in morning_customer_id:
                all_satellite_morning_id += z
            for z in afternoon_customer_id:
                all_satellite_afternoon_id += z
            if i == 0:
                fitness = satellite_calculate_fitness(all_satellite_morning_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[i])
                self.all_morning_customer_id_and_fitness1.append([all_satellite_morning_id , fitness])
                fitness2 = satellite_calculate_fitness(all_satellite_afternoon_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[i])
                self.all_afternoon_customer_id_and_fitness1.append([all_satellite_afternoon_id , fitness2])
            elif i == 1:
                fitness = satellite_calculate_fitness(all_satellite_morning_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[i])
                self.all_morning_customer_id_and_fitness2.append([all_satellite_morning_id , fitness])
                fitness2 = satellite_calculate_fitness(all_satellite_afternoon_id , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[i])
                self.all_afternoon_customer_id_and_fitness2.append([all_satellite_afternoon_id , fitness2])


        # print(self.all_morning_customer_id_and_fitness1 , "\n" , self.all_morning_customer_id_and_fitness2)
        # print(self.all_afternoon_customer_id_and_fitness1 , "\n" , self.all_afternoon_customer_id_and_fitness2)

        self.all.append([self.all_morning_customer_id_and_fitness1 , self.all_afternoon_customer_id_and_fitness1 , self.all_morning_customer_id_and_fitness2 , self.all_afternoon_customer_id_and_fitness2])
        # print(self.all)
    def runGenerations(self):
        for gen in range(self.num_gen):
            # print(f"{20*'#'} Currently Evaluating {gen} Generation {20*'#'}")
            
            # VNS
            for j in range(len(self.all[0])):
                i = 1
                while(i <= self.VNS_method):
                    # print(self.all[0][j])
                    fitmess1 = self.all[0][j][0][1]
                    ind1 = exchange(self.all[0][j][0][0] , self.se_vehicle_capacity , self.json_instance)
                    fitmess2 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = self.all_time[j] , satellite = self.centers[self.all_s[j]])
                    if fitmess1[1] > fitmess2[1]:
                        self.all[0][j].append([ind1 , fitmess2])
                        del self.all[0][j][0]
                        i = 0

                    ind1 = opt2(self.all[0][j][0][0] , self.se_vehicle_capacity , self.json_instance)
                    fitmess2 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = self.all_time[j] , satellite = self.centers[self.all_s[j]])
                    if fitmess1[1] > fitmess2[1]:
                        self.all[0][j].append([ind1 , fitmess2])
                        del self.all[0][j][0]
                        i = 0
                    
                    ind1 = relocate(self.all[0][j][0][0] , self.se_vehicle_capacity , self.json_instance)
                    fitmess2 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = self.all_time[j] , satellite = self.centers[self.all_s[j]])
                    if fitmess1[1] > fitmess2[1]:
                        self.all[0][j].append([ind1 , fitmess2])
                        del self.all[0][j][0]
                        i = 0
                    # print(self.all[0][j][0][0])
                    ind1 = s_exchange(self.all[0][j][0][0] , self.se_vehicle_capacity , self.json_instance)
                    fitmess2 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = self.all_time[j] , satellite = self.centers[self.all_s[j]])
                    if fitmess1[1] > fitmess2[1] and fitmess1[0] >= fitmess2[0]:
                        self.all[0][j].append([ind1 , fitmess2])
                        del self.all[0][j][0]
                        i = 0

                    ind1 = all_s_exchange(self.all[0][j][0][0] , self.se_vehicle_capacity , self.json_instance)
                    fitmess2 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = self.all_time[j] , satellite = self.centers[self.all_s[j]])
                    if fitmess1[1] > fitmess2[1] and fitmess1[0] >= fitmess2[0]:
                        self.all[0][j].append([ind1 , fitmess2])
                        del self.all[0][j][0]
                        i = 0

                    i += 1

    def result(self):
        
        front = non_dominated_sorting(self.all_morning_customer_id_and_fitness1)
        best_all_morning_customer_id_and_fitness1=selection(self.pop_size,front[0],self.all_morning_customer_id_and_fitness1 , best=True)

        front = non_dominated_sorting(self.all_morning_customer_id_and_fitness2)
        best_all_morning_customer_id_and_fitness2=selection(self.pop_size,front[0],self.all_morning_customer_id_and_fitness2 , best=True)

        front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness1)
        best_all_afternoon_customer_id_and_fitness1=selection(self.pop_size,front[0],self.all_afternoon_customer_id_and_fitness1 , best=True)

        front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness2)
        best_all_afternoon_customer_id_and_fitness2=selection(self.pop_size,front[0],self.all_afternoon_customer_id_and_fitness2 , best=True)

        # print("衛星0的最短路徑:")
        # print("上午：")
        # for i in best_all_morning_customer_id_and_fitness1:
        #     print(i)
        # print("下午：")
        # for i in best_all_afternoon_customer_id_and_fitness1:
        #     print(i)

        # print("衛星1的最短路徑：")
        # print("上午：")
        # for i in best_all_morning_customer_id_and_fitness2:
        #     print(i)
        # print("下午：")
        # for i in best_all_afternoon_customer_id_and_fitness2:
        #     print(i)
        # print(sum([best_all_morning_customer_id_and_fitness1[0][1][1] , best_all_afternoon_customer_id_and_fitness1[0][1][1] , best_all_morning_customer_id_and_fitness2[0][1][1]  , best_all_afternoon_customer_id_and_fitness2[0][1][1]]))
        self.all_cost = sum([best_all_morning_customer_id_and_fitness1[0][1][1] , best_all_afternoon_customer_id_and_fitness1[0][1][1] , best_all_morning_customer_id_and_fitness2[0][1][1]  , best_all_afternoon_customer_id_and_fitness2[0][1][1]])
        self.all_number_vehicles = sum([best_all_morning_customer_id_and_fitness1[0][1][0] , best_all_afternoon_customer_id_and_fitness1[0][1][0] , best_all_morning_customer_id_and_fitness2[0][1][0]  , best_all_afternoon_customer_id_and_fitness2[0][1][0]])
 
    def Computation_time(self):
        print("總共花費時間：" ,time.time() -  self.start_time)

    def plot_fitness(self):
        title = "CW_NSGAII with 2EMDCVRP, mute_prob={}__day = {}".format(self.mut_prob , self.day)
        plt.cla()
        plt.plot(self.score_history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title(title)
        plt.savefig(f'figures/mute_prob{self.mut_prob}__day = {self.day}.png')
        # plt.show()

    def runMain(self):
        self.load_instance()
        self.initial_solution()
        self.runGenerations()
        self.result()
        self.Computation_time()
        # self.plot_fitness()

if __name__ == "__main__":
    # someinstance = nsgaAlgo()
    # someinstance.runMain()
    print("Running file directly, Executing nsga2vrp")

    start = 9000
    end = 9049
    N = 1
    for i in range(10):
        A = []
        T = []
        # print("測試資料：" , N , "開始：" , start , "結束：" , end)
        for j in range(1):
            someinstance = nsgaAlgo()
            someinstance.start_customer_number = start
            someinstance.end_customer_number = end
            print("第" , N , "個實例" , "     開始：" , someinstance.start_customer_number , "      結束：" , someinstance.end_customer_number)

            someinstance.runMain()
            print("總成本：" , someinstance.all_cost , "總車輛數：" , someinstance.all_number_vehicles ,  "總時間：" , someinstance.all_time)
            
            # A.append(someinstance.all_cost)
            # T.append(someinstance.all_time)
        # print(N , " " , start , ", " , end)
        start += 50
        end += 50
        N += 1
        