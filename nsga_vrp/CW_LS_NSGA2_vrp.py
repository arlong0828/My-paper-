import random , openpyxl , copy , time
from haversine_python import haversine
from sklearn.cluster import KMeans
from operator import itemgetter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
from FE_gurobipy import FE_gurobi

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


def mutation(ind1 , mut_prob):
    if len(ind1) > 1 and random.random() <= mut_prob:
        ind2 = copy.deepcopy(ind1)
        a , b= sorted(random.sample(range(len(ind2)), 2))
        mutated = ind2[:a] + list(reversed(ind2[a:b + 1]))
        if b < len(ind2) - 1:
            mutated += ind2[b + 1 :]
        return mutated
    return ind1

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
        self.start_customer_number = 0
        self.end_customer_number = 99
        self.json_instance = ""
        self.crossover_probability = 0.85
        self.mut_prob = 0.1
        self.num_gen = 10
        self.pop_size = 20
        self.best_pop = 1
        self.fe_vehicle_capacity = 1000
        self.se_vehicle_capacity = 200
        self.se_vehicle_speed = 50
        self.depot = [["d1" , 120.373731 , 36.185609] , ["d2" , 118.054927 , 36.813487] , ["d3" , 116.897877 , 36.611274]]
        self.number_satellite = 2
        self.centers , self.customer_group = "" , ""
        self.all_morning_customer_id_and_fitness1 = []
        self.all_afternoon_customer_id_and_fitness1 = []
        self.all_morning_customer_id_and_fitness2 = []
        self.all_afternoon_customer_id_and_fitness2 = []
        self.start_time = time.time()
        self.score_history = []
        self.day = 0
        self.all_number_vehicles = 0
        self.all_time = 0
        self.FE_all_cost = 0
        self.SE_all_cost = 0

    def load_instance(self):
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
        self.centers , self.customer_group = calculate_satellite_coordinates(self.json_instance , self.number_satellite)
    def filter_customers_in_china(self):
        df = pd.read_excel("./data/customer_data2.xlsx")

        # 確認含有 'latitude' 和 'longitude' 欄位
        geometry = [Point(xy) for xy in zip(df['經度'], df['緯度'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84座標系統

        # 載入世界國界（GeoPandas 內建世界地圖）
        world = gpd.read_file("./data/ne_110m_admin_0_countries.shp")
        # 選出中國國界
        china = world[world['NAME'] == 'China']

        # 精確判斷是否在中國內
        in_china = gdf[gdf.within(china.geometry.iloc[0])]

        # 將在中國內的資料輸出為 Excel
        in_china.to_excel("filtered_customer_data_precise.xlsx", index=False)
    

    def show_customer_on_china_map(self):
        # 原始顧客座標 + 擾動
        customer_lons = [i[1] + np.random.uniform(-0.02, 0.02) for i in self.json_instance]
        customer_lats = [i[2] + np.random.uniform(-0.02, 0.02) for i in self.json_instance]

        # 倉庫座標
        depot_lons = [d[0] for d in self.depot]
        depot_lats = [d[1] for d in self.depot]

        # 👉 合併兩者用來計算邊界範圍
        all_lons = customer_lons + depot_lons
        all_lats = customer_lats + depot_lats
        # print(max(lons) , min(lons) , max(lats) , min(lats))
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # 自動計算留白範圍
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        padding_ratio = 0.05

        ax.set_extent([110.80054706818404 ,127.5746712695365, 21.25950285013494, 49.64801823827201
        ], crs=ccrs.PlateCarree())

        # ax.set_extent([min(lons) - 2, max(lons) + 2, min(lats)-2 , max(lats)+2], crs=ccrs.PlateCarree())

        # 地圖基本要素
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.LAKES, facecolor='white', edgecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.RIVERS, edgecolor='lightblue', linewidth=0.4)

        # 畫出顧客點（使用倒三角形）
        ax.scatter(customer_lons, customer_lats, color='black', marker='v', s=20, alpha=0.5,
           label='Customer', transform=ccrs.PlateCarree())

        # 若要加亂數擾動避免完全重疊，先處理 lons/lats 為：

        # ax.scatter(lons, lats, color='black', marker='v', s=100, label='Customer' , transform=ccrs.PlateCarree())

        for i in range(len(self.depot)):
            label = 'Depot' if i == 0 else None
            ax.scatter([depot_lons[i]], [depot_lats[i]], color='red', marker='+', s=100,
                    label=label, transform=ccrs.PlateCarree())
        # ax.set_title('Customer Distribution on China Map')
        ax.legend()
        plt.show()
        # plt.draw()
        # fig.savefig(f"./Day/china_customers{self.day}.png", bbox_inches='tight', pad_inches=0, dpi=300)

    def initial_solution(self):
        for i in range(self.number_satellite):
            group_ = []
            all_satellite_morning_id = []
            all_satellite_afternoon_id = []
            for j in range(len(self.customer_group)):
                if int(i) == int(self.customer_group[j]):
                    group_.append(self.json_instance[j])
            morning_customer , morning_customer_id , afternoon_customer , afternoon_customer_id= distinguish_between_periods(group_)
            morning_customer.insert(0 , self.centers[i])
            afternoon_customer.insert(0 , self.centers[i])
            # print("上午客戶：" , morning_customer , "\n", "下午客戶：" ,afternoon_customer , "\n")
            # print(morning_customer_id , "\n" , afternoon_customer_id , "\n")
            morning_distances , afternoon_distances = get_distance(morning_customer , afternoon_customer)
            # print(morning_distances , "\n" , afternoon_distances , "\n")
            morning_savings , afternoon_savings = savingsAlgorithms(morning_distances , afternoon_distances , morning_customer_id , afternoon_customer_id)
            # print(morning_savings , "\n" , afternoon_savings , "\n")
            morning_customer_id , afternoon_customer_id = getRoute(self.json_instance , morning_savings , afternoon_savings , morning_customer_id , afternoon_customer_id , self.se_vehicle_capacity)
            morning_customer_fitness , afternoon_customer_fitness = eval_indvidual_fitness(self.se_vehicle_speed , morning_customer_id , afternoon_customer_id, instace = self.json_instance , SatelliteCoordinates = self.centers[i] )
            # print(morning_customer_fitness , afternoon_customer_fitness)
            # print("早上客戶路線：" , morning_customer_id , "finess：" , morning_customer_fitness , "\n" , "下午客戶綠線：" , afternoon_customer_id , "finess：" , afternoon_customer_fitness) 
            for z in morning_customer_id:
                all_satellite_morning_id += z
            if i == 0:
                self.all_morning_customer_id_and_fitness1.append([all_satellite_morning_id , morning_customer_fitness])
            elif i == 1:
                self.all_morning_customer_id_and_fitness2.append([all_satellite_morning_id , morning_customer_fitness])
            for z in afternoon_customer_id:
                all_satellite_afternoon_id += z
            if i == 0:
                self.all_afternoon_customer_id_and_fitness1.append([all_satellite_afternoon_id , afternoon_customer_fitness])
            elif i == 1:
                self.all_afternoon_customer_id_and_fitness2.append([all_satellite_afternoon_id , afternoon_customer_fitness])

        # print(self.all_morning_customer_id_and_fitness1 , "\n" , self.all_morning_customer_id_and_fitness2)
        # print(self.all_afternoon_customer_id_and_fitness1 , "\n" , self.all_afternoon_customer_id_and_fitness2)

    def initial_population(self):
        if len(self.all_morning_customer_id_and_fitness1[0][0]) > 5:
            while(len(self.all_morning_customer_id_and_fitness1) < 20):
                for i in range(len(self.all_morning_customer_id_and_fitness1)):

                    exchange_ind = exchange(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness1 , exchange_ind):
                        fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                        self.all_morning_customer_id_and_fitness1.append([exchange_ind , fitmess2])

                    opt2_ind = opt2(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness1 , opt2_ind):
                        fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                        self.all_morning_customer_id_and_fitness1.append([opt2_ind , fitmess2])
    
                    relocate_ind = relocate(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness1 , relocate_ind):
                        fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                        self.all_morning_customer_id_and_fitness1.append([relocate_ind , fitmess2])


        if len(self.all_morning_customer_id_and_fitness2[0][0]) > 5:
            while(len(self.all_morning_customer_id_and_fitness2) < 20):
                for i in range(len(self.all_morning_customer_id_and_fitness2)):

                    exchange_ind = exchange(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness2 , exchange_ind):
                        fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                        self.all_morning_customer_id_and_fitness2.append([exchange_ind , fitmess2])

                    opt2_ind = opt2(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness2 , opt2_ind):
                        fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                        self.all_morning_customer_id_and_fitness2.append([opt2_ind , fitmess2])
    
                    relocate_ind = relocate(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_morning_customer_id_and_fitness2 , relocate_ind):
                        fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                        self.all_morning_customer_id_and_fitness2.append([relocate_ind , fitmess2])

        if len(self.all_afternoon_customer_id_and_fitness1[0][0]) > 5:
            while(len(self.all_afternoon_customer_id_and_fitness1) < 20):
                for i in range(len(self.all_afternoon_customer_id_and_fitness1)):

                    exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness1 , exchange_ind):
                        fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
                        self.all_afternoon_customer_id_and_fitness1.append([exchange_ind , fitmess2])

                    opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness1 , opt2_ind):
                        fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
                        self.all_afternoon_customer_id_and_fitness1.append([opt2_ind , fitmess2])
    
                    relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness1 , relocate_ind):
                        fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
                        self.all_afternoon_customer_id_and_fitness1.append([relocate_ind , fitmess2])

        if len(self.all_afternoon_customer_id_and_fitness2[0][0]) > 5:
            while(len(self.all_afternoon_customer_id_and_fitness2) < 20):
                for i in range(len(self.all_afternoon_customer_id_and_fitness2)):

                    exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness2 , exchange_ind):
                        fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                        self.all_afternoon_customer_id_and_fitness2.append([exchange_ind , fitmess2])

                    opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness2 , opt2_ind):
                        fitmess2 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                        self.all_afternoon_customer_id_and_fitness2.append([opt2_ind , fitmess2])
    
                    relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                    if find_smae(self.all_afternoon_customer_id_and_fitness2 , relocate_ind):
                        fitmess2 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                        self.all_afternoon_customer_id_and_fitness2.append([relocate_ind , fitmess2])
        # print(self.all_morning_customer_id_and_fitness1 , "\n" , self.all_morning_customer_id_and_fitness2)
        # print(self.all_afternoon_customer_id_and_fitness1 , "\n" , self.all_afternoon_customer_id_and_fitness2)

    def runGenerations(self):
        for gen in range(self.num_gen):
            # print(f"{20*'#'} Currently Evaluating {gen} Generation {20*'#'}")
            # local search 
            for i in range(len(self.all_morning_customer_id_and_fitness1)):
                fitness1 = satellite_calculate_fitness(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                
                exchange_ind = exchange(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness1 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                    if fitness1[1] > fitmess2[1]:
                        self.all_morning_customer_id_and_fitness1.append([exchange_ind , fitmess2])
                        del self.all_morning_customer_id_and_fitness1[i]
                        break

                opt2_ind = opt2(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness1 , opt2_ind):
                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                    if fitness1[1] > fitmess3[1]:
                        self.all_morning_customer_id_and_fitness1.append([opt2_ind , fitmess3])
                        del self.all_morning_customer_id_and_fitness1[i]
                        break

                relocate_ind = relocate(self.all_morning_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness1 , relocate_ind):
                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                    if fitness1[1] > fitmess4[1]:
                        self.all_morning_customer_id_and_fitness1.append([relocate_ind , fitmess4])
                        del self.all_morning_customer_id_and_fitness1[i]
                        break

            for i in range(len(self.all_morning_customer_id_and_fitness2)):
                fitness1 = satellite_calculate_fitness(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                
                exchange_ind = exchange(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness2 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                    if fitness1[1] > fitmess2[1]:
                        self.all_morning_customer_id_and_fitness2.append([exchange_ind , fitmess2])
                        del self.all_morning_customer_id_and_fitness2[i]
                        break
                
                opt2_ind = opt2(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness2 , opt2_ind):

                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                    if fitness1[1] > fitmess3[1]:
                        self.all_morning_customer_id_and_fitness2.append([opt2_ind , fitmess3])
                        del self.all_morning_customer_id_and_fitness2[i]
                        break

                relocate_ind = relocate(self.all_morning_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_morning_customer_id_and_fitness2 , relocate_ind):

                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                    if fitness1[1] > fitmess4[1]:
                        self.all_morning_customer_id_and_fitness2.append([relocate_ind , fitmess4])
                        del self.all_morning_customer_id_and_fitness2[i]
                        break
            for i in range(len(self.all_afternoon_customer_id_and_fitness1)):
                fitness1 = satellite_calculate_fitness(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
            
                exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness1 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
                    if fitness1[1] > fitmess2[1]:
                        self.all_afternoon_customer_id_and_fitness1.append([exchange_ind , fitmess2])
                        del self.all_afternoon_customer_id_and_fitness1[i]
                        break
                
                opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness1 , opt2_ind):
                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
                    if fitness1[1] > fitmess3[1]:
                        self.all_afternoon_customer_id_and_fitness1.append([opt2_ind , fitmess3])
                        del self.all_afternoon_customer_id_and_fitness1[i]
                        break
                
                relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness1[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness1 , relocate_ind):
                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
                    if fitness1[1] > fitmess4[1]:
                        self.all_afternoon_customer_id_and_fitness1.append([relocate_ind , fitmess4])
                        del self.all_afternoon_customer_id_and_fitness1[i]
                        break

            for i in range(len(self.all_afternoon_customer_id_and_fitness2)):

                fitness1 = satellite_calculate_fitness(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                exchange_ind = exchange(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness2 , exchange_ind):
                    fitmess2 = satellite_calculate_fitness(exchange_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                    if fitness1[1] > fitmess2[1]:
                        self.all_afternoon_customer_id_and_fitness2.append([exchange_ind , fitmess2])
                        del self.all_afternoon_customer_id_and_fitness2[i]
                        break
                opt2_ind = opt2(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness2 , opt2_ind):
                    fitmess3 = satellite_calculate_fitness(opt2_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                    if fitness1[1] > fitmess3[1]:
                        self.all_afternoon_customer_id_and_fitness2.append([opt2_ind , fitmess3])
                        del self.all_afternoon_customer_id_and_fitness2[i]
                        break
                relocate_ind = relocate(self.all_afternoon_customer_id_and_fitness2[i][0] , self.se_vehicle_capacity , self.json_instance)
                if find_smae(self.all_afternoon_customer_id_and_fitness2 , relocate_ind):
                    fitmess4 = satellite_calculate_fitness(relocate_ind , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                    if fitness1[1] > fitmess4[1]:
                        self.all_afternoon_customer_id_and_fitness2.append([relocate_ind , fitmess4])
                        del self.all_afternoon_customer_id_and_fitness2[i]
                        break
            # print(self.all_morning_customer_id_and_fitness1 , "\n" , self.all_morning_customer_id_and_fitness2)
            # print(self.all_afternoon_customer_id_and_fitness2)
            # 交配
            if len(self.all_morning_customer_id_and_fitness1[0][0]) > 5 and self.crossover_probability >= random.random():
                while(len(self.all_morning_customer_id_and_fitness1) < 40):
                    for i in range(0 , len(self.all_morning_customer_id_and_fitness1) , 2):
                        if i + 1 < len(self.all_morning_customer_id_and_fitness1):
                        # print(self.all_morning_customer_id_and_fitness1[i][0] , self.all_morning_customer_id_and_fitness1[i + 1][0])
                            ind1, ind2  = cxOrderedVrp(self.all_morning_customer_id_and_fitness1[i][0] , self.all_morning_customer_id_and_fitness1[i + 1][0])
                            # print(ind1, ind2)
                            if find_smae(self.all_morning_customer_id_and_fitness1 , ind1):
                                fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                                self.all_morning_customer_id_and_fitness1.append([ind1 , fitness1])
                            if find_smae(self.all_morning_customer_id_and_fitness1 , ind2):
                                fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                                self.all_morning_customer_id_and_fitness1.append([ind2 , fitness2])
            # print(self.all_morning_customer_id_and_fitness1)
            # print(len(self.all_morning_customer_id_and_fitness1))

            if len(self.all_morning_customer_id_and_fitness2[0][0]) > 5 and self.crossover_probability >= random.random():
                while(len(self.all_morning_customer_id_and_fitness2) < 40):
                    for i in range(0 , len(self.all_morning_customer_id_and_fitness2) , 2):
                        if i + 1 < len(self.all_morning_customer_id_and_fitness2):
                        # print(self.all_morning_customer_id_and_fitness1[i][0] , self.all_morning_customer_id_and_fitness1[i + 1][0])
                            ind1, ind2  = cxOrderedVrp(self.all_morning_customer_id_and_fitness2[i][0] , self.all_morning_customer_id_and_fitness2[i + 1][0])
                            # print(ind1, ind2)
                            if find_smae(self.all_morning_customer_id_and_fitness2 , ind1):
                                fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                                self.all_morning_customer_id_and_fitness2.append([ind1 , fitness1])
                            if find_smae(self.all_morning_customer_id_and_fitness2 , ind2):
                                fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                                self.all_morning_customer_id_and_fitness2.append([ind2 , fitness2])
            # print(self.all_morning_customer_id_and_fitness2)
            # print(len(self.all_morning_customer_id_and_fitness2))

            if len(self.all_afternoon_customer_id_and_fitness1[0][0]) > 5 and self.crossover_probability >= random.random():
                while(len(self.all_afternoon_customer_id_and_fitness1) < 40):
                    for i in range(0 , len(self.all_afternoon_customer_id_and_fitness1) , 2):
                        if i + 1 < len(self.all_afternoon_customer_id_and_fitness1):
                        
                        # print(self.all_morning_customer_id_and_fitness1[i][0] , self.all_morning_customer_id_and_fitness1[i + 1][0])
                            ind1, ind2  = cxOrderedVrp(self.all_afternoon_customer_id_and_fitness1[i][0] , self.all_afternoon_customer_id_and_fitness1[i + 1][0])
                            # print(ind1, ind2)
                            if find_smae(self.all_afternoon_customer_id_and_fitness1 , ind1):
                                fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
                                self.all_afternoon_customer_id_and_fitness1.append([ind1 , fitness1])
                            if find_smae(self.all_afternoon_customer_id_and_fitness1 , ind2):
                                fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])
                                self.all_afternoon_customer_id_and_fitness1.append([ind2 , fitness2])
            # print(self.all_afternoon_customer_id_and_fitness1)
            # print(len(self.all_afternoon_customer_id_and_fitness1))

            if len(self.all_afternoon_customer_id_and_fitness2[0][0]) > 5 and self.crossover_probability >= random.random():
                while(len(self.all_afternoon_customer_id_and_fitness2) < 40):
                    for i in range(0 , len(self.all_afternoon_customer_id_and_fitness2) , 2):
                        if i + 1 < len(self.all_afternoon_customer_id_and_fitness2):
                        # print(self.all_morning_customer_id_and_fitness1[i][0] , self.all_morning_customer_id_and_fitness1[i + 1][0])
                            ind1, ind2  = cxOrderedVrp(self.all_afternoon_customer_id_and_fitness2[i][0] , self.all_afternoon_customer_id_and_fitness2[i + 1][0])
                            # print(ind1, ind2)
                            if find_smae(self.all_afternoon_customer_id_and_fitness2 , ind1):
                                fitness1 = satellite_calculate_fitness(ind1 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                                self.all_afternoon_customer_id_and_fitness2.append([ind1 , fitness1])
                            if find_smae(self.all_afternoon_customer_id_and_fitness2 , ind2):
                                fitness2 = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])
                                self.all_afternoon_customer_id_and_fitness2.append([ind2 , fitness2])
            # print(self.all_afternoon_customer_id_and_fitness2)
            # print(len(self.all_afternoon_customer_id_and_fitness2))

            # 突變
            if len(self.all_morning_customer_id_and_fitness1[0][0]) > 4:
                for i in self.all_morning_customer_id_and_fitness1:
                    ind2 = mutation(i[0] , self.mut_prob)
                    if find_smae(self.all_morning_customer_id_and_fitness1 , ind2):
                        fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[0])
                        self.all_morning_customer_id_and_fitness1.append([ind2 , fitness])
                        break
            # print(self.all_morning_customer_id_and_fitness1)    

            if len(self.all_morning_customer_id_and_fitness2[0][0]) > 4:
                for i in self.all_morning_customer_id_and_fitness2:
                    ind2 = mutation(i[0] , self.mut_prob)
                    if find_smae(self.all_morning_customer_id_and_fitness2 , ind2):
                        fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 540 , satellite = self.centers[1])
                        self.all_morning_customer_id_and_fitness2.append([ind2 , fitness]) 
                        break
            # print(self.all_morning_customer_id_and_fitness2) 
            #  
            if len(self.all_afternoon_customer_id_and_fitness1[0][0]) > 4:
                for i in self.all_afternoon_customer_id_and_fitness1:
                    ind2 = mutation(i[0] , self.mut_prob)
                    if find_smae(self.all_afternoon_customer_id_and_fitness1 , ind2):
                        fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[0])  
                        self.all_afternoon_customer_id_and_fitness1.append([ind2 , fitness]) 
                        break
            # print(self.all_afternoon_customer_id_and_fitness1)  

            if len(self.all_afternoon_customer_id_and_fitness2[0][0]) > 4:
                for i in self.all_afternoon_customer_id_and_fitness2:
                    ind2 = mutation(i[0] , self.mut_prob)
                    if find_smae(self.all_afternoon_customer_id_and_fitness2 , ind2):
                        fitness = satellite_calculate_fitness(ind2 , self.se_vehicle_speed , self.se_vehicle_capacity , self.json_instance , time_vehicles = 720 , satellite = self.centers[1])         
                        self.all_afternoon_customer_id_and_fitness2.append([ind2 , fitness])  
                        break
            # print(self.all_afternoon_customer_id_and_fitness2)

            # non-dominated sorting
            front = non_dominated_sorting(self.all_morning_customer_id_and_fitness1)
            # selection
            population_list,new_pop=selection(self.pop_size,front,self.all_morning_customer_id_and_fitness1)
            self.all_morning_customer_id_and_fitness1 = population_list

            # non-dominated sorting
            front = non_dominated_sorting(self.all_morning_customer_id_and_fitness2)
            # selection
            population_list,new_pop=selection(self.pop_size,front,self.all_morning_customer_id_and_fitness2)
            self.all_morning_customer_id_and_fitness2 = population_list

            # non-dominated sorting
            front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness1)
            # selection
            population_list,new_pop=selection(self.pop_size,front,self.all_afternoon_customer_id_and_fitness1)
            self.all_afternoon_customer_id_and_fitness1 = population_list

            # non-dominated sorting
            front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness2)
            # selection
            population_list,new_pop=selection(self.pop_size,front,self.all_afternoon_customer_id_and_fitness2)
            self.all_afternoon_customer_id_and_fitness2 = population_list
            

            self.score_history.append((self.all_morning_customer_id_and_fitness1[0][1][1] + self.all_morning_customer_id_and_fitness2[0][1][1] + self.all_afternoon_customer_id_and_fitness1[0][1][1]+
                                      self.all_afternoon_customer_id_and_fitness2[0][1][1]))

            # print((self.all_morning_customer_id_and_fitness1[0][1][1] + self.all_morning_customer_id_and_fitness2[0][1][1] + self.all_afternoon_customer_id_and_fitness1[0][1][1]+
            #                           self.all_afternoon_customer_id_and_fitness2[0][1][1]))

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
        # print(best_all_morning_customer_id_and_fitness1[0][1][1])
        self.all_cost = sum([best_all_morning_customer_id_and_fitness1[0][1][1] , best_all_afternoon_customer_id_and_fitness1[0][1][1] , best_all_morning_customer_id_and_fitness2[0][1][1]  , best_all_afternoon_customer_id_and_fitness2[0][1][1]])
        self.all_number_vehicles = sum([best_all_morning_customer_id_and_fitness1[0][1][0] , best_all_afternoon_customer_id_and_fitness1[0][1][0] , best_all_morning_customer_id_and_fitness2[0][1][0]  , best_all_afternoon_customer_id_and_fitness2[0][1][0]])
        # self.all_cost = sum([best_all_morning_customer_id_and_fitness1[0][1][1] , best_all_afternoon_customer_id_and_fitness1[0][1][1] , best_all_morning_customer_id_and_fitness2[0][1][1]  , best_all_afternoon_customer_id_and_fitness2[0][1][1]])
        # print(sum([best_all_morning_customer_id_and_fitness1[0][1][1] , best_all_afternoon_customer_id_and_fitness1[0][1][1] , best_all_morning_customer_id_and_fitness2[0][1][1]  , best_all_afternoon_customer_id_and_fitness2[0][1][1]]))
        # print(sum([best_all_morning_customer_id_and_fitness1[0][1][0] , best_all_afternoon_customer_id_and_fitness1[0][1][0] , best_all_morning_customer_id_and_fitness2[0][1][0]  , best_all_afternoon_customer_id_and_fitness2[0][1][0]]))
        self.SE_all_cost = sum([best_all_morning_customer_id_and_fitness1[0][1][1] , best_all_afternoon_customer_id_and_fitness1[0][1][1] , best_all_morning_customer_id_and_fitness2[0][1][1]  , best_all_afternoon_customer_id_and_fitness2[0][1][1]])
        self.SE_all_cost += self.all_number_vehicles * 50 + self.number_satellite * 1000
        # print("第二階段成本：", self.SE_all_cost)
        self.all_number_vehicles = sum([best_all_morning_customer_id_and_fitness1[0][1][0] , best_all_afternoon_customer_id_and_fitness1[0][1][0] , best_all_morning_customer_id_and_fitness2[0][1][0]  , best_all_afternoon_customer_id_and_fitness2[0][1][0]])
        # print("第二階段車輛數：",self.all_number_vehicles)

    def First_route(self):
        gurobi_Model = FE_gurobi()
        self.FE_all_cost = gurobi_Model.main(self.depot , self.number_satellite , self.centers , [1000,1000] , self.fe_vehicle_capacity)
        # print("第一階段成本：",self.FE_all_cost)
        # pass

    def Computation_time(self):
        # print("總共花費時間：" ,time.time() -  self.start_time)
        self.all_time = time.time() -  self.start_time

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
        # self.filter_customers_in_china()
        # self.show_customer_on_china_map()
        self.initial_solution()
        self.initial_population()
        self.runGenerations()
        self.result()
        self.First_route()
        self.Computation_time()
        # self.plot_fitness()

if __name__ == "__main__":
    # someinstance = nsgaAlgo()
    # someinstance.runMain()
    print("Running file directly, Executing nsga2vrp")
    start = 0
    end = 99    
    C = 0.9
    M = 0.0

    for d in range(30):
        crossover_stats = []  # 儲存交配率與平均值 (C, avg)
        
        for i in range(5):
            A = []
            V = []
            for j in range(10):
                someinstance = nsgaAlgo()
                someinstance.start_customer_number = start
                someinstance.end_customer_number = end
                someinstance.crossover_probability = C
                someinstance.mut_prob = M
                someinstance.runMain()
                print("總成本：" , someinstance.SE_all_cost , "總車輛數：" , someinstance.all_number_vehicles)
                cost = someinstance.FE_all_cost + someinstance.SE_all_cost
                A.append(cost)
                V.append(someinstance.all_number_vehicles)

            arr = np.array(A)
            avg_cost = np.mean(arr)
            crossover_stats.append((C, avg_cost))  # 存交配率與平均值
            print("第", d + 1, "天 第", i + 1, "組")
            print("交配率：" , C , "突變率：" , M)
            print("平均值：", avg_cost , "最小值：" , np.min(arr))
            
            C = round(C - 0.05 , 2)  # 每組交配率遞減
        
        # 找出當天平均成本最小的交配率
        best_C, best_avg = min(crossover_stats, key=lambda x: x[1])
        print("====== 第", d + 1, "天最佳交配率 ======")
        print("最佳交配率：", best_C, "  對應平均總成本：", best_avg)
        print("===================================")

        # 重設交配率、切換客戶編號範圍
        C = 0.9
        start += 100
        end += 100
        # 第一梯隊路線