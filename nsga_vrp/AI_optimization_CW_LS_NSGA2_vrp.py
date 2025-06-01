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

# 固定亂數種子
random.seed(1)

def calculate_satellite_coordinates(instance, number_satellite):
    """
    使用 KMeans 將顧客依照經緯度分群，返回中心點與所屬群組。
    """
    coordinates = [[row[1], row[2]] for row in instance]
    model = KMeans(n_clusters=number_satellite, n_init='auto')
    model.fit(coordinates)
    return model.cluster_centers_, model.labels_

def distinguish_between_periods(instance, periods=720):
    """
    根據時間閾值 (periods)，區分早上與下午顧客。
    """
    morning_customer = []
    morning_customer_id = []
    afternoon_customer = []
    afternoon_customer_id = []
    
    for customer in instance:
        if customer[4] <= periods:
            morning_customer.append(customer)
            morning_customer_id.append([customer[0]])
        else:
            afternoon_customer.append(customer)
            afternoon_customer_id.append([customer[0]])

    return morning_customer, morning_customer_id, afternoon_customer, afternoon_customer_id

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    計算兩點間的 Haversine 距離。
    """
    return haversine().getDistanceBetweenPointsNew(lat1, lon1, lat2, lon2)

def calculate_time(time_elapsed, customer, distance, speed, cost):
    """
    更新累計時間與成本。
    """
    time_elapsed += (distance / speed) * 60
    cost += (distance * 0.05 + 
             0.2 * max(customer[3] - time_elapsed, 0) + 
             0.2 * max(time_elapsed - customer[4], 0))
    return time_elapsed, cost

def get_distance(morning_customers, afternoon_customers):
    def compute_distance_matrix(customers):
        n = len(customers)
        matrix = np.zeros((n, n))
        for i in range(n):
            lon1, lat1 = customers[i][0], customers[i][1]
            for j in range(n):
                if i != j:
                    lon2, lat2 = customers[j][0], customers[j][1]
                    matrix[i][j] = calculate_distance(lon1, lat1, lon2, lat2)
        return matrix.tolist()

    morning_distances = compute_distance_matrix(morning_customers)
    afternoon_distances = compute_distance_matrix(afternoon_customers)

    return morning_distances, afternoon_distances


def savings_algorithms(morning_distances, afternoon_distances, morning_customer_id, afternoon_customer_id):
    def compute_savings(distances, customer_ids):
        savings = []
        n = len(customer_ids)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                saving = (distances[i][0] + distances[0][j]) - distances[i][j]
                savings.append([customer_ids[i - 1][0], customer_ids[j - 1][0], saving])
        return sorted(savings, key=itemgetter(2), reverse=True)

    return compute_savings(morning_distances, morning_customer_id), compute_savings(afternoon_distances, afternoon_customer_id)

# 路徑建構函數

def get_routes(instance, morning_savings, afternoon_savings, morning_customer_id, afternoon_customer_id, se_vehicle_capacity):
    def merge_routes(savings_list, customer_routes):
        for saving in savings_list:
            start_route, end_route, demand = [], [], 0
            for route in customer_routes:
                if saving[0] == route[-1]:
                    end_route = route
                elif saving[1] == route[0]:
                    start_route = route
                if start_route and end_route:
                    merged_route = end_route + start_route
                    demand = sum(ins[5] + ins[6] for node in merged_route for ins in instance if ins[0] == node)
                    if demand <= se_vehicle_capacity:
                        customer_routes.remove(end_route)
                        customer_routes.remove(start_route)
                        customer_routes.append(merged_route)
                    break
        return customer_routes

    updated_morning_routes = merge_routes(morning_savings, morning_customer_id)
    updated_afternoon_routes = merge_routes(afternoon_savings, afternoon_customer_id)
    return updated_morning_routes, updated_afternoon_routes

# 適應度計算函數

def eval_individual_fitness(se_vehicle_speed, morning_customer_id, afternoon_customer_id, instance, satellite_coordinates):
    def evaluate_route(route, start_time):
        time_elapsed = start_time
        cost = 0
        for idx, customer_id in enumerate(route):
            customer = next(c for c in instance if c[0] == customer_id)
            if idx == 0:
                dist = calculate_distance(satellite_coordinates[1], satellite_coordinates[0], customer[2], customer[1])
            else:
                prev_customer = next(c for c in instance if c[0] == route[idx - 1])
                dist = calculate_distance(prev_customer[2], prev_customer[1], customer[2], customer[1])
            time_elapsed, cost = calculate_time(time_elapsed, customer, dist, se_vehicle_speed, cost)
        last_customer = next(c for c in instance if c[0] == route[-1])
        cost += calculate_distance(last_customer[2], last_customer[1], satellite_coordinates[1], satellite_coordinates[0])
        return cost

    morning_total_cost = sum(evaluate_route(route, 540) for route in morning_customer_id)
    afternoon_total_cost = sum(evaluate_route(route, 720) for route in afternoon_customer_id)

    return [len(morning_customer_id), round(morning_total_cost, 0)], [len(afternoon_customer_id), round(afternoon_total_cost, 0)]

# 有序交配操作（OX）

def cx_ordered_vrp(parent1, parent2):
    ind1, ind2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    size = min(len(ind1), len(ind2))
    a, b = sorted(random.sample(range(size), 2))

    def ordered_crossover(source, target, a, b):
        hole = [item for item in target if item not in source[a:b+1]]
        return hole[:a] + source[a:b+1] + hole[a:]

    child1 = ordered_crossover(ind1, ind2, a, b)
    child2 = ordered_crossover(ind2, ind1, a, b)
    return child1, child2

# 反轉突變操作

def mutation(ind1, mut_prob):
    if len(ind1) > 1 and random.random() <= mut_prob:
        ind2 = copy.deepcopy(ind1)
        a, b = sorted(random.sample(range(len(ind2)), 2))
        ind2[a:b+1] = reversed(ind2[a:b+1])
        return ind2
    return ind1


def exchange(ind1, se_vehicle_capacity, instance):
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instance)
    if route:
        r = random.randrange(len(route))
        if len(route[r]) > 1:
            a, b = random.sample(range(len(route[r])), 2)
            route[r][a], route[r][b] = route[r][b], route[r][a]
        return [node for subroute in route for node in subroute]
    return ind

# 2-opt 路徑內部優化

def opt2(ind1, se_vehicle_capacity, instance):
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instance)
    if route:
        r = random.randrange(len(route))
        if len(route[r]) > 1:
            a, b = sorted(random.sample(range(len(route[r])), 2))
            re = route[r][:a] + list(reversed(route[r][a:b+1])) + route[r][b+1:]
        else:
            re = route[r]
        return [node for i, subroute in enumerate(route) for node in (re if i == r else subroute)]
    return ind

# 節點重新插入操作

def relocate(ind1, se_vehicle_capacity, instance):
    ind = copy.deepcopy(ind1)
    route = update_route(ind, se_vehicle_capacity, instance)
    if route:
        r = random.randrange(len(route))
        if len(route[r]) > 1:
            a, b = random.sample(range(len(route[r])), 2)
            segment = route[r][:b] + route[r][b+1:]
            segment.insert(a, route[r][b])
        else:
            segment = route[r]
        return [node for i, subroute in enumerate(route) for node in (segment if i == r else subroute)]
    return ind

def update_route(individual, se_vehicle_capacity, instance):
    routes, route, capacity = [], [], 0
    for node in individual:
        customer = next(ins for ins in instance if ins[0] == node)
        demand = customer[5] + customer[6]
        if capacity + demand > se_vehicle_capacity:
            routes.append(route)
            route, capacity = [node], demand
        else:
            route.append(node)
            capacity += demand
    if route:
        routes.append(route)
    return routes

# 評估單一衛星適應度

def satellite_calculate_fitness(individual, speed, capacity, instance, time_vehicles, satellite):
    total_cost = 0
    routes = update_route(individual, capacity, instance)
    for route in routes:
        time_now, cost = time_vehicles, 0
        for idx, node in enumerate(route):
            current = next(ins for ins in instance if ins[0] == node)
            if idx == 0:
                dist = calculate_distance(satellite[1], satellite[0], current[2], current[1])
            else:
                prev = next(ins for ins in instance if ins[0] == route[idx - 1])
                dist = calculate_distance(prev[2], prev[1], current[2], current[1])
            time_now, cost = calculate_time(time_now, current, dist, speed, cost)
        last = next(ins for ins in instance if ins[0] == route[-1])
        cost += calculate_distance(last[2], last[1], satellite[1], satellite[0])
        total_cost += cost
    return [len(routes), round(total_cost, 0)]

# 判斷個體是否重複

def find_same(all_individuals, candidate):
    return not any(existing[0] == candidate for existing in all_individuals)

# 非支配排序（NSGA-II）

def non_dominated_sorting(records):
    s, n, rank = {}, {}, {}
    front = {0: []}
    for p in range(len(records)):
        s[p], n[p] = [], 0
        for q in range(len(records)):
            if dominates(records[p][1], records[q][1]):
                s[p].append(q)
            elif dominates(records[q][1], records[p][1]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)
    i = 0
    while front[i]:
        next_front = []
        for p in front[i]:
            for q in s[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        front[i] = next_front
    del front[i]  # remove last empty front
    return front

def dominates(obj1, obj2):
    return ((obj1[0] < obj2[0] and obj1[1] <= obj2[1]) or
            (obj1[0] <= obj2[0] and obj1[1] < obj2[1]))

# 擁擠度距離計算（Crowding Distance）

def calculate_crowding_distance(front, records):
    distance = {i: 0 for i in front}
    for m in range(2):  # 針對兩個目標值
        obj = {i: records[i][1][m] for i in front}
        sorted_f = sorted(obj, key=obj.get)
        distance[sorted_f[0]] = distance[sorted_f[-1]] = float('inf')
        obj_min, obj_max = obj[sorted_f[0]], obj[sorted_f[-1]]
        if obj_max > obj_min:
            for j in range(1, len(sorted_f) - 1):
                distance[sorted_f[j]] += (obj[sorted_f[j + 1]] - obj[sorted_f[j - 1]]) / (obj_max - obj_min)
    return distance

# 選擇下一代族群（Selection）

def selection(population_size, front, records, best=False):
    if best:
        return [records[i] for i in front]

    new_indices, n = [], 0
    for f in front.values():  # ✅ 修正這行
        if n + len(f) > population_size:
            crowd_dist = calculate_crowding_distance(f, records)
            sorted_f = sorted(f, key=lambda x: crowd_dist[x], reverse=True)
            for i in sorted_f:
                if len(new_indices) == population_size:
                    break
                new_indices.append(i)
            break
        else:
            new_indices.extend(f)
            n += len(f)

    return [records[i] for i in new_indices], new_indices

class NSGAAlgorithm:
    def __init__(self):
        self.start_customer_number = 0
        self.end_customer_number = 99
        self.json_instance = []
        self.crossover_probability = 0.85
        self.mut_prob = 0.1
        self.num_gen = 10
        self.pop_size = 20
        self.best_pop = 1
        self.fe_vehicle_capacity = 1000
        self.se_vehicle_capacity = 200
        self.se_vehicle_speed = 50
        self.depot = [["d1", 120.373731, 36.185609], ["d2", 118.054927, 36.813487], ["d3", 116.897877, 36.611274]]
        self.number_satellite = 2
        self.centers, self.customer_group = None, None
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
        arr = [[cell.value for cell in row if cell.value is not None] for row in s1]
        arr = arr[1:]  # 移除標題列
        self.json_instance = arr[self.start_customer_number:self.end_customer_number]
        self.centers, self.customer_group = calculate_satellite_coordinates(self.json_instance, self.number_satellite)

    def filter_customers_in_china(self):
        df = pd.read_excel("./data/customer_data2.xlsx")
        geometry = [Point(xy) for xy in zip(df['經度'], df['緯度'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        world = gpd.read_file("./data/ne_110m_admin_0_countries.shp")
        china = world[world['NAME'] == 'China']
        in_china = gdf[gdf.within(china.geometry.iloc[0])]
        in_china.to_excel("filtered_customer_data_precise.xlsx", index=False)

    def show_customer_on_china_map(self):
        customer_lons = [c[1] + np.random.uniform(-0.02, 0.02) for c in self.json_instance]
        customer_lats = [c[2] + np.random.uniform(-0.02, 0.02) for c in self.json_instance]
        depot_lons = [d[1] for d in self.depot]
        depot_lats = [d[2] for d in self.depot]

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([110.8, 127.6, 21.2, 49.6], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.LAKES, facecolor='white', edgecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.RIVERS, edgecolor='lightblue', linewidth=0.4)

        ax.scatter(customer_lons, customer_lats, color='black', marker='v', s=20, alpha=0.5, label='Customer', transform=ccrs.PlateCarree())
        for i, (lon, lat) in enumerate(zip(depot_lons, depot_lats)):
            label = 'Depot' if i == 0 else None
            ax.scatter([lon], [lat], color='red', marker='+', s=100, label=label, transform=ccrs.PlateCarree())

        ax.legend()
        plt.show()

    def initial_solution(self):
        for i in range(self.number_satellite):
            group = [self.json_instance[j] for j in range(len(self.customer_group)) if int(self.customer_group[j]) == i]
            morning, morning_id, afternoon, afternoon_id = distinguish_between_periods(group)
            morning.insert(0, self.centers[i])
            afternoon.insert(0, self.centers[i])
            m_dist, a_dist = get_distance(morning, afternoon)
            m_save, a_save = savings_algorithms(m_dist, a_dist, morning_id, afternoon_id)
            m_routes, a_routes = get_routes(self.json_instance, m_save, a_save, morning_id, afternoon_id, self.se_vehicle_capacity)
            m_fit, a_fit = eval_individual_fitness(self.se_vehicle_speed, m_routes, a_routes, self.json_instance, self.centers[i])

            flat_morning = [node for route in m_routes for node in route]
            flat_afternoon = [node for route in a_routes for node in route]

            if i == 0:
                self.all_morning_customer_id_and_fitness1.append([flat_morning, m_fit])
                self.all_afternoon_customer_id_and_fitness1.append([flat_afternoon, a_fit])
            elif i == 1:
                self.all_morning_customer_id_and_fitness2.append([flat_morning, m_fit])
                self.all_afternoon_customer_id_and_fitness2.append([flat_afternoon, a_fit])

    def _expand_population(self, group, satellite_idx, start_time):
        while len(group) < self.pop_size:
            for i in range(len(group)):
                for operator in [exchange, opt2, relocate]:
                    new_ind = operator(group[i][0], self.se_vehicle_capacity, self.json_instance)
                    if find_same(group, new_ind):
                        fit = satellite_calculate_fitness(new_ind, self.se_vehicle_speed, self.se_vehicle_capacity,
                                                          self.json_instance, time_vehicles=start_time,
                                                          satellite=self.centers[satellite_idx])
                        group.append([new_ind, fit])
        return group

    def initial_population(self):
        if len(self.all_morning_customer_id_and_fitness1[0][0]) > 5:
            self.all_morning_customer_id_and_fitness1 = self._expand_population(self.all_morning_customer_id_and_fitness1, 0, 540)
        if len(self.all_morning_customer_id_and_fitness2[0][0]) > 5:
            self.all_morning_customer_id_and_fitness2 = self._expand_population(self.all_morning_customer_id_and_fitness2, 1, 540)
        if len(self.all_afternoon_customer_id_and_fitness1[0][0]) > 5:
            self.all_afternoon_customer_id_and_fitness1 = self._expand_population(self.all_afternoon_customer_id_and_fitness1, 0, 720)
        if len(self.all_afternoon_customer_id_and_fitness2[0][0]) > 5:
            self.all_afternoon_customer_id_and_fitness2 = self._expand_population(self.all_afternoon_customer_id_and_fitness2, 1, 720)

    def _local_search(self, group, satellite_idx, time_vehicles):
        for i in range(len(group)):
            base_fitness = satellite_calculate_fitness(group[i][0], self.se_vehicle_speed, self.se_vehicle_capacity,
                                                       self.json_instance, time_vehicles, self.centers[satellite_idx])
            for operator in [exchange, opt2, relocate]:
                new_ind = operator(group[i][0], self.se_vehicle_capacity, self.json_instance)
                if find_same(group, new_ind):
                    new_fit = satellite_calculate_fitness(new_ind, self.se_vehicle_speed, self.se_vehicle_capacity,
                                                          self.json_instance, time_vehicles, self.centers[satellite_idx])
                    if base_fitness[1] > new_fit[1]:
                        group.append([new_ind, new_fit])
                        del group[i]
                        return

    def _crossover_population(self, group, satellite_idx, time_vehicles):
        if len(group[0][0]) > 5 and self.crossover_probability >= random.random():
            while len(group) < 2 * self.pop_size:
                for i in range(0, len(group), 2):
                    if i + 1 < len(group):
                        ind1, ind2 = cx_ordered_vrp(group[i][0], group[i + 1][0])
                        for offspring in [ind1, ind2]:
                            if find_same(group, offspring):
                                fit = satellite_calculate_fitness(offspring, self.se_vehicle_speed, self.se_vehicle_capacity,
                                                                 self.json_instance, time_vehicles, self.centers[satellite_idx])
                                group.append([offspring, fit])

    def _mutate_population(self, group, satellite_idx, time_vehicles):
        if len(group[0][0]) > 4:
            for i in group:
                mutated = mutation(i[0], self.mut_prob)
                if find_same(group, mutated):
                    fit = satellite_calculate_fitness(mutated, self.se_vehicle_speed, self.se_vehicle_capacity,
                                                     self.json_instance, time_vehicles, self.centers[satellite_idx])
                    group.append([mutated, fit])
                    break

    def _evolve_group(self, group, satellite_idx, time_vehicles):
        self._local_search(group, satellite_idx, time_vehicles)
        self._crossover_population(group, satellite_idx, time_vehicles)
        self._mutate_population(group, satellite_idx, time_vehicles)
        front = non_dominated_sorting(group)
        selected, _ = selection(self.pop_size, front, group)
        return selected

    def runGenerations(self):
        for _ in range(self.num_gen):
            self.all_morning_customer_id_and_fitness1 = self._evolve_group(self.all_morning_customer_id_and_fitness1, 0, 540)
            self.all_morning_customer_id_and_fitness2 = self._evolve_group(self.all_morning_customer_id_and_fitness2, 1, 540)
            self.all_afternoon_customer_id_and_fitness1 = self._evolve_group(self.all_afternoon_customer_id_and_fitness1, 0, 720)
            self.all_afternoon_customer_id_and_fitness2 = self._evolve_group(self.all_afternoon_customer_id_and_fitness2, 1, 720)
            total_score = (self.all_morning_customer_id_and_fitness1[0][1][1] +
                           self.all_morning_customer_id_and_fitness2[0][1][1] +
                           self.all_afternoon_customer_id_and_fitness1[0][1][1] +
                           self.all_afternoon_customer_id_and_fitness2[0][1][1])
            self.score_history.append(total_score)

    def result(self):
        
        front = non_dominated_sorting(self.all_morning_customer_id_and_fitness1)
        best_all_morning_customer_id_and_fitness1=selection(self.pop_size,front[0],self.all_morning_customer_id_and_fitness1 , best=True)

        front = non_dominated_sorting(self.all_morning_customer_id_and_fitness2)
        best_all_morning_customer_id_and_fitness2=selection(self.pop_size,front[0],self.all_morning_customer_id_and_fitness2 , best=True)

        front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness1)
        best_all_afternoon_customer_id_and_fitness1=selection(self.pop_size,front[0],self.all_afternoon_customer_id_and_fitness1 , best=True)

        front = non_dominated_sorting(self.all_afternoon_customer_id_and_fitness2)
        best_all_afternoon_customer_id_and_fitness2=selection(self.pop_size,front[0],self.all_afternoon_customer_id_and_fitness2 , best=True)
        self.all_number_vehicles = sum([best_all_morning_customer_id_and_fitness1[0][1][0] , best_all_afternoon_customer_id_and_fitness1[0][1][0] , best_all_morning_customer_id_and_fitness2[0][1][0]  , best_all_afternoon_customer_id_and_fitness2[0][1][0]])
        self.SE_all_cost = sum([best_all_morning_customer_id_and_fitness1[0][1][1] , best_all_afternoon_customer_id_and_fitness1[0][1][1] , best_all_morning_customer_id_and_fitness2[0][1][1]  , best_all_afternoon_customer_id_and_fitness2[0][1][1]])
        self.SE_all_cost += self.all_number_vehicles * 50 + self.number_satellite * 1000


    def First_route(self):
        model = FE_gurobi()
        self.FE_all_cost = model.main(self.depot, self.number_satellite, self.centers, [1000, 1000], self.fe_vehicle_capacity)

    def Computation_time(self):
        self.all_time = time.time() - self.start_time

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
    start, end, M , C= 0, 99, 0.0 , 0.9
    for day in range(30):
        crossover_stats = []
        for i in range(5):
            costs, vehicles = [], []
            for _ in range(10):
                model = NSGAAlgorithm()
                model.start_customer_number = start
                model.end_customer_number = end
                model.crossover_probability = C
                model.mut_prob = M
                model.day = day + 1
                model.runMain()
                total_cost = model.FE_all_cost + model.SE_all_cost
                costs.append(total_cost)
                vehicles.append(model.all_number_vehicles)
                print(f"總成本：{model.SE_all_cost} 總車輛數：{model.all_number_vehicles}")

            avg_cost = np.mean(costs)
            crossover_stats.append((C, avg_cost))
            print(f"第{day + 1}天 第{i + 1}組")
            print(f"交配率：{C} 突變率：{M}")
            print(f"平均值：{avg_cost} 最小值：{np.min(costs)}")
            C = round(C - 0.05, 2)

        best_C, best_avg = min(crossover_stats, key=lambda x: x[1])
        print(f"====== 第{day + 1}天最佳交配率 ======")
        print(f"最佳交配率：{best_C}  對應平均總成本：{best_avg}")
        print("===================================")
        C, start, end = 0.9, start + 100, end + 100