import xlsxwriter
import math , random
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import Model,quicksum,GRB
import numpy as np



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
def read_input(filename):
    """
    :param filename: 数据文件路径
    :return:
    """
    df = pd.read_csv(filename)
    x_coord = { df['id'][i]:df['x_coord'][i] for i in range(df.shape[0]) }       # 节点横坐标
    y_coord = { df['id'][i]:df['y_coord'][i] for i in range(df.shape[0]) }       # 节点纵坐标
    demand = { df['id'][i]:df['demand'][i] for i in range(df.shape[0]) }         # 节点需求
    cost = {}
    N = df['id'].tolist()
    for f_n in N:
        for t_n in N:
            dist = math.sqrt( (x_coord[f_n]-x_coord[t_n])**2 + (y_coord[f_n] - y_coord[t_n])**2 )
            cost[f_n,t_n] = dist
    return N,cost,demand,x_coord,y_coord

def build_model(N,K,depot,CAP,cost,demand):
    """
    :param N: 网络节点集合
    :param K: 车队集合
    :param depot: 配送中心id
    :param CAP: 车辆容量
    :param cost: 网络弧费用集合
    :param demand: 网络节点需求集合
    :return:
    """
    cvrp_model = Model('CVRP')
    # cvrp_model.setParam('MIPGap', 0)
    cvrp_model.setParam("TimeLimit" , 7200)
    # 添加变量
    X = cvrp_model.addVars(N, N, K, vtype=GRB.BINARY, name='X[i,j,k]')
    Y = cvrp_model.addVars(N, K, vtype=GRB.BINARY, name='Y[i,k]')
    U = cvrp_model.addVars(N, K, vtype=GRB.INTEGER, name='U[i,k]')
    cvrp_model.update()
    # 设置目标函数
    cvrp_model.setObjective( quicksum(X[i,j,k] * cost[i,j] for i in N for j in N for k in K), GRB.MINIMIZE)
    # 添加约束
    #  需求覆盖约束
    cvrp_model.addConstrs( quicksum(Y[i,k] for k in K) == 1 for i in N[1:] )
    #  车辆启用约束
    cvrp_model.addConstr( quicksum(Y[depot,k] for k in K) == len(K) )
    #  车辆流平衡约束
    cvrp_model.addConstrs( quicksum(X[i,j,k] for j in N ) == quicksum(X[j,i,k] for j in N ) for i in N for k in K )
    #  车辆路径限制
    cvrp_model.addConstrs( quicksum(X[i,j,k] for j in N) == Y[i,k] for i in N for k in K )
    #  车辆容量约束
    cvrp_model.addConstrs( quicksum(Y[i,k] * demand[i] for i in N) <= CAP for k in K )
    #  破圈约束
    cvrp_model.addConstrs( U[i,k] - U[j,k] + CAP * X[i,j,k] <= CAP - demand[i] for i in N[1:] for j in N[1:] for k in K )
    cvrp_model.addConstrs( U[i,k] <=  CAP for i in N[1:] for k in K)
    cvrp_model.addConstrs( U[i,k] >= demand[i] for i in N[1:] for k in K )
    cvrp_model.optimize()
    return cvrp_model,X,Y,U

def draw_routes(route_list,x_coord,y_coord):
    for route in route_list:
        path_x = []
        path_y = []
        for n in route:
            path_x.append(x_coord[n])
            path_y.append(y_coord[n])
        plt.plot(path_x, path_y,linewidth=0.5, marker='s',ms=5)
    plt.show()

def save_file(route_list,total_cost):
    wb = xlsxwriter.Workbook('路径方案.xlsx')
    ws = wb.add_worksheet()
    ws.write(0,0,'总费用')
    ws.write(0,1,total_cost)
    ws.write(1,0,'车辆')
    ws.write(1,1,'路径')
    row = 2
    for route in route_list:
        ws.write(row,0,route[0])
        route_str = [str(i) for i in route[1:]]
        ws.write(row,1,'-'.join(route_str))
        row += 1
    wb.close()

if __name__ == '__main__':
    # filename = './datasets/CVRP/r101-31.csv'
    dataset = [
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r101.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r102.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r103.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r104.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r105.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r106.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r107.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r108.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r109.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r110.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r111.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r1/r112.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r201.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r202.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r203.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r204.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r205.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r206.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r207.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r208.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r209.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r210.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/r2/r211.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc1/rc101.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc1/rc102.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc1/rc103.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc1/rc104.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc1/rc105.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc1/rc106.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc1/rc107.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc1/rc108.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc2/rc201.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc2/rc202.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc2/rc203.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc2/rc204.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc2/rc205.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc2/rc206.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc2/rc207.txt",
        "Capacitated-Vehicle-Routing-Problem-master/test/rc2/rc208.txt"
    ]

                                                                                       #| 客戶20           | #|客戶30          |
    test_num = [26, 14, 23, 19, 25, 5, 11, 2, 17, 36, 4, 1, 8, 6, 33, 0, 37, 16, 21, 35, 22, 24, 15, 10 , 3, 28, 12, 34, 38, 31]
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
    CAP = 200
    if num >= 12 and num <= 22:
        CAP = 1000
    if num >= 31:
        CAP = 1000
    df = load(dataset[num])
    r = data_num_30[tpr-15]
    print(r)
    print(num)
    r.insert(0 , 0)
    N = [n for n in range(len(r))]
    # print(N)
    cost = {}
    demand = {n : df['DEMAND'][r[n]] for n in range(len(r))}
    x_coord = {n : df['XCOORD.'][r[n]] for n in range(len(r))}
    y_coord = {n : df['YCOORD.'][r[n]] for n in range(len(r))}
    
    for f_n in N:
        for t_n in N:
            dist = math.sqrt( (x_coord[f_n]-x_coord[t_n])**2 + (y_coord[f_n] - y_coord[t_n])**2 )
            cost[f_n,t_n] = dist
    depot = N[0]
    K = [n for n in range(10)]
     
    cvrp_model, X, Y, U = build_model(N, K, depot, CAP, cost, demand)
    # print(X ,Y , U)
    # cvrp_model.setParam(GRB.Param.LogFile, './gurobi_r101-31.log')
    
    route_list = []
    for k in K:
        route = [depot]
        cur_node = depot
        cur_k = None
        for j in N[1:]:
            if X[depot, j, k].x > 0:
                cur_node = j
                cur_k = k
                route.append(j)
                N.remove(j)
                break
        if cur_k is None:
            continue
        while cur_node != depot:
            for j in N:
                if X[cur_node, j, cur_k].x > 0:
                    cur_node = j
                    route.append(j)
                    if j != depot:
                        N.remove(j)
                    break
        route_list.append(route)
    print("最优路径距离:", cvrp_model.objVal)
    print("最优路径使用车辆数:", len(route_list))
    draw_routes(route_list, x_coord, y_coord)
    save_file(route_list, cvrp_model.objVal)