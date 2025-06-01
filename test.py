import gurobipy as gp
try:
    m = gp.Model()
    print("Gurobi license is valid.")
except gp.GurobiError as e:
    print("Error:", e)