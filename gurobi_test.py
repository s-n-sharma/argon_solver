import gurobipy as gp
from gurobipy import GRB

try:
    # Create a new model
    m = gp.Model("lp_example")

    # Create variables
    x = m.addVar(name="x")
    y = m.addVar(name="y")

    # Set objective function: Minimize 5x + 4y
    m.setObjective(5*x + 4*y, GRB.MINIMIZE)

    # Add constraints
    m.addConstr(x + y >= 8, "c1")
    m.addConstr(2*x + y >= 10, "c2")
    m.addConstr(x + 4*y >= 11, "c3")

    # Optimize model
    m.optimize()

    # Print solution
    if m.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {m.ObjVal}")
        print(f"x = {x.X}")
        print(f"y = {y.X}")
    elif m.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif m.status == GRB.UNBOUNDED:
        print("Model is unbounded.")

except gp.GurobiError as e:
    print(f"Error code: {e.errno}: {e.message}")

except AttributeError:
    print("Encountered an attribute error")
