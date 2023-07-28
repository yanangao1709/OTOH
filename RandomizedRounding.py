from gurobipy import *

F_thr = 0.000001 # integer 存在解
# F_thr = 0.001 #   Linear
r1_capacity = 6
r2_capacity = 9
s1_capacity = 2
d1_capacity = 3
s2_capacity = 2
d2_capacity = 2
# time slot t = 0
def integerRun():
    try:
        m = Model("IntegerProblem")
        y11 = m.addVar(vtype=GRB.BINARY, name='y11')
        y12 = m.addVar(vtype=GRB.BINARY, name='y12')
        y21 = m.addVar(vtype=GRB.BINARY, name='y21')
        y22 = m.addVar(vtype=GRB.BINARY, name='y22')

        m.setObjective((144/27000)*y11 + (123/27000)*y12
                       + (280/27000)*y21 + (302/27000)*y22, GRB.MINIMIZE)

        # fidelity
        m.addConstr((3/250)*y11 >= F_thr)
        m.addConstr((0.1639)*y12 >= F_thr)
        m.addConstr((48/700)*y21 >= F_thr)
        m.addConstr((159/1400)*y22 >= F_thr)
        # node capacity
        m.addConstr(2*y11 + 2*y12 <= s1_capacity)
        m.addConstr(3*y11 + 3*y12 <= d1_capacity)
        m.addConstr(2*y21 + 2*y22 <= s2_capacity)
        m.addConstr(2*y21 + 2*y22 <= d2_capacity)
        m.addConstr(4*y11 + 4*y12 + 2*y21 + 2*y22 <= r1_capacity)
        m.addConstr(5*y11 + 5*y12 + 3*y21 + 3*y22 <= r2_capacity)
        # route selection
        m.addConstr(y11 + y12 >= 1)
        m.addConstr(y11 + y12 <= 1)
        m.addConstr(y21 + y22 >= 1)
        m.addConstr(y21 + y22 <= 1)

        m.write("IntegerProblem.lp")

        m.optimize()

        print('Optimal solution', end=" ")
        for i in m.getVars():
            print('%s = %g' % (i.varName, i.x), end=" ")

    except GurobiError as e:
        print('Error code' + str(e.errno) + ":" + str(e))

    except AttributeError:
        print('Encountered an attribute error')

def linearRun():
    try:
        m = Model("LinearProblem")
        y11 = m.addVar(vtype=GRB.CONTINUOUS, name='y11')
        y12 = m.addVar(vtype=GRB.CONTINUOUS, name='y12')
        y21 = m.addVar(vtype=GRB.CONTINUOUS, name='y21')
        y22 = m.addVar(vtype=GRB.CONTINUOUS, name='y22')

        m.setObjective((144/27000)*y11 + (123/27000)*y12
                       + (280/27000)*y21 + (302/27000)*y22, GRB.MINIMIZE)

        # fidelity
        m.addConstr((3/250)*y11 >= F_thr)
        m.addConstr((0.1639)*y12 >= F_thr)
        m.addConstr((48/700)*y21 >= F_thr)
        m.addConstr((159/1400)*y22 >= F_thr)
        # node capacity
        m.addConstr(2*y11 + 2*y12 <= s1_capacity)
        m.addConstr(3*y11 + 3*y12 <= d1_capacity)
        m.addConstr(2*y21 + 2*y22 <= s2_capacity)
        m.addConstr(2*y21 + 2*y22 <= d2_capacity)
        m.addConstr(4*y11 + 4*y12 + 2*y21 + 2*y22 <= r1_capacity)
        m.addConstr(5*y11 + 5*y12 + 3*y21 + 3*y22 <= r2_capacity)
        # route selection
        m.addConstr(y11 + y12 >= 1)
        m.addConstr(y11 + y12 <= 1)
        m.addConstr(y21 + y22 >= 1)
        m.addConstr(y21 + y22 <= 1)

        m.write("LinearProblem.lp")

        m.optimize()

        print('Optimal solution', end=" ")
        for i in m.getVars():
            print('%s = %g' % (i.varName, i.x), end=" ")

    except GurobiError as e:
        print('Error code' + str(e.errno) + ":" + str(e))

    except AttributeError:
        print('Encountered an attribute error')

# -----------test example-------
def testGorubi():
    try:

        # Create a new model
        m = Model("mip1")

        # Create variables
        x = m.addVar(vtype=GRB.BINARY, name="x")
        y = m.addVar(vtype=GRB.BINARY, name="y")
        z = m.addVar(vtype=GRB.BINARY, name="z")

        # Set objective
        m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

        # Add constraint: x + 2 y + 3 z <= 4
        m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

        # Add constraint: x + y >= 1
        m.addConstr(x + y >= 1, "c1")

        m.optimize()

        for v in m.getVars():
            print(v.varName, v.x)

        print('Obj:', m.objVal)

    except GurobiError:
        print('Error reported')

# integerRun()
linearRun()
