\ Model LinearProblem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.00533333333333333 y11 + 0.00455555555555556 y12 + 0.0103703703703704 y21
   + 0.0111851851851852 y22
Subject To
 R0: 0.012 y11 >= 1e-06
 R1: 0.1639 y12 >= 1e-06
 R2: 0.0685714285714286 y21 >= 1e-06
 R3: 0.1135714285714286 y22 >= 1e-06
 R4: 2 y11 + 2 y12 <= 2
 R5: 3 y11 + 3 y12 <= 3
 R6: 2 y21 + 2 y22 <= 2
 R7: 2 y21 + 2 y22 <= 2
 R8: 4 y11 + 4 y12 + 2 y21 + 2 y22 <= 6
 R9: 5 y11 + 5 y12 + 3 y21 + 3 y22 <= 9
 R10: y11 + y12 >= 1
 R11: y11 + y12 <= 1
 R12: y21 + y22 >= 1
 R13: y21 + y22 <= 1
Bounds
End
