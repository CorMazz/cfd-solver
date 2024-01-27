# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:32:52 2023

@author: 212808354
"""

from sympy import symbols, Eq, solve, Matrix

########################################################################################################################
#%%% Analytically Solve the Slip Boundary Conditions System of Equations
########################################################################################################################
if True:
    # Create the symbols
    u0, v0, u1, v1, S_eta_x, S_eta_y = symbols('u0 v0 u1 v1 S_eta_x S_eta_y')
    
    
    # Define the equations (from topic 24.4 notes equation 5)
    eqn1 = Eq(S_eta_y * u0 - S_eta_x * v0, S_eta_y * u1 - S_eta_x * v1)
    eqn2 = Eq(S_eta_x * u0 + S_eta_y * v0, -S_eta_x * u1 - S_eta_y * v1)
    
    # Solve the system
    
    sol = solve((eqn1, eqn2), (u0, v0))
    
    print(sol)
