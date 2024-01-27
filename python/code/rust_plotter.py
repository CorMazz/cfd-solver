# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:22:31 2023

@author: 212808354
"""

import numpy as np
from mesh_class import MeshClass
from solver_class import SolverClass


# # Check the flux vectors
# e_flux_path = r"C:\Users\212808354\Box\Misc\Grad School\CFD\project\code\saved_solutions\e_flux.npy" 
# with open(e_flux_path, 'rb') as file:
#     e_flux = np.load(file)
# asdf
########################################################################################################################
#%% Load and Plot the Saved Rust Solution
########################################################################################################################

mesh_file_paths = ["../grids/g33x25u.dat", "../grids/g65x49u.dat", "../grids/g65x65s.dat"]
mesh_option = 1 # 0, 1, or 2, which tells you which index in the "mesh file paths" to look in

mesh = MeshClass(file_path = mesh_file_paths[mesh_option])
mesh.preprocess_mesh(plot_metrics = False)
solver = SolverClass(mesh, airfoil_slip = 'slip', MUSCL_kwargs= None, max_cfl = None)
solver.load_solution("./saved_solutions/rust_soln.npy", rust = True)
Q = solver.Q[0]


solver.plot_summary(
    save_fig=False,
    plot_idx_identifier="ICs",
    plot_save_dir = None,
    figure_title = "Initial Conditions with BC's Applied",
    )