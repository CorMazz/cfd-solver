# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:11:00 2023

@author: 212808354


You'll need to set up the proper file structure. Mine looked like this

project**
-code**
--saved_solutions
--solver_class.py
--mesh_class.py
--run_me.py
-grids**
--meshes 0, 1, & 2**
-plots**
--folders will be made here when plots are ran
"""
import numpy as np
from mesh_class import MeshClass
from solver_class import SolverClass
import matplotlib
import os
from tqdm import trange

########################################################################################################################
#%% Determine What Case to Run
########################################################################################################################

MUSCL_scheme_DOE = True
mesh_option = 0 # 0, 1, or 2, which tells you which index in the "mesh file paths" to look in

########################################################################################################################
#%% Define Common Plotting Settings
########################################################################################################################

plot_gridlines = True # Decide if I want the gridlines to plot
linewidth = 0.15 if mesh_option == 2 else 1 # Make the really tight mesh have much smaller gridlines
figsize = (36,12) # A useful common figsize for the summary plots
color_halos = False if mesh_option == 2 else True # Don't fill in the halo cells for the fine mesh
matplotlib.rcParams['savefig.dpi'] = 600 # Raise the savefig dpi for high resolution plots

# Store the kwargs (key word args) in a dict
plot_kwargs = {
    "figsize": figsize,
    "plot_gridlines": plot_gridlines,
    "linewidth": linewidth,
    "color_halos": color_halos
    }


########################################################################################################################
#%% Intialize the Mesh
########################################################################################################################

mesh_file_paths = ["../grids/g33x25u.dat", "../grids/g65x49u.dat", "../grids/g65x65s.dat"]

mesh = MeshClass(file_path = mesh_file_paths[mesh_option])
mesh.preprocess_mesh(plot_metrics = False)


########################################################################################################################
#%% Helper Functions
########################################################################################################################

def mk_plot_dir(dir_name):
    """Make a directory in the ../plots/ folder with the given dir_name if it doesn't already exist."""
    parent_dir = "../plots"
    
    # Join the directory name
    plot_dir = os.path.join(parent_dir, dir_name)
    
    # If it exists, do nothing, else, make it 
    os.makedirs(plot_dir, exist_ok=True)
    
    return plot_dir

#-----------------------------------------------------------------------------------------------------------------------
# Prep For Iteration
#-----------------------------------------------------------------------------------------------------------------------

def prep_for_iterating(
        solver,
        plot_save_dir: str, 
        plot_kwargs: dict, 
        save_fig: bool = True,
        fig_title: str | None = None, 
        load_soln: bool = False,
        saved_soln: str | None = None,
        ):
    """Set the initial conditions and boundary conditions for the solver and plot them to the specified directory."""
    
    print("Preparing for Iteration")
    
    if load_soln:
        if saved_soln is None:
            raise ValueError("saved_soln should be a string locating the saved solution value and must be specified.")
        else:
            solver.load_solution(saved_soln)
            fig_identifier="Loaded Solution"
    else:
        solver.set_initial_conditions()
        solver.plot_summary(
            save_fig=save_fig,
            plot_idx_identifier="ICs",
            plot_save_dir = save_dir,
            figure_title = fig_title,
            **plot_kwargs,
            )

        solver.apply_all_BCs()
        fig_identifier = "IC&BCs"
    solver.plot_summary(
        save_fig=save_fig,
        plot_idx_identifier=fig_identifier,
        plot_save_dir = save_dir,
        figure_title = fig_title,
        **plot_kwargs,
        )
    
#-----------------------------------------------------------------------------------------------------------------------
# Iterate
#-----------------------------------------------------------------------------------------------------------------------
    
def iterate(
        num_plots: int, 
        iterations_per_plot: int, 
        save_dir: str,
        plot_kwargs: dict,
        fig_title: str | None = None, 
        save_fig: bool = True, ):
    """Iterate 'iterations_per_plot iterations, and then save a plot. Do this for num_plots."""
    print("Beginning Iteration")
    for i in range(num_plots):
        for j in trange(iterations_per_plot):
            solver.solve_one_step()
        solver.plot_summary(
            save_fig=save_fig,
            plot_idx_identifier=f"{i+1}",
            plot_save_dir = save_dir,
            figure_title = fig_title,
            **plot_kwargs,
            )
         
########################################################################################################################
#%% Run a MUSCL Scheme DoE
########################################################################################################################
        
"""
Vary all of these MUSCL Kwargs and run all of them.

MUSCL Kwargs is a dict with keys "epsilon" and "kappa" specifying the order of MUSCL interpolation to use.
    epsilon = 0 --> 1st Order Accurate Upwind
    
    epsilon = 1 --> 2nd or 3rd Order Accurate Depending on Kappa
        kappa = -1: 2nd order "full" upwind
                 0: 2nd order upwind biased (Fromm Scheme)
                 1/3: 3rd order full upwind
                 1/2 3rd order upwind biased (QUICK Scheme)
                 1: 2nd order central

"""

if MUSCL_scheme_DOE:
    scheme_settings = [
        # (Scheme name, (epsilon, kappa), flux limiter, Full scheme name for plot title, CFL)
          ("1OAU", (0, 0), 'no flux limiter', "First Order Accurate Upwind", 0.97),
        # ("2OAFU", (1, -1), 'van Leer', '2nd Order "Full" Upwind', 0.5),
        ("2OAFU_low_cfl", (1, -1), 'no flux limiter', '2nd Order "Full" Upwind', 0.2),
        # ("2OAUB_Fromm", (1, 0), 'van Leer', '2nd Order Upwind Biased (Fromm Scheme)', 0.5),
        # ("3OAFU", (1, 1/3), 'minmod', '3rd Order Full Upwind', 0.5),
        ("3OAUB_QUICK_low_cfl", (1, 1/2),  'no flux limiter', '3rd Order Upwind Biased (QUICK Scheme)', 0.01),
        # ("3OAUB_QUICK", (1, 1/2),  'minmod', '3rd Order Upwind Biased (QUICK Scheme)', 0.5),
        # ("3OAUB_QUICK", (1, 1/2),  'van leer', '3rd Order Upwind Biased (QUICK Scheme)', 0.9),
        # ("2OAC", (1, 1), '2nd Order Central'),
        ]
    
    for scheme_dir_name, (epsilon, kappa), flux_limiter_type, plot_title, max_cfl in scheme_settings:
        # Add the flux limiter type to the plot title
        plot_title = f"{plot_title} -- '{flux_limiter_type.title()}' Flux Limiter"
        
        print(f"\nProcessing {plot_title}")
        # Grab the plotting directory for a given scheme
        save_dir = mk_plot_dir(f"{scheme_dir_name.lower()}_{flux_limiter_type.replace(' ', '_')}")
        
        # Create the MUSCL Interpolation settings dict
        
        MUSCL_kwargs = {
            "epsilon": epsilon,
            "kappa": kappa,
            "flux_limiter_type":flux_limiter_type,
            }
        
        # Instantiate the solver with the desired properties
        solver = SolverClass(mesh, airfoil_slip = 'slip', MUSCL_kwargs= MUSCL_kwargs, max_cfl = max_cfl)
        
        # Prep the solver for iteration
        prep_for_iterating(
            solver, 
            save_fig = True, 
            plot_save_dir = save_dir,
            plot_kwargs = plot_kwargs,
            load_soln= True,
            saved_soln = './saved_solutions/2oafu_low_cfl_no_flux_limiter.npy')
        
        # Iterate
        iterate(
            num_plots = 2, 
            iterations_per_plot = 2000, 
            save_fig = True,
            save_dir = save_dir, 
            plot_kwargs = plot_kwargs, 
            fig_title = plot_title,
            )
        
        solver.save_solution(f"./saved_solutions/{scheme_dir_name.lower()}_{flux_limiter_type.replace(' ', '_')}.npy")

########################################################################################################################
#%% Graveyard
########################################################################################################################




# # Plot the first iteration results

# # Grab the plotting directory for a given scheme
# plot_title = "1 Iteration"
# save_dir = mk_plot_dir("/delete_me/for_aman")

# # # Create the MUSCL Interpolation settings dict

# MUSCL_kwargs = {
#     "epsilon": 0,
#     "kappa": 1,
#     "flux_limiter_type": 'no_flux_limiter',
#     }

# # # Instantiate the solver with the desired properties
# solver = SolverClass(mesh, airfoil_slip = 'slip', MUSCL_kwargs= MUSCL_kwargs, max_cfl = 0.97)

# Q = solver.Q[0,:,:,0]

# # import matplotlib.pyplot as plt
# # from matplotlib.colors import ListedColormap

# # white_cm = ListedColormap(['white'])
# # # plt.register_cmap(cmap=white_cm)
# # mesh.plot_mesh_contour(
# #     np.zeros_like(Q), 
# #     plot_save_path = "../plots/65x49_processed_mesh_new.png",
# #     cmap = white_cm,
# #     linewidth = 1)

# # Prep the solver for iteration
# prep_for_iterating(
#     solver, 
#     save_fig = False, 
#     plot_save_dir = save_dir,
#     plot_kwargs = plot_kwargs,
#     load_soln= True,
#     saved_soln = './saved_solutions/3oaub_quick_no_flux_limiter_fully_converged.npy')

# analytical_results = solver.compare_to_analytical_soln(plot_coords = True)


# # Iterate
# iterate(
#     num_plots = 1, 
#     iterations_per_plot = 1, 
#     save_fig = True,
#     save_dir = save_dir, 
#     plot_kwargs = plot_kwargs, 
#     fig_title = plot_title)

# for i in range(4):
#     mesh.plot_mesh_contour(solver.E_flux[1:-1,1:-2, i], title = fr"$E Flux$ {i}", color_halos = False)
#     mesh.plot_mesh_contour(solver.F_flux[1:-2,1:-1, i], title = fr"$F Flux$ {i}", color_halos = False)


# Plot area contours
# solver = SolverClass(mesh, airfoil_slip = 'slip', MUSCL_kwargs={})
# S_xsi_x = solver.S_xsi_x
# S_xsi_y = solver.S_xsi_y
# S_xsi = solver.S_xsi
# S_eta_x = solver.S_eta_x
# S_eta_y = solver.S_eta_y
# S_eta = solver.S_eta
# delta_V = solver.delta_V
# mesh.plot_mesh_contour(solver.S_xsi_x[:,:-1], title = r"$S_{\xi x}$" )
# mesh.plot_mesh_contour(solver.S_xsi_y[:,:-1], title = r"$S_{\xi y}$" )
# mesh.plot_mesh_contour(solver.S_xsi[:,:-1], title = r"$S_{\xi}$" )
# mesh.plot_mesh_contour(solver.S_eta_x[:-1,:], title = r"$S_{\eta x}$" )
# mesh.plot_mesh_contour(solver.S_eta_y[:-1,:], title = r"$S_{\eta y}$" )
# mesh.plot_mesh_contour(solver.S_eta[:-1,:], title = r"$S_{\eta}$" )
# mesh.plot_mesh_contour(solver.delta_V, title = r"$V$" )
