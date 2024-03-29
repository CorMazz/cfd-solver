# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:23:55 2023

@author: 212808354
"""

import numpy as np
from os.path import join as osjoin
import matplotlib.pyplot as plt
from numba import njit
from numpy import ndarray



class SolverClass:
    
########################################################################################################################
########################################################################################################################
#%% Initialization Method
########################################################################################################################
########################################################################################################################

    def __init__(self, mesh_class, airfoil_slip: str, MUSCL_kwargs: dict, max_cfl: float = 0.97):
        """Initialize the solution vector. The solution vector is a 4D matrix. 3 dimensions are x, y, and Q. The fourth
        is time. Initially t0 is in Q[0] and t1 is in Q[1].
        
        Airfoil slip should be a string value of 'slip' or 'no-slip'.
        
        MUSCL Kwargs is a dict with keys "epsilon" and "kappa" specifying the order of MUSCL interpolation to use.
            epsilon = 0 --> 1st Order Accurate Upwind
            
            epsilon = 1 --> 2nd or 3rd Order Accurate Depending on Kappa
                kappa = -1: 2nd order "full" upwind
                         0: 2nd order upwind biased (Fromm Scheme)
                         1/3: 3rd order full upwind
                         1/2 3rd order upwind biased (QUICK Scheme)
                         1: 2nd order central
        
        """
        self.mesh = mesh_class
        
        # If the mesh has not been pre-processed
        if not self.mesh.preprocessed:
            raise Exception("The solver must take a mesh that has already been preprocessed")
        
        # Get mesh properties 
        
        # Projected cell face areas (negative signs per the class notes)
        self.S_xsi_x = self.mesh.dvy
        self.S_xsi_y = -self.mesh.dvx
        self.S_eta_x = -self.mesh.dhy
        self.S_eta_y = self.mesh.dhx
        
        # Cell face area magnitudes
        self.S_xsi = self.mesh.vertical_cell_face_lengths
        self.S_eta = self.mesh.horizontal_cell_face_lenghts
        
        # Cell volumes
        self.delta_V = self.mesh.cell_volumes
        
        # Create arrays to store the solution data
        
        # Q_shape = (2 timesteps, cell volume cells, X.shape[1]-1 eta cells, 4 equations to solve)
        Q_shape = (2, self.delta_V.shape[0], self.delta_V.shape[1], 4)
        self.Q = np.zeros(Q_shape)
        
        # Create the Q_v vector, but make it only 1 time step deep
        Q_v_shape = Q_shape[1:]
        self.Q_v = np.zeros(Q_v_shape)
        
        # Flux shape = same shape as S_eta or S_xsi, since there is one area (and one flux across that face) per face
        # The flux is a vector since there are 4 eqns to solve
        self.E_flux = np.zeros(((*self.S_xsi.shape, 4)))
        self.F_flux = np.zeros((*self.S_eta.shape, 4))
        
        # Initialize a matrix to store all of the time steps for each local cell
        self.delta_T = np.zeros_like(self.delta_V)
        
        # Set universal constants
        self.R = 287 # J/kg.K
        self.gamma = 1.4
        self.C_p = 1005 #J/kg.K
        
        # Initialize something to be set by the initial conditions method
        self.Q_ref = None
        self.Q_IC = None
        
        # Initialize an iteration counter and an L_inf norm history
        self.iteration = 0
        self.L_inf_norm_history = []
        self.L2_norm_history = []
        
        # Initialize the boundary conditions specifier
        self.airfoil_slip_setting = airfoil_slip
        
        # Initalize the MUSCL Kwargs Setting
        self.MUSCL_kwargs = MUSCL_kwargs
        
        # Initialize the max cfl setting
        self.max_cfl = max_cfl
        

        
        
########################################################################################################################
########################################################################################################################
#%% Primary Methods
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Initial Conditions
########################################################################################################################

    def set_initial_conditions(self):
        """Automatically set the state vector Q based on a given static pressure of 101325 Pa, T = 300K, M = 2.00,
        which translates to an axial convective velocity of 694.4 m/s since the sound speed is 347.2 m/s.
        
        Using the perfect gas equation of state, p = rho * R * T
        """
        
        p = 101325 # Pa
        T = 300 # K
        M = 2
        
        rho = p / (self.R * T) # kg / m^3
        
        c = np.sqrt(self.gamma*self.R*T) # m/s
        
        u = M * c # m/s
        v = 0 # Define this to be 0 initially
        
        # From Topic 24.1 notes
        q0 = rho 
        q1 = rho*u
        q2 = rho*v
        q3 = p / (self.gamma - 1) + rho*(u**2 + v**2)/2 # From project problem statement section 5
        
        # Make it an array
        self.Q_IC = np.array([q0, q1, q2, q3])
        
        self.Q[:,:,:,:] = self.Q_IC # Assign initial conditions to the solution
        
        # Modify the velocity components to be the magnitude of velocity and store it as Q_ref
        self.Q_ref = np.array([rho, rho*np.sqrt(u**2 + v**2), rho*np.sqrt(u**2 + v**2), q3])
        
        # Update Q_v
        self.__update_global_Q_v()
        

     
########################################################################################################################
# Boundary Conditions
########################################################################################################################
        
    def apply_all_BCs(self, airfoil_slip: str | None = None):
        """A utility function that applies all boundary conditions, including inlet, outlet, and walls.
        Since the boundary conditions all act on the next time step, roll_Q is called to see the effects of the 
        boundary conditions in the 0th time index.
        
        The airfoil slip can be either 'slip' or 'no-slip'.
        """
        
        # Apply the boundary conditions
        self.apply_inlet_BCs()
        self.apply_outlet_BCs()
        self.apply_wall_BCs(airfoil_slip = airfoil_slip)
        
        # Roll Q
        self.roll_Q()

########################################################################################################################
# Solve One Time Step
########################################################################################################################

    def solve_one_step(self):
        
        # Ensure the global Q_v vector is updated
        p, u, v, T = self.__update_global_Q_v()
        
        c = np.sqrt(self.gamma * self.R * T)
        
        # Modify the Q vector in place to time march the solution
        solve_one_step(
            Q = self.Q, 
            E_flux = self.E_flux,
            F_flux = self.F_flux,
            c = c,
            S_xsi_x = self.S_xsi_x,
            S_xsi_y  = self.S_xsi_y,
            S_xsi = self.S_xsi,
            S_eta_x = self.S_eta_x,
            S_eta_y = self.S_eta_y,
            S_eta = self.S_eta,
            delta_V = self.delta_V,
            gamma = self.gamma,
            MUSCL_kwargs = self.MUSCL_kwargs,
            max_cfl = self.max_cfl,
            )
        
        # Check the convergence
        self.calculate_residuals()
        
        # Re-enforce the boundary conditions and automatically roll the Q_vector
        self.apply_all_BCs()
        
        # Update the iteration counter
        self.iteration += 1

########################################################################################################################
# Save Solution
########################################################################################################################

    def save_solution(self, file_name: str):
        """Saves the Q vector as a numpy array given a file name ending in .npy"""
        
        # Open it in a context manager to prevent errors
        with open(file_name, 'wb') as file:
            # Save all 3 to the same file, this does not overwrite them, they will be read in order if you call
            # np.load 3 times, see the docs https://numpy.org/doc/stable/reference/generated/numpy.save.html
            np.save(file, self.Q[0])
            np.save(file, self.mesh.X)
            np.save(file, self.mesh.Y)
            
########################################################################################################################
# Load Solution
########################################################################################################################

    def load_solution(self, file_name:str, rust = False):
        """Load a pre-run Q vector as a numpy array, interpolate it to the current mesh, and enforce the desired 
        boundary conditions. At that point the solution is ready to continue solving and the initial conditions do not
        need to be re-initialized."""
        
        # Initial conditions need to be calculated because they are also the inlet conditions
        self.set_initial_conditions()
        
        # Open it in a context manager to prevent errors
        with open(file_name, 'rb') as file:
            new_Q = np.load(file)
            # new_X = np.load(file)
            # new_Y = np.load(file)
        # Check the shape of the loaded solution
        if new_Q.shape == self.Q[1].shape:
            # If the solution fits the mesh exactly, just apply it as the current solution
            self.Q[:] = new_Q

         
        else: # This means I need to interpolate to a new mesh
            # from scipy.interpolate import griddata
            raise ValueError("I didn't figure out how to interpolate one mesh to another and frankly I do not care.")
            # # Sort the X components because they apparently need to be sorted
            
            # # sort_idx = np.argsort(new_X.ravel())
            # # new_X = new_X.ravel()[sort_idx]
            # # new_Y = new_Y.ravel()[sort_idx]
            
            
            # # Calculate the x and y coordinates of the cell centers
            # new_x_centers = 0.5 * (new_X[:-1, :-1] + new_X[1:, 1:])
            # new_y_centers = 0.5 * (new_Y[:-1, :-1] + new_Y[1:, 1:])
            
            # x_centers = 0.5 * (self.mesh.X[:-1, :-1] + self.mesh.X[1:, 1:])
            # y_centers = 0.5 * (self.mesh.Y[:-1, :-1] + self.mesh.Y[1:, 1:])
            
            # # Create an interpolation object
            # self.Q[:] = griddata(
            #     points = (new_X_centers.ravel(), new_Y_centers.ravel()),
            #     values = new_Q[:,:,0].ravel(),
            #     xi = (self.mesh.X.ravel(), self.mesh.Y.ravel()),
            #     method = 'linear',
            #     ).reshape(self.Q[0].shape)
            
            
            # # Call the interpolator on the current mesh and set that as Q
            # self.Q[:,:,:,0] = interp().reshape(self.mesh.X.shape)
            
        # Enforce the boundary conditions
        if not rust:
            self.apply_all_BCs()
            
            

########################################################################################################################
########################################################################################################################
#%% Secondary Methods
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
#%%% Boundary Conditions
########################################################################################################################
########################################################################################################################
        
########################################################################################################################
# Apply Inlet BCs
########################################################################################################################
        
    def apply_inlet_BCs(self):
        """Apply boundary conditions to the t1 section of Q
        
        Per section 5 of the project statement, since the flow is supersonic, characteristics (information) is only
        flowing from right to left. Therefore, the flow at the inlet plane is constant, and set by the inlet conditions
        only (independent of what is happening in our domain)."""
              
        # Set the inlet boundary condition
        self.Q[1,:,0,:] = self.Q_IC 
        
########################################################################################################################
# Apply Outlet BCs
########################################################################################################################
        
    def apply_outlet_BCs(self):
        """Apply boundary conditions to the t1 section of Q
        
        Per section 5 of the project statement, since the flow is supersonic, characteristics (information) is only
        flowing from right to left. I can't fully explain this at the moment."""
        
        # ---- Explain why the outlet BC is how it is
    
        # Set the outlet boundary condition
        self.Q[1,:,-1,:] = self.Q[0,:,-2,:]

########################################################################################################################
# Apply Wall BCs
########################################################################################################################
    
    def apply_wall_BCs(self, airfoil_slip: str | None = None):
        """Apply either adiabatic slip conditions at the airfoil walls, """
        
        # Check if the airfoil slip condition is specified
        
        airfoil_slip = airfoil_slip if airfoil_slip is not None else self.airfoil_slip_setting
        
        # Change the global setting to match this one in case it was overwritten. 
        self.airfoil_slip_setting = airfoil_slip
        
        # Ensure the Q_v vector is updated for calculations
        p, u, v, T = self.__update_global_Q_v()
        
        # Define the indices for the bottom and the top row of non-halo cells (indexing starts from the bottom left)
        # as well as the halo cells on the bottom and top
        row_indices, halo_row_indices = [1, -2], [0, -1]

        # This sets all walls to slip condition                  
        for row, halo_row in zip(row_indices, halo_row_indices):

#-----------------------------------------------------------------------------------------------------------------------
# ---- Determine u and v 
# ----------------------------------------------------------------------------------------------------------------------

            # For an inviscid slip condition
            
            # Retrive the correct row of area vector sizes
            S_eta_x, S_eta_y, = self.S_eta_x[row], self.S_eta_y[row]
                 
            # u1 and v1 are notation from the 24.4 Notes
            u1, v1, = u[row], v[row]
                
            # Apply the inviscid 'slip' wall condition everywhere to get this working 
            # ---- TODO Change BC's here for next step in solver
            
            # u0 and v0 are notation from 24.4 Notes (Eqn 5) and were solved via sympy in a separate .py file.
            u0 = (-S_eta_x**2*u1 - 2*S_eta_x*S_eta_y*v1 + S_eta_y**2*u1)/(S_eta_x**2 + S_eta_y**2) 
            v0 = (S_eta_x**2*v1 - 2*S_eta_x*S_eta_y*u1 - S_eta_y**2*v1)/(S_eta_x**2 + S_eta_y**2)
            
#-----------------------------------------------------------------------------------------------------------------------
# ---- Determine p and T
# ----------------------------------------------------------------------------------------------------------------------
            
            # Per Equation 7 in the 24.4 Notes set the pressure gradient to 0
            p0 = p[row] #p[row] is like p1
            
            # Per eqn 8 if the wall condition is adiabatic the temperature gradient is also 0
            T0 = T[row] # T[row] is like T1
            
#-----------------------------------------------------------------------------------------------------------------------
# ---- Calculate and Update Q Vector Values
# ----------------------------------------------------------------------------------------------------------------------
            
            # Calculate Q0, Q2 and Q3
            q0 = rho = p0 / (self.R * T0) # Ideal Gas Law
            q1, q2 = (rho * u0), (rho* v0)
            q3 = self.__calc_rho_et(p0, rho, u0, v0)
            
            # Apply the BC to the next time step of the Q_vector
            self.Q[1,halo_row,:,:] = np.stack([q0, q1, q2, q3], axis = 1)
            
        # If we define the no_slip condition over the airfoil
        match airfoil_slip:
            case 'slip':
                pass # We don't need to do anything since this is applied by default
            case 'no-slip':
                # Grab only the indices of the halo cells that correspond to the airfoil
                # Those are cells that have X values that correspond to 0 <= x <= 1
                
                # Technically it would be faster if I did this once at initialization but who cares.
                wall_indices = np.argwhere(np.logical_and((self.mesh.X[0,:] >= 0), (self.mesh.X[0,:] <= 1)))
                
                # Now that we have the X node indices that say where the airfoil starts and ends,
                # they correspond directly to the cell indices
                
                # Retrive the correct row of area vector sizes. These are the airfoil walls
                S_eta_x, S_eta_y, = self.S_eta_x[1,wall_indices], self.S_eta_y[1,wall_indices]
                     
                # u1 and v1 are notation from the 24.4 Notes
                u1, v1, = u[1, wall_indices], v[1, wall_indices]
                
                #u0 and v0 are defined by the 24.4 Notes for a viscous no-slip condition
                u0, v0 = -u1, -v1
                
#-----------------------------------------------------------------------------------------------------------------------
# ---- Determine p and T
# ----------------------------------------------------------------------------------------------------------------------
                # Per Equation 7 in the 24.4 Notes set the pressure gradient to 0
                p0 = p[1, wall_indices] #p[row] is like p1
                
                # Per eqn 8 if the wall condition is adiabatic the temperature gradient is also 0
                T0 = T[1, wall_indices] # T[row] is like T1
                

                # Calculate Q0, Q2 and Q3
                q0 = rho = p0 / (self.R * T0) # Ideal Gas Law
                q1, q2 = (rho * u0), (rho* v0)
                q3 = self.__calc_rho_et(p0, rho, u0, v0)
                
                # Apply the BC to the next time step of the Q_vector
                self.Q[1,0,wall_indices,:] = np.stack([q0, q1, q2, q3], axis = -1)
    
            case _:
                print(f"Airfoil slip must be 'slip' or 'no-slip', not {airfoil_slip}")
    
########################################################################################################################
########################################################################################################################
#%%% Solver Methods
########################################################################################################################
######################################################################################################################## 


########################################################################################################################
# Roll Q
########################################################################################################################

    def roll_Q(self):
        """A function to roll Q (2,nx, ny, 4) along the first axis, effectively shifting the next time step which is
        normally located at the 1st index of Q into the 0th position to prepare for solving the next time step."""
        
        # Roll Q once
        self.Q = np.roll(self.Q, 1, axis = 0)


########################################################################################################################
# Check Convergence
########################################################################################################################

    def calculate_residuals(self):
        """A function to calculate the L infinity norm, which is the magnitude of how much the Q_vector changed.
        This is to be checked before rolling Q."""

        # Calculate the change in Q in all interior cells
        delta_Q = (self.Q[1, 1:-1, 1:-1, :] - self.Q[0, 1:-1, 1:-1, :]) / self.Q_ref
        
        # Calculate the L2 norm
        L2_norm = np.sqrt(np.sum(delta_Q**2))
        
        # Calculate the L_inf norm
        L_inf_norm = np.max(np.linalg.norm(delta_Q, axis = -1))
        
        # Store them both
        self.L2_norm_history.append(L2_norm)
        self.L_inf_norm_history.append(L_inf_norm)
        
        return L2_norm, L_inf_norm
        
        

        
        
    
########################################################################################################################
########################################################################################################################
#%%% Utility Methods
########################################################################################################################
########################################################################################################################    
        
########################################################################################################################
# Update Qv
########################################################################################################################

    def __update_global_Q_v(self):
        """A method to update the Q_v vector which contains p, u, v, and T"""
        
        # Get pointers to all of vectors in Q
        data_list = [self.Q[0, :, :, i] for i in range(self.Q.shape[-1])]
        
        rho, u, v, rho_et = data_list[0], data_list[1] / data_list[0], data_list[2] / data_list[0], data_list[3]
        
        p = (rho_et - rho*(u**2 + v**2) / 2 ) * (self.gamma - 1)
        
        # Using ideal gas law
        T = p / (rho * self.R)
        
        self.Q_v = np.array([p, u, v, T])     
        
        return p, u, v, T
    
########################################################################################################################
# Calculate rho_et
########################################################################################################################

    def __calc_rho_et(self, p, rho, u, v):
        """A function to calculate rho*e_t from pressure, density, u, and v."""
        return p / (self.gamma - 1) + rho * (u**2 + v**2) / 2
    
########################################################################################################################
# Compare 
########################################################################################################################

    def compare_to_analytical_soln(
            self,
            coords: tuple | None = None,
            plot_coords: bool = False,
            ):
        """A function to compare the properties after the shock to the analytical weak oblique shock solution for the 
        given conditions. 
        """
        
        from scipy.spatial import KDTree
        import pandas as pd
        
        # Find the index of the right spot in the mesh to grab
        
        # First unwrap the meshgrid
        nodes = np.column_stack((self.mesh.X.flatten(), self.mesh.Y.flatten()))
        
        if coords is None:
            coords = (0.3, 0.1)
        
        # Create the KD Tree
        tree = KDTree(nodes)
        
        # Query the tree
        distance, idx = tree.query(np.asarray(coords))
        
        # Convert the id idx from the KDTree into a 2d index
        idx_2d = np.unravel_index(idx, self.mesh.X.shape)
    
        
        # Ensure the q_v vector is updated
        self.__update_global_Q_v()
        
        # Grab the desired Q values from that point 
        rho2, _, _, _  = self.Q[0,idx_2d[0], idx_2d[1], :]
        p2, u2, v2, T2 = self.Q_v[:, idx_2d[0], idx_2d[1]]
        

        # Plot the cell we grabbed so I know its correct. Do this after grabbing it so that I don't set it to zero
        if plot_coords:
            self.Q[0,idx_2d[0], idx_2d[1], 1:] = 0
            
            self.plot_solution(
                modes = 'p',
                )

        # Now calculate the Mach number and stagnation pressure since we don't have that
        # Get sound speed at that point
        c = np.sqrt(self.gamma*self.R*T2)
        M2 = np.sqrt(u2**2 + v2**2) / c
        
        def calc_stagnation_pressure(p, M, gamma = self.gamma):
            return p * (1 + (gamma-1)/2 * M**2)**(gamma/(gamma-1))
        p_t2 = calc_stagnation_pressure(p2, M2)
        
        # Define the pre-shock state values
        p1, T1, M1 = 101325, 300, 2
        p_t1 = calc_stagnation_pressure(p1, M1)
        rho1 = p1 / (self.R * T1) # kg / m^3

        results_dict = {
            "M2": M2,
            "p2/p1":p2/p1,
            "rho2/rho1":rho2/rho1,
            "T2/T1": T2/T1,
            "p_t2/p_t1":p_t2/p_t1
            }
    
        results_df = pd.DataFrame(results_dict, index = ['Numerical Solution']) #.from_dict(results_dict, orient = 'columns',)
    
        # Calculate percent errors and add them
        
        analytical_solution = np.array([1.641, 1.707, 1.458, 1.170, 0.9846])
        
        results_df.loc["Analytical Solution"] = analytical_solution
        
        results_df.loc["Percent Error"] = (results_df.iloc[0,:].to_numpy() - analytical_solution) / analytical_solution *100
        
    
        return results_df
    
########################################################################################################################
########################################################################################################################
#%% Visualization Methods
########################################################################################################################
########################################################################################################################
    

########################################################################################################################
# Plot Solution
########################################################################################################################

    def plot_solution(
            self, 
            modes: str | list[str, ...],
            save_plots: bool = False,
            plot_save_dir: str | None = None,
            plot_idx_identifier: str | None = None,
            fig_axes: tuple | None = None,
            plot_gridlines: bool = True,
            linewidth: float | None = None,
            color_halos: bool = True,
      ):
        """A wrapper over the self.plot_mesh_contour function to plot various volume centered properties of interest,
        such as q0, q1, q2, q3, rho, u, v, rho*e_t, p (pressure), T (temperature), c (speed of sound), M (Mach number), 
        or all. 
        """
        
        # Make mode a list if it is not already
        modes = modes if isinstance(modes, list) else [modes]
        
        if self.Q is None:
            raise Exception("The solution has not yet been initialized, the 'add_halo_cells' method or preprocess_mesh methods must be run.")
        
#-----------------------------------------------------------------------------------------------------------------------
# ---- Calculate Properties
# ----------------------------------------------------------------------------------------------------------------------
        
        # Get pointers to all of vectors in Q
        data_list = [self.Q[0, :, :, i] for i in range(self.Q.shape[-1])]
        
        # Ensure Q_v vector is updated
        p, u, v, T = self.__update_global_Q_v()
    
        # Speed of sound
        c = np.sqrt(self.gamma * self.R * T)
        
        # Mach number = vector sum of velocity components, divided by speed of sound
        M = np.sqrt(u**2 + v**2) / c
        
        # Add the physical properties like rho, u, v, and rho*e_t to the data list
        data_list.extend(
            [
                data_list[0], # rho
                u, # u
                v, # v
                data_list[3], # rho e_t
                p,
                T,
                c,
                M,
                (self.L2_norm_history, self.L_inf_norm_history)
            ]
        )
        
#-----------------------------------------------------------------------------------------------------------------------
# ---- Specify Labels and Units
# ----------------------------------------------------------------------------------------------------------------------
        
        # Create a list of labels which modes entries should match
        data_list_labels = ["q0", "q1", "q2", "q3", "rho", "u", "v", "rho*e_t", "p", "T", "c", "M", "R"]
        
        data_list_units = [
            r"$kg/m^3$", # q0
            r"$kg/(m^2s)$", # q1
            r"$kg/(m^2s)$", # q2
            r"Jkg/$m^3$", # q3
            r"$kg/m^3$", # rho
            r"$m/s$", # u
            r"$m/s$", # v
            r"Jkg/$m^3$", # rho*e_t,
            r"$Pa$", # p
            r"$K$", # T 
            r"$m/s$", # c
            r"$Mach$ $Number$", # mach number
            ]
        
        titles = [
            "q0",
            "q1", 
            "q2", 
            "q3", 
            r"$\rho$",
            "u",
            "v", 
            r"$\rho$$e_t$",
            r"$Pressure$",
            r"$Temperature$",
            r"$Speed$ $of$ $Sound$",
            r"$Mach$ $Number$",
            ]

#-----------------------------------------------------------------------------------------------------------------------
# ---- Select What to Plot
# ----------------------------------------------------------------------------------------------------------------------
        
        # Initialize an empty list
        data_list_desired_indices_list = []
        for mode in modes:
            
            if mode == "all":
                
                # If all are desired, create a list of all the indices
                data_list_desired_indices_list = list(range(len(data_list)))
                break
            # Otherwise, add the index of that specific label
            try:
                data_list_desired_indices_list.append(data_list_labels.index(mode))
            except ValueError:
                print(f"{mode} not identified as a valid plotting parameter, skipped.")

        # Now that I know what I want to plot, plot it

#-----------------------------------------------------------------------------------------------------------------------
# ---- Plot Specified Items
# ----------------------------------------------------------------------------------------------------------------------
    
        # If no figure/axes list was provided, create a generator of new figures and new axes, each one separate
        fig_axes = (plt.subplots() for _ in range(len(data_list_desired_indices_list))) if fig_axes is None else fig_axes

        for data_idx, fig_ax in zip(data_list_desired_indices_list, fig_axes):
            
            # this will be used to identify what time step is being plotted
            file_name_prefix = f"{plot_idx_identifier}_" if plot_idx_identifier is not None else '' 
            
            # Generate the identifying part of the file name
            file_name_body = data_list_labels[data_idx].replace("*","_") + ".png"
            
            # Put em together
            file_name = file_name_prefix + file_name_body
            
            # Determine if a plot_save_dir was specified, otherwise it automatically uses the CWD
            plot_save_dir = "" if plot_save_dir is None else plot_save_dir
            
            # Create the save path for that plot, but only if save_plots is True
            plot_save_path = osjoin(plot_save_dir, file_name) if save_plots else None
            
            if data_list_labels[data_idx] == "R": # This is a residuals plot and must be handled separately.
                self.plot_residuals(
                    fig_ax = fig_ax)
            else:
                # Generate the 
                
                # Chop the ends off of the data if we're not plotting halo cells
                data = data_list[data_idx] if color_halos else data_list[data_idx][1:-1,1:-1]
                
                self.mesh.plot_mesh_contour(
                    data, 
                    title = titles[data_idx],
                    cbar_units= data_list_units[data_idx],
                    plot_save_path = plot_save_path,
                    fig_ax = fig_ax,
                    plot_gridlines = plot_gridlines,
                    linewidth = linewidth,
                    color_halos = color_halos
                    )
                
        return fig_axes
    
########################################################################################################################
# Plot Summary
########################################################################################################################

    def plot_summary(
        self, 
        modes: str | list[str, ...] = ['p', 'u', 'v', 'T','M', 'R'],
        layout: tuple[int, int] = (2, 3),
        save_fig: bool = False,
        plot_save_dir: str | None = None,
        plot_idx_identifier: str | None = None,
        figsize: tuple | None = None,
        plot_gridlines: bool = True,
        linewidth: float | None = None,
        color_halos: bool = True,
        figure_title: str | None = None,
    ):
        """A function that creates one figure with multiple plots for easy visualization of the solution."""
        
        # Make mode a list if it is not already
        modes = modes if isinstance(modes, list) else [modes]
        
        # Create the figures and the axes
        fig, axes = plt.subplots(nrows = layout[0], ncols = layout[1], figsize = figsize)
        
        # These must be formatted such that they are in a list of [(fig, ax), (fig, ax), ...] tuples
        fig_axes = [(fig, ax) for ax in axes.reshape(-1)]
        
        fig_axes = self.plot_solution(
            modes, 
            fig_axes = fig_axes,
            plot_gridlines = plot_gridlines, 
            linewidth = linewidth,
            color_halos = color_halos,
            )
        
        # Grab the figure and axes object back
        fig = fig_axes[0][0]
        axes = fig.get_axes()
        
        # Space out the subplots vertically so that they don't overlap
        
        fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
        # Set the figure title
        if figure_title is not None:
            fig.suptitle(figure_title, fontsize = 24)
        
        if save_fig:
            # this will be used to identify what time step is being plotted
            file_name_prefix = f"{plot_idx_identifier}_" if plot_idx_identifier is not None else '' 
            
            # Generate the identifying part of the file name
            file_name_body = "summary.png"
            
            # Put em together
            file_name = file_name_prefix + file_name_body
            
            # Determine if a plot_save_dir was specified, otherwise it automatically uses the CWD
            plot_save_dir = "" if plot_save_dir is None else plot_save_dir
            
            # Create the save path for that plot
            plot_save_path = osjoin(plot_save_dir, file_name)  

            
            # Save the fig
            fig.savefig(plot_save_path)
   
########################################################################################################################
# Plot Residuals
########################################################################################################################
       
    def plot_residuals(
        self, 
        save_fig: bool = False,
        plot_save_dir: str | None = None,
        plot_idx_identifier: str | None = None,
        fig_ax: tuple | None = None,
    ):
        """A function that creates one figure with multiple plots for easy visualization of the solution."""
        
        # Create the figures and the axes or grab them if they exist
        fig, ax = plt.subplots if fig_ax is None else fig_ax
        
        ax.plot(self.L2_norm_history, label = r"$L_2$ $Norm$")
        ax.plot(self.L_inf_norm_history, label = r"$L_{\infty}$ $Norm$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Magnitude")
        ax.set_title(r"$L_2$ & $L_{\infty}$ $Norms$")
        ax.set_yscale('log')
        ax.grid()
        ax.legend()
        
        if save_fig:
            # this will be used to identify what time step is being plotted
            file_name_prefix = f"{plot_idx_identifier}_" if plot_idx_identifier is not None else '' 
            
            # Generate the identifying part of the file name
            file_name_body = "summary.png"
            
            # Put em together
            file_name = file_name_prefix + file_name_body
            
            # Determine if a plot_save_dir was specified, otherwise it automatically uses the CWD
            plot_save_dir = "" if plot_save_dir is None else plot_save_dir
            
            # Create the save path for that plot
            plot_save_path = osjoin(plot_save_dir, file_name)  
            
            # Save the fig
            fig.savefig(plot_save_path)
            
        return fig, ax

########################################################################################################################
########################################################################################################################
#%% Jit Compiled Functions Called in Methods
########################################################################################################################
########################################################################################################################
"""These are functions that need to be looped incessantly and would normally be dummy slow in Python, so we're throwing
them outside of the class so that we can use Numba to compile them to C-like speeds. 

*insert Austin Powers "yeaahhhhhh babbbyyyyy"* """        

########################################################################################################################
# Solve One Time Step
########################################################################################################################

def solve_one_step(
        Q: ndarray[(2, ..., ...,4), np.float64], 
        E_flux: ndarray[(...,...), np.float64],
        F_flux: ndarray[(...,...), np.float64],
        c: ndarray[(...,...), np.float64],
        S_xsi_x: ndarray, S_xsi_y: ndarray, S_xsi: ndarray, 
        S_eta_x: ndarray, S_eta_y: ndarray, S_eta: ndarray, delta_V: ndarray, # Area and volume components
        gamma: float,
        MUSCL_kwargs: dict,
        max_cfl: float,
        ):

    # Update the fluxes in both the eta and the xsi directions
    
    flux_matrices = [E_flux, F_flux]
    flux_directions = ['xsi', 'eta']
    area_components_list = [[S_xsi_x, S_xsi_y, S_xsi] , [S_eta_x, S_eta_y, S_eta,]]
    
    # TODO
    # ---- Determine why F_flux rho*vn component is not symmetric
    # Boundary conditions may be incorrect
    
    for flux_matrix, flux_direction, area_components in zip(flux_matrices, flux_directions, area_components_list):
        # Unpack the area components
        S_x, S_y, S = area_components
        
        # Call the update flux function which changes the flux matrix (E_flux or F_flux) in place
        update_flux(
            Q = Q, 
            flux = flux_matrix,
            flux_direction = flux_direction,
            S_x = S_x,
            S_y = S_y,
            S = S,
            delta_V = delta_V,
            gamma = gamma,
            **MUSCL_kwargs)
                
    # Calculate the local delta T for each cell
    delta_T = calculate_local_delta_T(
        Q = Q, 
        c = c, 
        S_xsi_x = S_xsi_x, 
        S_xsi_y = S_xsi_y, 
        S_xsi = S_xsi, 
        S_eta_x = S_eta_x, 
        S_eta_y = S_eta_y, 
        S_eta = S_eta, 
        delta_V = delta_V,
        max_cfl = max_cfl,
        )
    
    # Interpolate the cell face areas to the center of the cell. 
    # _, _, S_xsi_centered, _, _, S_eta_centered = interpolate_cell_area_metrics(
    # S_xsi_x, S_xsi_y, S_xsi, S_eta_x, S_eta_y, S_eta)
    
    # Debugging values, delete me later
    
    # E_diff = np.diff(E_flux, axis = 1)
    # F_diff = np.diff(F_flux, axis = 0)
    
    # E_diff_area = np.diff(E_flux * S_xsi[:,:,None], axis = 1)
    # F_diff_area = np.diff(F_flux * S_eta[:,:,None], axis = 0)
    # # Per Eqn 1 of Topic 24.1 Notes, Calculate Q at the next time step
    # T_over_V = delta_T[:,:,None] / delta_V[:,:,None]
    
    # delta_Q = (
    #         -(delta_T[:,:,None] / delta_V[:,:,None]) * 
    #             (
    #             np.diff(E_flux * S_xsi[:,:,None], axis = 1) +
    #             np.diff(F_flux * S_eta[:,:,None], axis = 0)
    #             )
    #         )
    

    
    Q[1] = (
        Q[0] -
        (delta_T[:,:,None] / delta_V[:,:,None]) * 
            (
            np.diff(E_flux * S_xsi[:,:,None], axis = 1) +
            np.diff(F_flux * S_eta[:,:,None], axis = 0)
            )
        )
    
    # t0 = Q[0]
    # t1 = Q[1]
    
    # t0_rho, t0_u, t0_v, t0_rho_et = (t0[:,:,0],  
    #                                 t0[:,:,1] / t0[:,:,0],
    #                                 t0[:,:,2] / t0[:,:,0],
    #                                 t0[:,:,3])
    
    # t0_p = (t0_rho_et - t0_rho*(t0_u**2 + t0_v**2) / 2 ) * (gamma - 1)
    # t0_T = t0_p / (t0_rho * 287)
    
    # t1_rho, t1_u, t1_v, t1_rho_et = (t1[:,:,0],  
    #                                 t1[:,:,1] / t1[:,:,0],
    #                                 t1[:,:,2] / t1[:,:,0],
    #                                 t1[:,:,3])
    
    # t1_p = (t1_rho_et - t1_rho*(t1_u**2 + t1_v**2) / 2 ) * (gamma - 1)
    # t1_T = t1_p / (t1_rho * 287)
    
    # delta_rho = t1_rho - t0_rho
    # delta_u = t1_u - t0_u
    # delta_v = t1_v - t0_v
    # delta_rho_et = t1_rho_et - t0_rho_et
    # delta_p = t1_p - t0_p
    # delta_T = t1_T - t0_T
    # End of debugging values

########################################################################################################################
# Calculate Fluxes
########################################################################################################################

@njit
def update_flux(
        Q: ndarray[(2, ..., ...,4), np.float64], 
        flux: ndarray[(...,...), np.float64],
        flux_direction: str,
        S_x, S_y, S, delta_V, # Area and volume components, can be either xsi or eta
        gamma: float,
        epsilon: int,
        kappa: float,
        flux_limiter_type: str,
        ) -> ndarray[(...,...), np.float64]:
        """A function to calculate the flux over a single face. The flux matrix will be padded with zeros since those
        correspond to halo cell fluxes that we don't care about. I'm leaving them there for ease of indexing.
        
        This updates the input vector flux in-place, but also returns it for good measure. """
        
        # See the processed indexed mesh diagram. The (0,0) row and column are all exterior cell boundaries, so we 
        # can skip the 0th index. The 
        for i in range(1, S.shape[0]-1): # Every row, skip the halo cells
            for j in range(1, S.shape[1] - 1): # Every column, skip the halo cells
                
                # Determine Q_left and Q_right for that wall, in either the xsi or the eta 'flux direction'
                Q_LR = interpolate_Q_LR(
                    Q, 
                    i, 
                    j, 
                    mode = flux_direction, 
                    epsilon = epsilon, 
                    kappa = kappa, 
                    limiter_type = flux_limiter_type
                )
        
                # Calculate Roe Averages of Properties (xsi direction)
                RA_xsi = calc_rho_averages(Q_LR, gamma)
    
                # Calculate the delta_V_bar (the average cell volume of the two adjacent cells to the wall)
                

    
                # Calculate Diagonalized A at the Wall
                L, lambda_ij, R = calc_diagonalized_A(
                    rho = RA_xsi[0],  # Primitive components (rho_bar, u_bar, v_bar, h_t_bar, c_bar)
                    u = RA_xsi[1],
                    v = RA_xsi[2],
                    h_t = RA_xsi[3],
                    c = RA_xsi[4],
                    S_x = S_x[i, j],
                    S_y = S_y[i, j],
                    S = S[i, j], # Area components for the given direction
                    gamma = gamma,
                    )
                
                # The delta_V ends up cancelling out so I don't even need to calculate it
                
                # if flux_direction == 'xsi':
                #     delta_V_bar = (delta_V[i, j] + delta_V[i, j - 1]) / 2
                # elif flux_direction == 'eta':
                #     delta_V_bar = (delta_V[i, j] + delta_V[i - 1, j]) / 2
                
                
                # Take the absolute value of the eigenvectors and recombine A
                A_bar = R @ np.abs(lambda_ij) @ L # / delta_V_bar
    
                # Calculate Q_hat from the given Q vectors
                # Q_hat_LR = Q_LR * delta_V_bar 
    
        # Calculate E or F (fv for flux vector) based on the Left and Right Q vectors
                fv_Q_LR = calc_flux_vector_of_Q(
                    Q_LR = Q_LR,
                    S_x = S_x[i,j],
                    S_y = S_y[i,j],
                    S = S[i,j], # Area components
                    gamma = gamma,
                )
                
                # Per Eqn 8 of the Topic 25 Notes
                flux[i,j] = (fv_Q_LR[:, 0] + fv_Q_LR[:, 1]) / 2 - A_bar @ (Q_LR[:, 1] - Q_LR[:, 0]) / 2
                
        return flux

########################################################################################################################
# Interpolate Q Left
########################################################################################################################
    
@njit   
def interpolate_Q_LR(Q: ndarray, i: int, j: int, mode: str, epsilon: int, kappa: float, limiter_type: str) -> ndarray[(4,2)]:
    """A function to interpolate the Q vector to the left and right states of a given wall, vertical or horizontal.
    The indices i and j here correspond to what wall is being looked at. Returns Q_L and Q_R, which are (4) shaped 
    numpy arrays containing the values within the Q-vector interpolated to the left and right side of the given wall.
    Does this in either the 'xsi' or the 'eta' direction, selected with a 'mode' string.
    
    Can have higher order accuracy if epsilon, kappa, and a limiter_type are selected.
    
    Available limiters are currently:
        'mc' or "monotonized central"
        'minmod'
        "van Leer"
    """
    # See the Topic 25 Notes Eqn 11 & 12
    
    # Grab the correct row or column as a Q array
    if mode == "xsi":
        Q_vec = Q[0, i, :, :] # Now rows of Q increasing == left to right 
        k = j # Select the proper indexer
    if mode == "eta":
        Q_vec = Q[0, :, j, :]
        k = i
        
    # Simple first order interpolation done if we decide to, or if we're in one of the internal boundary cells
    if epsilon == 0 or k == 1 or k == Q_vec.shape[0]-1:            
        Q_L = Q_vec[k-1] # Take the values of Q at the cell just to the left
        Q_R = Q_vec[k] # Take the values of Q at the current cell
        
    elif epsilon == 1:
    
        # Monotonized Central
        if limiter_type.lower() == 'mc' or limiter_type.lower() == "monotonized central":
            phi = lambda r: np.maximum(0, np.minimum(2*r, np.minimum(0.5*(1+r), 2) ) )
        # MinMod
        elif limiter_type.lower() == 'minmod':
            phi = lambda r: np.maximum(0, np.minimum(1, r) )
        elif limiter_type.lower() == 'no flux limiter':
            phi = lambda r: np.ones_like(r)
        elif limiter_type.lower() == "van leer":
            phi = lambda r: (r + np.abs(r)) / (1 + np.abs(r))
        else:
            raise ValueError("f{limiter_type} is an unknown flux limiter type. See documentation for options.")
        
        # Calculate r
        # ---- Check this if it ever stops working, I manually added an error fudge factor to prevent 0/0 in freestream
        r_L = (Q_vec[k] - Q_vec[k-1] + 0.0001) / (Q_vec[k-1] - Q_vec[k-2] + 0.0001)
        r_R = (Q_vec[k]-  Q_vec[k-1] + 0.0001) / (Q_vec[k+1] - Q_vec[k] + 0.0001)
        
        # Per lecture 25 Eqn 11
        
        Q_L = (
            Q_vec[k-1] 
                + (1/4) 
                * (
                    (1-kappa)*(Q_vec[k-1]-Q_vec[k-2])*phi(r_L) 
                    + (1+kappa)*(Q_vec[k] - Q_vec[k-1])*phi(1/r_L)
                  )
        )
        Q_R = (
            Q_vec[k] 
                - (1/4)
                * (
                    (1+kappa)*(Q_vec[k]-Q_vec[k-1])*phi(1/r_R)
                    + (1-kappa)*(Q_vec[k+1] - Q_vec[k])*phi(r_R)
                  )
        )

       
    else:  
        raise ValueError("Invalid epsilon parameter specified for MUSCL Interpolation.")
        
    Q_LR = np.column_stack((Q_L, Q_R))
    
    return Q_LR

########################################################################################################################
# Calculate Diagonalized AE
########################################################################################################################

@njit
def calc_diagonalized_A(
    rho: float, u:float, v: float, h_t: float, c: float, # Primitive components
    S_x: float, S_y: float, S: float, # Area components
    gamma: float,
    ) -> tuple[ndarray, ndarray, ndarray]:
    """A function which takes scalar values of rho, u, v, h_t, c, area components, and gamma, and returns a tuple 
    containing L1, lambda_i, R1 as denoted in the Topic 26.1 Notes, where L1 and R1 are the left and right eigenvectors
    respectively, and lambda_i is a diagonal matrix of eigenvalues."""

    # Translate from class notation to the notation of the AIAA-2001-2609 Paper 
    nx = S_x/S
    ny = S_y/S
    v_n = u*nx + v*ny
    h_0 = h_t
    e_k = (u**2 + v**2) / 2
    a = c
 
    # R1 from Topic 26.1 Notes, Delete the 4th Row, 5th Column for 2D
    R1 = np.array([  
        [1.0,              1.0,      1.0,              0.0], # If you don't put the .0 afterwards numba cries
        [u - a*nx,       u,      u + a*nx,       ny],
        [v - a*ny,       v,      v + a*ny,       -nx],
        [h_0 - a*v_n,    e_k,    h_0 + a*v_n,    u*ny - v*nx]
        ])


    
    # Define a common denominator for convenience 
    d0 = 2*a**2
    
    # L1 from Topic 26.1 Notes, Delete the 5th Row, 4th Column for 2D
    L1 = np.array([
        [((gamma-1)*e_k+a*v_n)/d0,   ((1-gamma)*u-a*nx)/d0,      ((1-gamma)*v-a*ny)/d0,  (gamma-1)/d0],
        [(a**2-(gamma-1)*e_k)/a**2,  ((gamma-1)*u)/a**2,         ((gamma-1)*v)/a**2,     (1-gamma)/a**2],
        [((gamma-1)*e_k-a*v_n)/d0,   ((1-gamma)*u + a*nx)/d0,    ((1-gamma)*v+a*ny)/d0,  (gamma-1)/d0],
        [v*nx-u*ny,                  ny,                         -nx,                    0.0],
        ])
    
    # lambda_ij from Topic 26.1 Notes, Delete the 4th or 5th column idk, but they're identical so who cares.
    # Eqn 22 from the notes
    lambda_ij = np.array([v_n-a, v_n, v_n + a, v_n])
    lambda_ij = np.diag(lambda_ij)
    
    # # Split lambda_i into positive and negative parts
    # lambda_ijm = np.minimum(lambda_ij, 0)
    # lambda_ijp = np.maximum(lambda_ij, 0)
    
    return L1, lambda_ij, R1 
       
########################################################################################################################
# Calculate Roe Averaged Values
########################################################################################################################
    
@njit
def calc_rho_averages(Q_LR: ndarray[(4,2)], gamma: float) -> ndarray[(5)]:
    """Calculate the rho averaged values given Q_LR, a tuple containing the left and right values of Q that need to be 
    averaged. The left and right values are numpy arrays of length 4 (corresponding to Q vectors)."""
    
    # Unpack Q_LR
    Q_L, Q_R = Q_LR.T
    
    # Grab the desired primitive values
    rho_l, u_l, v_l, _, h_t_l = calc_primitives(Q_L, gamma = gamma)
    rho_r, u_r, v_r, _, h_t_r = calc_primitives(Q_R, gamma = gamma)
    
    # Per the Topic 23 Notes
    denom = np.sqrt(rho_r) + np.sqrt(rho_l) # Commonly used denominator
    
    u_bar = (np.sqrt(rho_r) * u_r + np.sqrt(rho_l) * u_l) / denom # Eqn 22
    v_bar = (np.sqrt(rho_r) * v_r + np.sqrt(rho_l) * v_l) / denom # Eqn 22
    rho_bar = np.sqrt(rho_r * rho_l) # Eqn 23
    h_t_bar = (np.sqrt(rho_r)*h_t_r + np.sqrt(rho_l)*h_t_l) / denom # Eqn 24
    c_bar = np.sqrt((gamma-1) *(h_t_bar - (u_bar**2 + v_bar**2) / 2)) # Eqn 25
    
    return np.array([rho_bar, u_bar, v_bar, h_t_bar, c_bar])
    
########################################################################################################################
# Calculate Primitive Values
########################################################################################################################

@njit
def calc_primitives(Q: np.ndarray, gamma: float) -> tuple[np.ndarray, ...]:
    """Given a 1 dimensional numpy array (ny, nx, 4) representing the Q vector at every cell, which contains 
    (rho, rho*u, rho*v, rho*e_t), calculate a vector of primitives, u, which is (rho, u, v, p, h_t).
    """
    rho, u, v, e_t = Q[0], Q[1]/Q[0], Q[2]/Q[0], Q[3]/Q[0]
    p = (gamma - 1) * rho * (e_t - (u**2 + v**2)/2)
    
    # PG 12 of Topic 23 Notes
    h_t = e_t + p/rho
    
    return np.array([rho, u, v, p, h_t])

########################################################################################################################
# Calculate E from Q
########################################################################################################################
    
@njit
def calc_flux_vector_of_Q(
        Q_LR:  ndarray[(4,2), np.float64], 
        S_x: float,
        S_y: float, 
        S: float, 
        gamma: float) -> ndarray[(4,2), np.float64]:
    """Given an ndarray containing a left and a right Q vector 4 units long, calculate the E or F vector. Per the Topic 
    26 Notes Pg. 3
    """
    
    # Calculate Nx, Ny
    nx, ny = S_x / S, S_y / S
    
    # Initialize an empty array
    fv_LR = np.zeros_like(Q_LR)
    
    for i, Q in enumerate(Q_LR.T):
    # Unpack Q for ease of use
        q1, q2, q3, q4 = Q
        
        e1 = q2*nx + q3*ny
        e2 = q2**2/q1*nx + q2*q3/q1*ny + (gamma-1)*(q4 - (q2**2/q1 + q3**2/q1)/2)*nx
        e3 = q2*q3/q1*nx + q3**2/q1*ny + (gamma-1)*(q4 - (q2**2/q1 + q3**2/q1)/2)*ny
        e4 = (gamma*q4 - (gamma-1)/2*(q2**2/q1 + q3**2/q1))*(q2/q1*nx + q3/q1*ny)
        
        fv_LR[:, i] = np.array([e1, e2, e3, e4])
    
    return fv_LR
    
########################################################################################################################
# Calculate the Time Step
########################################################################################################################

def calculate_local_delta_T(
    Q: ndarray[(2, ..., ...,4), np.float64],
    c: ndarray, 
    S_xsi_x: ndarray, S_xsi_y: ndarray, S_xsi: ndarray, 
    S_eta_x: ndarray, S_eta_y: ndarray, S_eta: ndarray, # Area and volume components
    delta_V: ndarray,
    max_cfl: float,
    ):
    """A function to calculate the largest delta_t allowable in any given cell per the Topic 24.3 notes."""
    
    # Grab u and v components of velocity
    u, v = Q[0,:,:,1] / Q[0,:,:,0],  Q[0,:,:,2] / Q[0,:,:,0]
    
    S_xsi_x1, S_xsi_y1, S_xsi1, S_eta_x1, S_eta_y1, S_eta1 = interpolate_cell_area_metrics(
    S_xsi_x, S_xsi_y, S_xsi, S_eta_x, S_eta_y, S_eta)
    
    u_xsi = u * S_xsi_x1 / S_xsi1 + v * S_xsi_y1 / S_xsi1
    v_eta = u * S_eta_x1 / S_eta1 + v * S_eta_y1 / S_eta1
    
    xsi_spectral_radius = np.abs(u_xsi) + c
    eta_spectral_radius = np.abs(v_eta) + c
    
    # Per the Topic 26 Notes
    xsi_x, xsi_y = S_xsi_x1 / delta_V, S_xsi_y1 / delta_V
    eta_x, eta_y = S_eta_x1 / delta_V, S_eta_y1 / delta_V
    
    # Per the topic 24.3 notes
    xsi_metric = np.sqrt(xsi_x**2 + xsi_y**2)
    eta_metric = np.sqrt(eta_x**2 + eta_y**2)
    
    # Per the Topic 24.3 Notes
    delta_T = max_cfl * np.minimum(1/(xsi_spectral_radius * xsi_metric), 1/(eta_spectral_radius * eta_metric))
    
    return delta_T
 
########################################################################################################################
# Interpolate Cell Area Metrics
########################################################################################################################   

def interpolate_cell_area_metrics(    
    S_xsi_x: ndarray, S_xsi_y: ndarray, S_xsi: ndarray, 
    S_eta_x: ndarray, S_eta_y: ndarray, S_eta: ndarray):
    """A function to interpolate the cell area metrics to the center of the cells"""
    
    # Interpolate the cell area metrics to the cell centers
    # I know I should've looped this but I got lazy it's 10:30 pm and I'm tired.
    S_xsi_x = (S_xsi_x + np.roll(S_xsi_x, 1, axis = 1) / 2)[:,:-1]
    S_xsi_y = (S_xsi_y + np.roll(S_xsi_y, 1, axis = 1) / 2)[:,:-1]
    S_xsi = (S_xsi + np.roll(S_xsi, 1, axis = 1) / 2)[:,:-1]
    S_eta_x = (S_eta_x + np.roll(S_eta_x, 1, axis = 0) / 2)[:-1,:]
    S_eta_y = (S_eta_y + np.roll(S_eta_y, 1, axis = 0) / 2)[:-1,:]
    S_eta = (S_eta + np.roll(S_eta, 1, axis = 0) / 2)[:-1,:]
    
    return S_xsi_x, S_xsi_y, S_xsi, S_eta_x, S_eta_y, S_eta