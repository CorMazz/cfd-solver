 # -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:33:40 2023

@author: 212808354
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from os.path import join as osjoin
import os


class MeshClass:
    """A class which reads a structured mesh, adds halo cells, and then calculates 2D mesh metrics."""
    
########################################################################################################################
########################################################################################################################
#%% Initialization Method
########################################################################################################################
########################################################################################################################

    def __init__(self, file_path):
        self.file_path = file_path
        
        # Set by read_mesh() method
        self.X, self.Y = None, None
        
        # Set by calculate_2D_metrics() method
        self.dhx, self.dvx, self.dhy, self.dvy = None, None, None, None
        self.horizontal_cell_face_lenghts, self.vertical_cell_face_lengths = None, None
        
        # Determine if the mesh has already gone through preprocessing
        self.preprocessed = False
        

########################################################################################################################
########################################################################################################################
#%% Combined Methods
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Preprocess Mesh
########################################################################################################################

    def preprocess_mesh(self, plot_metrics = False):
        """A utility method that just calls all the other ones I created.
        
        If you want to save plots, call the individual methods manually.
        """
        
        self.read_mesh()
        self.add_halo_cells()
        self.calculate_2D_metrics(plot = plot_metrics)
        self.preprocessed = True
        
########################################################################################################################
########################################################################################################################
#%% Individual Primary Mesh Methods
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Read Mesh
########################################################################################################################
    
    def read_mesh(self, file_path = None, plot = False, plot_save_path = None):
        """A function to read a structured 2D mesh formatted the way professor Oefelein provided for HW4 and the project.
        
        Takes in a file name (mesh.dat) and returns X and Y, 2D numpy arrays forming a meshgrid.
        Sets self.X and self.Y for future use.
        """
        
        # Determine if the filepath has been specified
        if file_path is None:
            file_path = self.file_path
        
        
        # Initialize empty lists
        x = []
        y = []
        
        # Open the file and loop over every line and retrieve the values
        with open(file_path,"r") as fp:
            [nx, ny] = [int(m) for m in fp.readline().strip("\n").split(", ")]
            for line in fp:
                x.append(line.strip("\n").split(", ")[0])
                y.append(line.strip("\n").split(", ")[1])
        
        # Turn x and y into numpy arrays
        x = np.asarray(x, dtype = np.float64)
        y = np.asarray(y, dtype = np.float64)
        
        # Create a meshgrid of x and y
        
        self.X = x.reshape(-1, nx)
        self.Y = y.reshape(-1, nx)
        
        
        if plot:
            self.__plot_mesh_gridlines(title = 'Default Mesh with Gridlines', save_path = plot_save_path)
            
        return self.X, self.Y
    
########################################################################################################################
# Add Halo Cells
########################################################################################################################
    
    def add_halo_cells(self, plot = False, plot_save_path = None):
        """A function to take a mesh that has been read in and pad it with halo cells. Modifies the self.X and Y arrays
        and returns them. Also initializes the solution storage attribute, self.Q."""
        
        if self.X is None or self.Y is None:
            raise Exception("You must perform the read_mesh() method before adding halo cells.")
        
        # Use the padding function to add halo cells
        self.X = 2*np.pad(self.X, pad_width = 1, mode = "edge") - np.pad(self.X, pad_width = 1, mode = "reflect")
        self.Y = 2*np.pad(self.Y, pad_width = 1, mode = "edge") - np.pad(self.Y, pad_width = 1, mode = "reflect")
        
        if plot:
            self.__plot_mesh_gridlines(title = "Mesh with Halo Cells", save_path = plot_save_path)
            
        # Initialize the data array
        

        
        return self.X, self.Y
    
########################################################################################################################
# Calculate 2D Metrics
########################################################################################################################

    def calculate_2D_metrics(self, plot = False, plots_save_dir = None):
        """A function which calculates 2D mesh metrics such as projected X length, projected Y length, and cell area."""
        
        if self.X is None or self.Y is None:
            raise Exception("You must perform the read_mesh() method before adding halo cells.")
            
        # The lengths of each of these sides corresponds to the components of the normal vector to this side
        self.dhx = np.diff(self.X, axis = 1) # Change in x value of horizontal faces (xsi)
        self.dvx = np.diff(self.X, axis = 0) # Change in x value of vertical faces (eta)
        self.dhy = np.diff(self.Y, axis = 1) # Change in y value of horizontal faces (xsi)
        self.dvy = np.diff(self.Y, axis = 0) # Change in y value of vertical faces (eta)
        
        self.horizontal_cell_face_lenghts = np.sqrt(self.dhx**2 + self.dhy**2) # Just Euclidean distance
        self.vertical_cell_face_lengths = np.sqrt(self.dvx**2 + self.dvy**2) # Just Euclidean distance
        
        # Calculate the diagonal vectors of each cell
        bl_tr_diag_x = self.dhx[:-1,:] + self.dvx[:,1:] # bottom left to top right diagonal x component
        bl_tr_diag_y = self.dhy[:-1,:] + self.dvy[:,1:] # bottom left to top right diagonal y component
        
        # ---- Check below if stuff doesn't work right
        # I negated these here to match my HW answer and didn't think about why
        
        tl_br_diag_x = -(self.dhx[1:,:] - self.dvx[:,1:]) # top left to bottom right diagonal x component
        tl_br_diag_y = -(self.dhy[1:,:] - self.dvy[:,1:]) # top left to bottom right diagonal y component

        # Calculate the cell areas via professor's method (cross product of diagonals / 2)
        self.cell_volumes = (bl_tr_diag_x * tl_br_diag_y - tl_br_diag_x * bl_tr_diag_y)  / 2
        
        if plot:
            properties = [
                self.dhx[:-1,:],
                self.dvx[:,:-1],
                self.dhy[:-1,:], 
                self.dvy[:,:-1], 
                self.cell_volumes
            ]
            
            labels = ["Bottom Wall of Cell Projected X Component Contour Plot",
            "Left Wall of Cell Projected X Component Contour Plot",
            "Bottom Wall of Cell Projected Y Component Contour Plot",
            "Left Wall of Cell Projected Y Component Contour Plot",
            "Cell Area Contour Plot",
            ]
            
            for prop, label in zip(properties, labels):

                plot_save_path = (osjoin(plots_save_dir, label.lower().replace(" ","_") + ".png") 
                                  if plots_save_dir is not None else None)
                
                fig, ax = self.plot_mesh_contour(
                    volume_centered_property = prop, 
                    title = label,
                    plot_save_path = plot_save_path
                    )
                
                
########################################################################################################################
########################################################################################################################
#%% Mesh Visualization Methods
########################################################################################################################
########################################################################################################################
            
########################################################################################################################
# Plot PColor Mesh
########################################################################################################################
    
    def plot_mesh_contour(
            self, 
            volume_centered_property: np.ndarray, 
            title: str | None = None, 
            cmap: str | None = None, 
            plot_save_path: str | None = None,
            cbar_units: str | None = None,
            fig_ax: tuple | None = None,
            plot_gridlines: bool = True,
            linewidth: float | None = None,
            color_halos: bool = True,
            ):
        """A function which takes in volume_centered_property which is a 2D array corresponding to a value which is 
        stored at the cell volumes, and plots a contour plot."""
        
        # Generate the figure and axes objects
        
        fig, ax = plt.subplots() if fig_ax is None else fig_ax
        
        # Define the color map if not specified as an input
        cmap = 'jet' if cmap is None else cmap
        
        # Cut off the halo cells if they're not necessary
        X = self.X if color_halos else self.X[1:-1, 1:-1]
        Y = self.Y if color_halos else self.Y[1:-1, 1:-1]
        
        # Plot the colormesh on the axes
        color_mesh = ax.pcolormesh(X, Y, volume_centered_property, shading = 'flat', cmap = cmap)
        
        # Add the colorbar
        cbar = fig.colorbar(color_mesh)
        
        # Set the colorbar units if specified
        if cbar_units is not None:
            cbar.set_label(cbar_units)
        
        # Add the gridlines and other mesh properties
        if plot_gridlines:
            fig, ax = self.__plot_mesh_gridlines(title = title, fig_ax = (fig, ax), linewidth = linewidth)
        else:
            # Set certain hardcoded figure properties that would otherwise be called in plot_mesh_gridlines
            fig, ax = MeshClass.__set_hardcoded_plot_properties(fig, ax)
            
            
        # Save the figure if save path is specified
        if plot_save_path is not None:
            fig.savefig(plot_save_path, bbox_inches = 'tight')
        
        return fig, ax
    

            
########################################################################################################################
########################################################################################################################
#%% Private Methods
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Plot Mesh Gridlines
########################################################################################################################

    def __plot_mesh_gridlines(self, title = None, fig_ax = None, save_path = None, linewidth: float | None = None):
        """A function which plots the generic mesh and then returns the figure and axes objects.
        
        This could've been done by specifying an edge color and activating antialiasing in the pcolormesh function but 
        it didn't look as good. 
        """
        
        # Set a default linewidth if None is specified
        linewidth = 0.15 if linewidth is None else linewidth
        
        # Create the figure and axes object or retrieve them if specified
        fig, ax = plt.subplots() if fig_ax is None else fig_ax
        
        # This came from a stack overflow answer and I have no clue how it works :)
        # Plot the mesh grid
        segs1 = np.stack((self.X,self.Y), axis = 2)
        segs2 = segs1.transpose(1,0,2)
        ax.add_collection(LineCollection(segs1, linewidths = (linewidth)))
        ax.add_collection(LineCollection(segs2, linewidths = (linewidth)))
        
        if title is not None:
            ax.set_title(title)
            
        # Set certain hardcoded figure properties
        fig, ax = MeshClass.__set_hardcoded_plot_properties(fig, ax)
        
        # Save the figure if save path is specified
        if save_path is not None:
            fig.savefig(save_path, bbox_inches = 'tight')
        
        return fig, ax
    
########################################################################################################################
# Set Hardcoded Plot Properties
########################################################################################################################
        
    @staticmethod
    def __set_hardcoded_plot_properties(fig, ax):
        """Set certain plot properties like x_lim, y_lim, x_label, etc. that are hardcoded and not going to change for 
        the purposes of this project."""
        
        # Set the x and the y label
        ax.set_xlabel("x", fontsize = 16)
        ax.set_ylabel("y", fontsize = 16)
        
        # Set the hardcoded mesh limits
        ax.set_xlim(-0.6,1.6)
        ax.set_ylim(-0.1,1.6)
        
        # Set the hardcoded mesh vertical and horizontal bounds
        ax.hlines(1.5, -0.5, 1.5, color = 'k')
        ax.vlines(-0.5, 0, 1.5,  color = 'k')
        ax.vlines(1.5, 0, 1.5, color = 'k')
        
        # Plot the airfoil outline
        
        # Plot the pressure side of the airfoil contour
        slope = np.tan(np.radians(10))
        x = np.linspace(0, 0.5, 2)
        y = slope * x
        ax.plot(x, y, color = 'k')
        
        # Plot the suction side of the airfoil contour
        slope = np.tan(np.radians(-10))
        x = np.linspace(0.5, 1, 2)
        y = slope * (x - 1) # Slope intercept form of the line
        ax.plot(x, y, color = 'k')
        
        # Plot the rest of the bottom that isn't the airfoil
        ax.hlines(0, -0.5,0, color = 'k')
        ax.hlines(0, 1, 1.5, color = 'k')
        
        
        return fig, ax