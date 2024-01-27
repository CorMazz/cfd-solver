use std::fs;
use ndarray::{Array2, s};
// use ndarray::prelude::*;


pub struct Mesh {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
    pub dhx: Array2<f64>,
    pub dvx: Array2<f64>,
    pub dhy: Array2<f64>,
    pub dvy: Array2<f64>,
    pub horizontal_cell_face_lengths: Array2<f64>,
    pub vertical_cell_face_lengths: Array2<f64>,
    pub cell_volumes: Array2<f64>,
}

impl Mesh {
    /// This function generates a new mesh instance.
    pub fn new() -> Mesh {
        Mesh {
            x: Array2::zeros((0,0)),
            y: Array2::zeros((0,0)),
            dhx: Array2::zeros((0,0)),
            dvx: Array2::zeros((0,0)),
            dhy: Array2::zeros((0,0)),
            dvy: Array2::zeros((0,0)),
            horizontal_cell_face_lengths: Array2::zeros((0,0)),
            vertical_cell_face_lengths: Array2::zeros((0,0)),
            cell_volumes: Array2::zeros((0,0)),
        }
    }

    /// This function loads a stuctured mesh from a .txt file, given a file path. The mesh dimensions needs to be the first row of the file. 
    pub fn load_mesh(&mut self, file_path: &str) {
        
        // Read the contents
        let contents = fs::read_to_string(file_path)
            .expect("Error reading file.");

        // Get the contents into a vector of tuples containing x, y coords.
        let mut coords: Vec<(f64, f64)> = contents
        .trim()
        .split('\n')
        .map(|row| {
            let values = row
            .trim()
            .split(',')
            .map(|val| {
                val
                .trim()
                .parse::<f64>()
                .unwrap()
            }).collect::<Vec<f64>>();
            (values[0], values[1])
        }).collect::<Vec<(f64, f64)>>();

        // Grab the mesh dimensions from the first row
        let (nx, ny) = coords.remove(0);

        // Store the x and y coordinates in self.x and self.y as a 2D meshgrid. 
        self.x = Array2::from_shape_vec((ny as usize, nx as usize), coords.iter().map(|&(x, _)| x).collect()).unwrap();
        self.y = Array2::from_shape_vec((ny as usize, nx as usize), coords.iter().map(|&(_, y)| y).collect()).unwrap();
    }

    /// This function adds halo cells to the mesh.
    pub fn add_halo_cells(&mut self) {

        self.x = pad_array(&self.x);
        self.y = pad_array(&self.y);

    }

    /// This function calculates 2D volume metrics for the mesh
    pub fn calculate_2d_metrics(&mut self) {
        let dhx = self.x.slice(s![.., 1..]).to_owned() - self.x.slice(s![.., ..-1]).to_owned();
        let dvx = self.x.slice(s![1.., ..]).to_owned() - self.x.slice(s![..-1, ..]).to_owned();
        let dhy = self.y.slice(s![.., 1..]).to_owned() - self.y.slice(s![.., ..-1]).to_owned();
        let dvy = self.y.slice(s![1.., ..]).to_owned() - self.y.slice(s![..-1, ..]).to_owned();

        // Calculate the lengths of each side
        let horizontal_cell_face_lengths = (dhx.mapv(|a: f64| a.powi(2)) + dhy.mapv(|a: f64| a.powi(2))).mapv(f64::sqrt);
        let vertical_cell_face_lengths = (dvx.mapv(|a: f64| a.powi(2)) + dvy.mapv(|a: f64| a.powi(2))).mapv(f64::sqrt);

        // Calculate the diagonal vectors of each cell
        let bl_tr_diag_x = &dhx.slice(s![..-1, ..]) + &dvy.slice(s![.., 1..]);
        let bl_tr_diag_y = &dhy.slice(s![..-1, ..]) + &dvy.slice(s![.., 1..]);

        let tl_br_diag_x = -(&dhx.slice(s![1.., ..]) - &dvy.slice(s![.., 1..]));
        let tl_br_diag_y = -(&dhy.slice(s![1.., ..]) - &dvy.slice(s![.., 1..]));

        // Calculate the cell areas
        let cell_volumes = (bl_tr_diag_x * tl_br_diag_y - tl_br_diag_x * bl_tr_diag_y) / 2.0;

        // Assign the calculated values to the mesh struct
        self.dhx = dhx;
        self.dvx = dvx;
        self.dhy = dhy;
        self.dvy = dvy;
        self.horizontal_cell_face_lengths = horizontal_cell_face_lengths;
        self.vertical_cell_face_lengths = vertical_cell_face_lengths;
        self.cell_volumes = cell_volumes;
    }



}


fn pad_array(array: &Array2<f64>) -> Array2<f64> {
    // Get the shape of the original mesh
    let (ny, nx) = array.dim();

    // Create a large array of zeros
    let mut array_padded = Array2::<f64>::zeros((ny + 2, nx + 2));

    // Fill the center values of that array with the original mesh
    array_padded.slice_mut(s![1..ny+1,1..nx+1]).assign(&array);

    // Modify the top and bottom rows
    for j in 1..nx + 1 {
        array_padded[[0, j]] = 2.0 * array_padded[[1, j]] - array_padded[[2, j]];
        array_padded[[ny + 1, j]] = 2.0 * array_padded[[ny, j]] - array_padded[[ny - 1, j]];
    }

    // Modify the left and right columns
    for i in 0..ny + 2 {
        array_padded[[i, 0]] = 2.0 * array_padded[[i, 1]] - array_padded[[i, 2]];
        array_padded[[i, nx + 1]] = 2.0 * array_padded[[i, nx]] - array_padded[[i, nx - 1]];
    }

    return array_padded
}
