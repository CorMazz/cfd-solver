
use ndarray::{Array4, Array3, Array2, Array1, s, stack, Axis};
use ndarray_npy::write_npy;
use crate::mesh;

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Create the Solver Struct
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// A struct containing data and methods to solve the inviscid Euler equations over prescribed structured meshes.
pub struct Solver<'a> {
    pub mesh: &'a mesh::Mesh,
    pub s_xsi_x: Array2<f64>,
    pub s_xsi_y: Array2<f64>,
    pub s_eta_x: Array2<f64>,
    pub s_eta_y: Array2<f64>,
    pub s_xsi: Array2<f64>,
    pub s_eta: Array2<f64>,
    pub delta_v: Array2<f64>,
    pub q: Array4<f64>,
    pub q_v: Array3<f64>,
    pub e_flux: Array3<f64>,
    pub f_flux: Array3<f64>,
    pub delta_t: Array2<f64>,
    pub r: f64,
    pub gamma: f64,
    pub c_p: f64,
    pub q_ref: Array1<f64>, // also initial conditions, to be stored here
    pub q_ic: Array1<f64>, // initial conditions to be stored here
    pub iteration: i64,
    pub l_inf_norm_history: Vec<f64>,
    pub l2_norm_history: Vec<f64>,
    pub max_cfl: f64,
}

impl<'a> Solver<'a> {

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Define an Init Method
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// This function creates a new solver object.
pub fn new(mesh_obj: &'a mesh::Mesh, max_cfl:f64) -> Solver {
    
    // Grab the shape of the mesh
    let q_dims = mesh_obj.cell_volumes.shape();
    let _3d_shape = (q_dims[0], q_dims[1], 4);

    // Grab the shapes of S_xsi and S_eta to create the flux vectors
    let s_xsi_shape = mesh_obj.vertical_cell_face_lengths.shape();
    let s_eta_shape = mesh_obj.horizontal_cell_face_lengths.shape();

    Solver {
        mesh: mesh_obj,
        // Projected cell face areas (negative signs per the class notes)
        s_xsi_x: mesh_obj.dvy.clone(),
        s_xsi_y: -mesh_obj.dvx.clone(),
        s_eta_x: -mesh_obj.dhy.clone(),
        s_eta_y: mesh_obj.dhx.clone(),
        s_xsi: mesh_obj.vertical_cell_face_lengths.clone(),
        s_eta: mesh_obj.horizontal_cell_face_lengths.clone(),
        delta_v: mesh_obj.cell_volumes.clone(),
        q: Array4::zeros((2, q_dims[0], q_dims[1], 4)),
        q_v: Array3::zeros(_3d_shape),
        e_flux: Array3::zeros((s_xsi_shape[0], s_xsi_shape[1], 4)),
        f_flux: Array3::zeros((s_eta_shape[0], s_eta_shape[1], 4)),
        delta_t: Array2::zeros((q_dims[0], q_dims[1])),
        r: 287.0, // J/kg
        gamma: 1.4,
        c_p: 1005.0, // J/kg.k
        q_ref: Array1::zeros(4),
        q_ic: Array1::zeros(4),
        iteration: 0,
        l_inf_norm_history: Vec::new(),
        l2_norm_history: Vec::new(),
        max_cfl: max_cfl,

    }
}

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Set the Initial Conditions
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Automatically set the state vector Q based on a given static pressure of 101325 Pa, T = 300K, M = 2.00,
/// which translates to an axial convective velocity of 694.4 m/s since the sound speed is 347.2 m/s.
/// Using the perfect gas equation of state, p = rho * R * T
pub fn set_initial_conditions(&mut self) {
    
    // Define given conditions
    let p = 101325.0; // Pa
    let t = 300.0; // K
    let m = 2.0; // Mach Number

    // Calculate density and sound speed
    let rho = p / (self.r * t); // kg / m^3
    let c = (self.gamma * self.r * t).sqrt(); // m / s
    let u = m * c; // m / s
    let v = 0.0; // m / s given as 0

    // From Topic 24.1 Notes
    let q0 = rho;
    let q1 = rho * u;
    let q2 = rho * v;
    let q3 = p / (self.gamma - 1.0) + rho * (u.powi(2) + v.powi(2)) / 2.0;

    // // Store it as the q_ic array
    self.q_ic = Array1::from(vec![q0, q1, q2, q3]);
    self.q.slice_mut(s![.., .., .., ..]).assign(&self.q_ic);

    // Modify the velocity components to be the magnitude of velocity and store it as Q_ref
    self.q_ref =  Array1::from(vec![
        rho, 
        rho* (u.powi(2) + v.powi(2)).sqrt(), 
        rho* (u.powi(2) + v.powi(2)).sqrt(), 
        q3
        ]
    );

}

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Set the Boundary Conditions
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Apply boundary conditions to the t1 section of Q
/// Per section 5 of the project statement, since the flow is supersonic, characteristics (information) is only
/// flowing from right to left. Therefore, the flow at the inlet plane is constant, and set by the inlet conditions
/// only (independent of what is happening in our domain).
fn apply_inlet_bcs(&mut self) {
    // Set the inlet boundary condition
    self.q.slice_mut(s![1,..,0,..]).assign(&self.q_ic);
}
/// Apply boundary conditions to the t1 section of Q
/// Per section 5 of the project statement, since the flow is supersonic, characteristics (information) is only
/// flowing from right to left. I can't fully explain this at the moment.
fn apply_outlet_bcs(&mut self) {
    // Set the outlet boundary condition
    let q_outlet = self.q.slice(s![0,..,-2,..]).to_owned();
    self.q.slice_mut(s![1,..,-1,..]).assign(&q_outlet);
}

/// Apply the adiabatic slip condition to all walls.
fn apply_wall_bcs(&mut self) {

    // Calculate primitives
    let [p, u, v, t] = self.calculate_qv();

    // Define the indices for the bottom and the top row of non-halo cells (indexing starts from the bottom left)
    // as well as the halo cells on the bottom and top
    let (row_indices, halo_row_indices) = ([1, -2], [0, -1]);

    // This sets all walls to slip condition                  
    for (&row, &halo_row) in row_indices.iter().zip(halo_row_indices.iter()) {

        // -----------------------------------------------------------------------------------------------------------------------
        // Determine u and v 
        // -----------------------------------------------------------------------------------------------------------------------

        // For an inviscid slip condition
        
        // Retrive the correct row of area vector sizes
        let s_eta_x = self.s_eta_x.slice(s![row, ..]);
        let s_eta_y = self.s_eta_y.slice(s![row, ..]);
            
        // u1 and v1 are notation from the 24.4 Notes
        let u1 = u.slice(s![row, ..]);
        let v1 = v.slice(s![row, ..]);
            
        // Apply the inviscid 'slip' wall condition everywhere to get this working 
        
        // u0 and v0 are notation from 24.4 Notes (Eqn 5) and were solved via sympy in a separate .py file.
        let u0 = 
            (-&s_eta_x.mapv(|a| a.powi(2))*&u1 - 2.0*&s_eta_x*&s_eta_y*&v1 + &s_eta_y.mapv(|a| a.powi(2))*&u1) 
            / (&s_eta_x.mapv(|a| a.powi(2)) + &s_eta_y.mapv(|a| a.powi(2))); 
        let v0 = 
            (&s_eta_x.mapv(|a| a.powi(2))*&v1 - 2.0*&s_eta_x*&s_eta_y*&u1 - &s_eta_y.mapv(|a| a.powi(2))*&v1)
            /(&s_eta_x.mapv(|a| a.powi(2)) + &s_eta_y.mapv(|a| a.powi(2)));
        
        // -----------------------------------------------------------------------------------------------------------------------
        // Determine p and T
        // -----------------------------------------------------------------------------------------------------------------------
        
            //  Per Equation 7 in the 24.4 Notes set the pressure gradient to 0
            let p0 = p.slice(s![row, ..]);// p[row] is like p1

            //  Per eqn 8 if the wall condition is adiabatic the temperature gradient is also 0
            let t0 = t.slice(s![row, ..]); //  T[row] is like T1

        // -----------------------------------------------------------------------------------------------------------------------
        // Calculate and Update Q Vector Values
        // -----------------------------------------------------------------------------------------------------------------------
        
        // Calculate Q0, Q2 and Q3
        let q0 = &p0 / (self.r * &t0); // Ideal Gas Law
        let q1 = &q0 * &u0;
        let q2 = &q0 * &v0;
        let q3 = &p0 / (self.gamma - 1.0) + &q0 * (&u0.mapv(|a| a.powi(2)) + &v0.mapv(|a| a.powi(2))) / 2.0;

        // Stack the arrays
        let q_stack = stack![Axis(1), q0, q1, q2, q3];
        
        //  Apply the BC to the next time step of the Q_vector
        self.q.slice_mut(s![1,halo_row,..,..]).assign(&q_stack);
    
    }
}
        
pub fn apply_all_bcs(&mut self) {
    self.apply_inlet_bcs();
    self.apply_outlet_bcs();
    self.apply_wall_bcs();
    self.roll_soln();
}


// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Write Q to Numpy
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// A function which writes the solution as a .npy file to a given file path ending with .npy.
/// Writes the Q matrix as a 3d array, so only the current timestep is written.
pub fn write_soln(&self, file_path: &str) {
    write_npy(file_path, &self.q.slice(s![0,..,..,..])).unwrap();
}

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calculate Q_v
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Calculates primitive variables in the Q_v vector, which are p, u, v, and T. Returns cell centered values for all cells.
fn calculate_qv(&self) -> [Array2<f64>; 4]  {
    
    let rho = self.q.slice(s![0,..,..,0]);
    let rho_u = self.q.slice(s![0,..,..,1]);
    let rho_v = self.q.slice(s![0,..,..,2]);
    let rho_et = self.q.slice(s![0,..,..,3]);
    
    let u = &rho_u / &rho;
    let v = &rho_v / &rho;
    
    let p = (&rho_et - &rho*(&u.mapv(|a| a.powi(2)) + &v.mapv(|a| a.powi(2))) / 2.0 ) * (self.gamma - 1.0);
    
    // Using ideal gas law
    let t = &p / (&rho * self.r);

    return [p, u, v, t]
}

/// A function to roll the solution vector, making the newly solved solution in the next timestep the current timestep.
fn roll_soln(&mut self) {
    // Have to do it this way because there is no 'roll' function
    let q_t1 = self.q.slice(s![1, .. , .., ..]).to_owned();
    self.q.slice_mut(s![0, .., .., ..]).assign(&q_t1);
}

pub fn solve_one_step(&mut self) {

    let mut flux_matrices = [&mut self.e_flux, &mut self.f_flux];
    let flux_directions = ["xsi", "eta"];
    let area_components_list = [
        [&self.s_xsi_x, &self.s_xsi_y, &self.s_xsi],
        [&self.s_eta_x, &self.s_eta_y, &self.s_eta,]
    ];
    
    for (flux_matrix, (flux_direction,area_components)) 
        in 
        flux_matrices.iter_mut()
        .zip(flux_directions.iter()
        .zip(area_components_list.iter())
        ) 
    {

        // Unpack the area components
        let [s_x, s_y, s] = *area_components;
        
        // Call the update flux function which changes the flux matrix (E_flux or F_flux) in place
        update_flux(
            &self.q, 
            *flux_matrix,
            &flux_direction,
            s_x,
            s_y,
            s,
            self.gamma
        )
}

// Calculate q_v primitives
let [_p, _u, _v, t] = self.calculate_qv();

let c = (self.gamma * self.r * t).mapv(|a| a.sqrt());

// Calculate the local delta T for each cell
self.delta_t = calculate_local_delta_t(
    &self.q, 
    &c, 
    &self.s_xsi_x, 
    &self.s_xsi_y, 
    &self.s_xsi, 
    &self.s_eta_x, 
    &self.s_eta_y, 
    &self.s_eta, 
    &self.delta_v,
    self.max_cfl,
    );

    // Q[1] = (
    //     Q[0] -
    //     (delta_T[:,:,None] / delta_V[:,:,None]) * 
    //         (
    //         np.diff(E_flux * S_xsi[:,:,None], axis = 1) +
    //         np.diff(F_flux * S_eta[:,:,None], axis = 0)
    //         )
    //     )

    let q_current = self.q.slice(s![0, .., .., ..]).to_owned();
    let effective_cfl = self.delta_t.clone().insert_axis(Axis(2)) / self.delta_v.clone().insert_axis(Axis(2));
    let e_diff = diff(&self.e_flux * &self.s_xsi.clone().insert_axis(Axis(2)),1);
    let f_diff = diff(&self.f_flux * &self.s_eta.clone().insert_axis(Axis(2)),0);
    let q_next = q_current - effective_cfl * (e_diff + f_diff);

    self.q.slice_mut(s![1, .., .., ..]).assign(&q_next);

    self.apply_all_bcs();

    self.roll_soln();

    self.iteration += 1;
}

}

/// Calculates the flux over a single face and stores each face's flux in the flux matrix. The flux matrix will be padded with zeros since those
/// correspond to halo cell fluxes that we don't care about. I'm leaving them there for ease of indexing.
fn update_flux(
    q: &Array4<f64>,
    flux_matrix: & mut Array3<f64>,
    flux_direction: &str,
    s_x: &Array2<f64>,
    s_y: &Array2<f64>,
    s: &Array2<f64>, 
    gamma: f64
    ) 
    {

    // See the processed indexed mesh diagram. The (0,0) row and column are all exterior cell boundaries, so we 
    // can skip the 0th index. The 
    for i in 1..s.shape()[0]-1 { // Every row, skip the halo cells
        for j in 1..s.shape()[1] - 1 {

            // Interpolate the left and right sides of the Q vector
            let q_lr = interpolate_q_lr(&q ,i, j, &flux_direction);

            // Calculate Roe Averages of Properties (xsi direction)
            let rho_avgs = calc_rho_averages(&q_lr, gamma);

            // Calculate the diagonalized A matrix
            let [l1, lambda, r1] = calc_diagonalized_a(rho_avgs, s_x[[i, j]], s_y[[i,j]], s[[i,j]], gamma);

            // Take the absolute value of the eigenvectors and recombine A
            let a_bar = r1.dot(&lambda.mapv(|a| a.abs())).dot(&l1); 

            // Calculate E or F (fv for flux vector) based on the Left and Right Q vectors
            let fv_q_lr = calc_flux_vector_of_q(
                &q_lr,
            s_x[[i,j]],
            s_y[[i,j]],
                s[[i,j]], // Area components
                gamma,
            );

            // Per Eqn 8 of the Topic 25 Notes
            let flux = (fv_q_lr[0].to_owned() + fv_q_lr[1].to_owned()) / 2.0 - a_bar.dot(&(((q_lr[1].to_owned() - q_lr[0].to_owned())) / 2.0));
            flux_matrix.slice_mut(s![i, j, ..]).assign(&flux);
            }
        } // Every column, skip the halo cells

}

/// A function to interpolate the Q vector to the left and right states of a given wall, vertical or horizontal.
/// The indices i and j here correspond to what wall is being looked at. Returns Q_L and Q_R, which are (4) shaped 
/// numpy arrays containing the values within the Q-vector interpolated to the left and right side of the given wall.
/// Does this in either the 'xsi' or the 'eta' direction, selected with a 'mode' string.

/// Can have higher order accuracy if epsilon, kappa, and a limiter_type are selected.

/// Available limiters are currently:
///     'mc' or "monotonized central"
///     'minmod'
///     "van Leer"
fn interpolate_q_lr(q: &Array4<f64>, i:usize, j:usize, mode: &str) -> [Array1<f64>; 2] {
    // See the Topic 25 Notes Eqn 11 & 12
    
    // Create the variables q_vec and K in the namespace
    let (q_vec, k): (Array2<f64>, usize);

    // Grab the correct row or column as a Q array
    if mode == "xsi" {
        q_vec = q.slice(s![0, i, .., ..]).to_owned(); // Now rows of Q increasing == left to right 
        k = j // Select the proper indexer
    }
    else if mode == "eta"{
        q_vec = q.slice(s![0, .., j, ..]).to_owned();
        k = i;
    } else {
        panic!("Mode must be xsi or eta.")
    }
    // Simple first order interpolation done if we decide to, or if we're in one of the internal boundary cells
            
    let q_l = q_vec.slice(s![k - 1, ..]).to_owned(); // Take the values of Q at the cell just to the left
    let q_r = q_vec.slice(s![k, ..]).to_owned(); // Take the values of Q at the current cell

    return [q_l, q_r]
}

/// Calculate the rho averaged values given Q_LR, a tuple containing the left and right values of Q that need to be 
/// averaged. The left and right values are numpy arrays of length 4 (corresponding to Q vectors)
fn calc_rho_averages(q_lr: &[Array1<f64>; 2], gamma: f64) -> [f64; 5] {
    
    // Unpack Q_LR
    let [q_l, q_r] = q_lr;
    
    // Grab the desired primitive values
    let [rho_l, u_l, v_l, _, h_t_l] = calc_primitives(q_l, gamma);
    let [rho_r, u_r, v_r, _, h_t_r] = calc_primitives(q_r, gamma);
    
    // Per the Topic 23 Notes
    let denom = rho_r.sqrt() + rho_l.sqrt(); // Commonly used denominator
    
    let u_bar = (rho_r.sqrt() * u_r + rho_l.sqrt() * u_l) / denom; // Eqn 22
    let v_bar = (rho_r.sqrt() * v_r + rho_l.sqrt() * v_l) / denom; // Eqn 22
    let rho_bar = (rho_r * rho_l).sqrt(); // Eqn 23
    let h_t_bar = (rho_r.sqrt()*h_t_r + rho_l.sqrt()*h_t_l) / denom; // Eqn 24
    let c_bar = ((gamma-1.0) *(h_t_bar - (u_bar.powi(2) + v_bar.powi(2)) / 2.0)).sqrt(); // Eqn 25
    
    return [rho_bar, u_bar, v_bar, h_t_bar, c_bar]


}

/// Given a 1 dimensional numpy array (ny, nx, 4) representing the Q vector at every cell, which contains 
/// (rho, rho*u, rho*v, rho*e_t), calculate a vector of primitives, u, which is (rho, u, v, p, h_t).
fn calc_primitives(q_1d: &Array1<f64>, gamma: f64) -> [f64; 5]  {
  
    let rho = q_1d[0];
    let u =  q_1d[1] / rho;
    let v = q_1d[2] / rho;
    let e_t = q_1d[3] / rho;
    let p = (gamma - 1.0) * rho * (e_t - (u.powi(2) + v.powi(2))) / 2.0;
    
    // PG 12 of Topic 23 Notes
    let h_t = e_t + p / rho;
    
    return [rho, u, v, p, h_t]

}

/// A function which takes scalar values of rho, u, v, h_t, c, area components, and gamma, and returns a tuple 
/// containing L1, lambda_i, R1 as denoted in the Topic 26.1 Notes, where L1 and R1 are the left and right eigenvectors
/// respectively, and lambda_i is a diagonal matrix of eigenvalues.
fn calc_diagonalized_a(rho_avgs: [f64; 5], s_x: f64, s_y: f64, s: f64, gamma: f64) -> [Array2<f64>; 3] {
    
    // Unpack the rho averaged values
    let [_rho, u, v, h_t, c] = rho_avgs;

    // Translate from class notation to the notation of the AIAA-2001-2609 Paper 
    let nx = s_x/s;
    let ny = s_y/s;
    let v_n = u*nx + v*ny;
    let h_0 = h_t;
    let e_k = (u.powi(2) + v.powi(2)) / 2.0;
    let a = c;
 
    // R1 from Topic 26.1 Notes, Delete the 4th Row, 5th Column for 2D
    let r1 = Array2::from(vec![  
        [1.0,              1.0,      1.0,              0.0], // If you don't put the .0 afterwards numba cries
        [u - a*nx,       u,      u + a*nx,       ny],
        [v - a*ny,       v,      v + a*ny,       -nx],
        [h_0 - a*v_n,    e_k,    h_0 + a*v_n,    u*ny - v*nx]
        ]);
    
    // Define a common denominator for convenience 
    let d0 = 2.0 * a.powi(2);
    
    // L1 from Topic 26.1 Notes, Delete the 5th Row, 4th Column for 2D
    let l1 = Array2::from(vec![
        [((gamma-1.0)*e_k+a*v_n)/d0,   ((1.0-gamma)*u-a*nx)/d0,      ((1.0-gamma)*v-a*ny)/d0,  (gamma-1.0)/d0],
        [(a.powi(2)-(gamma-1.0)*e_k)/a.powi(2),  ((gamma-1.0)*u)/a.powi(2),         ((gamma-1.0)*v)/a.powi(2),     (1.0-gamma)/a.powi(2)],
        [((gamma-1.0)*e_k-a*v_n)/d0,   ((1.0-gamma)*u + a*nx)/d0,    ((1.0-gamma)*v+a*ny)/d0,  (gamma-1.0)/d0],
        [v*nx-u*ny,                  ny,                         -nx,                    0.0],
        ]);
    
    // lambda_ij from Topic 26.1 Notes, Delete the 4th or 5th column idk, but they're identical so who cares.
    // Eqn 22 from the notes
    let lambda= Array2::from(vec![
        [v_n-a, 0.0, 0.0, 0.0],
        [0.0,   v_n, 0.0, 0.0],
        [0.0,   0.0, v_n+a, 0.0],
        [0.0, 0.0, 0.0, v_n]
    ]);

    // lambda_ij = np.diag(lambda_ij);
    
    return [l1, lambda, r1]
} 

///Given an ndarray containing a left and a right Q vector 4 units long, calculate the E or F vector. Per the Topic 
///26 Notes Pg. 3
fn calc_flux_vector_of_q(q_lr: &[Array1<f64>; 2], s_x: f64, s_y: f64, s: f64, gamma: f64) -> [Array1<f64>; 2] {
    // Calculate Nx, Ny
    let (nx, ny) = (s_x / s, s_y / s);

    // Initialize an empty array
    let mut fv_lr = [Array1::zeros(0),  Array1::zeros(0)];

    for (i, q) in q_lr.iter().enumerate() {
        // Unpack Q for ease of use
        let [q1, q2, q3, q4 ] = [q[0], q[1], q[2], q[3]];
        
        let e1 = q2*nx + q3*ny;
        let e2 = q2.powi(2)/q1*nx + q2*q3/q1*ny + (gamma-1.0)*(q4 - (q2.powi(2)/q1 + q3.powi(2)/q1)/2.0)*nx;
        let e3 = q2*q3/q1*nx + q3.powi(2)/q1*ny + (gamma-1.0)*(q4 - (q2.powi(2)/q1 + q3.powi(2)/q1)/2.0)*ny;
        let e4 = (gamma*q4 - (gamma-1.0)/2.0*(q2.powi(2)/q1 + q3.powi(2)/q1))*(q2/q1*nx + q3/q1*ny);
        
        fv_lr[i] = Array1::from(vec![e1, e2, e3, e4]);
    }
    return fv_lr

}

/// A function to calculate the largest delta_t allowable in any given cell per the Topic 24.3 notes
fn calculate_local_delta_t(
    q: &Array4<f64>, 
    c: &Array2<f64>, 
    s_xsi_x: &Array2<f64>, 
    s_xsi_y: &Array2<f64>, 
    s_xsi: &Array2<f64>, 
    s_eta_x: &Array2<f64>, 
    s_eta_y: &Array2<f64>, 
    s_eta: &Array2<f64>, 
    delta_v: &Array2<f64>, 
    max_cfl: f64
) -> Array2<f64> {

    // Grab u and v components of velocity
    let (u, v) = (&q.slice(s![0,..,..,1]) / &q.slice(s![0,..,..,0]),   &q.slice(s![0,..,..,2]) / &q.slice(s![0,..,..,0]));
    
    let [s_xsi_x1, s_xsi_y1, s_xsi1, s_eta_x1, s_eta_y1, s_eta1] = interpolate_cell_area_metrics(s_xsi_x, s_xsi_y, s_xsi, s_eta_x, s_eta_y, s_eta);
    
    let u_xsi = &u * &s_xsi_x1 / &s_xsi1 + &v * &s_xsi_y1 / &s_xsi1;
    let v_eta = &u * &s_eta_x1 / &s_eta1 + &v * &s_eta_y1 / &s_eta1;
    
    let xsi_spectral_radius = u_xsi.mapv(|a| a.abs()) + c;
    let eta_spectral_radius = v_eta.mapv(|a| a.abs()) + c;
    
    // Per the Topic 26 Notes
    let (xsi_x, xsi_y) = (s_xsi_x1 / delta_v, s_xsi_y1 / delta_v);
    let (eta_x, eta_y) = (s_eta_x1 / delta_v, s_eta_y1 / delta_v);
    
    // Per the topic 24.3 notes
    let xsi_metric = (xsi_x.mapv(|a| a.powi(2)) + xsi_y.mapv(|a| a.powi(2))).mapv(|a| a.sqrt());
    let eta_metric = (eta_x.mapv(|a| a.powi(2)) + eta_y.mapv(|a| a.powi(2))).mapv(|a| a.sqrt());
    
    // Per the Topic 24.3 Notes, all of this is to mimic the np.minimum function and do this. What a bitch
    // delta_T = max_cfl * np.minimum(1/(xsi_spectral_radius * xsi_metric), 1/(eta_spectral_radius * eta_metric))
    let compare1 = 1.0/(xsi_spectral_radius * xsi_metric);
    let compare2 = 1.0/(eta_spectral_radius * eta_metric);
    let mut comparison = Array2::zeros((compare1.nrows(), compare1.ncols()));

    for i in 0..comparison.nrows() {
        for j in 0..comparison.ncols() {
            comparison[[i, j]] = compare1[[i, j]].min(compare2[[i, j]]);
        }
    }

    let delta_t = max_cfl * comparison;
    
    return delta_t

 }
    
 /// A function to interpolate the cell area metrics to the center of the cells
 fn interpolate_cell_area_metrics(    
    s_xsi_x: &Array2<f64>, s_xsi_y: &Array2<f64>, s_xsi: &Array2<f64>, 
    s_eta_x: &Array2<f64>, s_eta_y: &Array2<f64>, s_eta: &Array2<f64>) ->  [Array2<f64> ;6] {
    

    // Interpolate the cell area metrics to the cell centers
    // Should loop this but I'm lazy 
    let s_xsi_x1 = (&s_xsi_x.slice(s![.., 1..]) + &s_xsi_x.slice(s![.., ..-1]))/ 2.0;
    let s_xsi_y1 = (&s_xsi_y.slice(s![.., 1..]) + &s_xsi_y.slice(s![.., ..-1]))/ 2.0;
    let s_xsi_1 = (&s_xsi.slice(s![.., 1..]) + &s_xsi.slice(s![.., ..-1]))/ 2.0;
    let s_eta_x1 = (&s_eta_x.slice(s![1.., ..]) + &s_eta_x.slice(s![..-1, ..]))/ 2.0;
    let s_eta_y1 = (&s_eta_y.slice(s![1.., ..]) + &s_eta_y.slice(s![..-1, ..]))/ 2.0;
    let s_eta_1 = (&s_eta.slice(s![1.., ..]) + &s_eta.slice(s![..-1, ..]))/ 2.0;
     return [s_xsi_x1, s_xsi_y1, s_xsi_1, s_eta_x1, s_eta_y1, s_eta_1]
}

/// An function to act as np.diff for a 3 dimensional array. 
fn diff(array: Array3<f64>, axis: usize) -> Array3<f64> {

    let diff_array: Array3<f64>;

    // Perform the difference over the first axis of the array
    if axis == 0 {
        diff_array = &array.slice(s![1.., .., ..]) - &array.slice(s![..-1, .., ..]);
    }

    // Perform the difference over the second axis of the array
    else if axis == 1 {

        diff_array = &array.slice(s![.., 1.., ..]) - &array.slice(s![.., ..-1, ..]);
    }
    else {
        panic!("The axis must be 0 or 1.")
    }
    return diff_array
}







