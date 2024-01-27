mod mesh;
mod solver;
use std::io;
use std::time::Instant;
// use ndarray::{Array2, Array3};
// use ndarray_npy::write_npy;



fn main() {
    // Load the mesh and process it
    let file_path: &str = r#"C:\Users\212808354\Box\Misc\Grad School\CFD\project\grids\g65x49u.dat"#;

    let mut mesh = mesh::Mesh::new();
    mesh.load_mesh(file_path);
    mesh.add_halo_cells();
    mesh.calculate_2d_metrics();

    // Initialize the CFD solver and prepare for iteration
    let mut solver = solver::Solver::new(&mesh, 0.3);
    solver.set_initial_conditions();
    solver.apply_all_bcs();

    // Get user input for number of iterations
    println!("Enter the number of iterations:");

    let mut input = String::new();

    // Read the console
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input.");

    // Parse the string into an integer
    let iterations: u32 = input.trim().parse().expect("Invalid input. Please enter a valid integer.");

    // Iterate and time the iterations

    let start_time = Instant::now();

    for _ in 1..iterations {
        solver.solve_one_step();
    }

    let elapsed_time = start_time.elapsed();
    let elapsed_seconds = elapsed_time.as_secs_f64();
    let iterations_per_second = iterations as f64/elapsed_seconds;

    solver.write_soln(r#"C:\Users\212808354\Box\Misc\Grad School\CFD\project\code\saved_solutions\rust_soln.npy"#);

    println!("{} iterations took: {:?}, which is a rate of {:.2} iterations per second.", iterations, elapsed_time, iterations_per_second);
}

// fn shape(matrix: &Array2<f64>) {
//     let shape = matrix.shape();
//     println!("Shape: {:?}", shape);
// }

// fn disp_matrix(matrix: &Array2<f64>) {

//     let shape = matrix.shape();
//     println!("Shape: {:?}", shape);


//     for row in matrix.rows() {
//         for &value in row {
//             print!("{:.5} ", value)
//         }
//         println!()
//     }
// }
 

// fn write_array(array: &Array3<f64>, file_path: &str) {
//     write_npy(file_path, array).unwrap();
// }