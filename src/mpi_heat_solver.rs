use mpi::traits::*;
use std::time::Instant;

pub fn solve_tridiagonal(a: &Vec<f64>, b: &Vec<f64>, c: &Vec<f64>, d: &mut Vec<f64>) {
    let n = d.len();
    let mut alpha = vec![0.0; n];
    let mut beta = vec![0.0; n];

    // Forward sweep
    alpha[0] = -c[0] / b[0];
    beta[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] + a[i] * alpha[i - 1];
        alpha[i] = -c[i] / denom;
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom;
    }

    // Back substitution
    for i in (0..n - 1).rev() {
        d[i] = alpha[i] * d[i + 1] + beta[i];
    }
}

pub fn solve_heat_equation_mpi(
    length: f64,
    temperature: f64,
    points: usize,
    dt: f64,
    time_steps: usize,
) -> (Vec<f64>, u128) {
    let start_time = Instant::now();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let dx = length / (points as f64);
    let alpha = dt / (dx * dx);

    // Determine the local range of points for each process
    let local_n = points / size as usize;
    let local_start = rank as usize * local_n + 1;
    let local_end = if rank == size - 1 {
        points - 1
    } else {
        (rank as usize + 1) * local_n
    };

    let mut u_local = vec![0.0; local_n + 2]; // +2 for ghost points at the boundaries
    let mut u_new_local = vec![0.0; local_n + 2];

    if rank == size - 1 {
        u_local[local_n + 1] = temperature; // Right boundary condition
    }

    let a = vec![-alpha; local_n];
    let b = vec![1.0 + 2.0 * alpha; local_n];
    let c = vec![-alpha; local_n];
    let mut d = vec![0.0; local_n];

    // Time iteration loop
    for _ in 0..time_steps {
        // Fill the `d` vector (right-hand side) based on the current temperature values
        for i in 0..local_n {
            d[i] = u_local[i + 1];
        }

        if rank == 0 {
            d[0] += alpha * 0.0; // Left boundary condition
        }
        if rank == size - 1 {
            d[local_n - 1] += alpha * temperature; // Right boundary condition
        }

        // Solve tridiagonal system locally
        solve_tridiagonal(&a, &b, &c, &mut d);

        // Exchange ghost points with neighboring processes
        if rank > 0 {
            // Send left boundary value to the left neighbor and receive from the left
            world.process_at_rank(rank - 1).send(&u_local[1]);
            u_local[0] = world.process_at_rank(rank - 1).receive::<f64>().0;
        }

        if rank < size - 1 {
            // Send right boundary value to the right neighbor and receive from the right
            world.process_at_rank(rank + 1).send(&u_local[local_n]);
            u_local[local_n + 1] = world.process_at_rank(rank + 1).receive::<f64>().0;
        }

        // Update local temperature array
        for i in 0..local_n {
            u_new_local[i + 1] = d[i];
        }

        u_local.copy_from_slice(&u_new_local);
    }

    // Gather the results to rank 0
    let mut final_result = if rank == 0 {
        vec![0.0; points + 1]
    } else {
        vec![]
    };

    world.all_gather_into(&u_local[1..local_n + 1], &mut final_result[local_start..local_end]);

    // Ensure boundary conditions
    if rank == 0 {
        final_result[points] = temperature; // Right boundary condition
    }

    // Return result on the root process
    if rank == 0 {
        (final_result, start_time.elapsed().as_micros())
    } else {
        (vec![], 0) // Return empty result for non-root processes
    }
}