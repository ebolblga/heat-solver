extern crate mpi;
use mpi::topology::Communicator;
use mpi::traits::*;
use mpi::request::WaitGuard;
use std::time::Instant;

pub fn solve_tridiagonal(a: &Vec<f64>, b: &Vec<f64>, c: &Vec<f64>, d: &mut Vec<f64>) {
    let n = d.len();
    let mut alpha = vec![0.0; n];
    let mut beta = vec![0.0; n];

    // Forward elimination
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
    world: mpi::topology::SystemCommunicator
) -> (Vec<f64>, u128) {
    let start_time = Instant::now();

    let dx = length / (points as f64);
    let alpha = dt / (dx * dx);

    let rank = world.rank();      // Current process rank
    let size = world.size();      // Number of processes

    // Divide the domain among processes
    let local_points = points / size as usize;  // Number of points per process
    let mut u = vec![0.0; local_points + 2];  // Solution vector (with ghost cells)
    let mut u_new = vec![0.0; local_points + 2];

    if rank == size - 1 {
        u[local_points + 1] = temperature;  // Rightmost process applies temperature at the boundary
    }

    let mut a = vec![-alpha; local_points];
    let mut b = vec![1.0 + 2.0 * alpha; local_points];
    let mut c = vec![-alpha; local_points];
    let mut d = vec![0.0; local_points];

    for _ in 0..time_steps {
        for i in 1..=local_points {
            d[i - 1] = u[i];  // Copy local part of the solution to d
        }

        // Communicate boundary values with neighbors
        let left = if rank > 0 {
            Some(world.process_at_rank(rank - 1))
        } else {
            None
        };

        let right = if rank < size - 1 {
            Some(world.process_at_rank(rank + 1))
        } else {
            None
        };

        let send_left = if rank > 0 {
            Some(left.unwrap().immediate_send(&u[1]))  // Send the first internal point to the left neighbor
        } else {
            None
        };

        let send_right = if rank < size - 1 {
            Some(right.unwrap().immediate_send(&u[local_points]))  // Send the last internal point to the right neighbor
        } else {
            None
        };

        let mut left_ghost = 0.0;
        let mut right_ghost = 0.0;

        if rank > 0 {
            left.unwrap().receive_into(&mut left_ghost);  // Receive from the left neighbor
        }
        if rank < size - 1 {
            right.unwrap().receive_into(&mut right_ghost);  // Receive from the right neighbor
        }

        // Wait for non-blocking sends to complete
        if let Some(send) = send_left {
            send.wait();
        }
        if let Some(send) = send_right {
            send.wait();
        }

        // Apply boundary conditions
        if rank > 0 {
            d[0] += alpha * left_ghost;  // Left boundary (from neighbor)
        } else {
            d[0] += alpha * 0.0;  // Left boundary (fixed at 0.0)
        }

        if rank < size - 1 {
            d[local_points - 1] += alpha * right_ghost;
        } else {
            d[local_points - 1] += alpha * temperature;
        }

        // Solve the tridiagonal system
        solve_tridiagonal(&a, &b, &c, &mut d);

        // Update the solution
        for i in 1..=local_points {
            u_new[i] = d[i - 1];
        }

        u.copy_from_slice(&u_new);  // Copy new values to u
    }

    // Gather all results on root (rank 0) for final result
    let mut final_result = if rank == 0 {
        vec![0.0; points + 1]
    } else {
        vec![]
    };

    world
        .gather_into_root(&u[1..=local_points], &mut final_result);

    (final_result, start_time.elapsed().as_micros())
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let length = 10.0;
    let temperature = 100.0;
    let points = 100;
    let dt = 0.01;
    let time_steps = 1000;

    let (final_result, duration) = solve_heat_equation_mpi(length, temperature, points, dt, time_steps, world);

    if world.rank() == 0 {
        println!("Final result: {:?}", final_result);
        println!("Execution time: {} microseconds", duration);
    }
}