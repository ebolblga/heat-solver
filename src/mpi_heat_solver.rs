use mpi::traits::*;
use std::time::Instant;

pub fn solve_heat_equation_mpi(
    length: f64,
    temperature: f64,
    points: usize,
    dt: f64,
    time_steps: usize,
) -> (Vec<f64>, u128) {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let start_time = Instant::now();

    let dx = length / (points as f64);
    let alpha = dt / (dx * dx);

    let local_n = points / size as usize; // Number of points per process
    let mut u_local = vec![0.0; local_n + 2]; // Local vector (with ghost cells)
    if rank == size - 1 {
        u_local[local_n + 1] = temperature; // Set the right boundary condition
    }

    for step in 0..time_steps {
        if rank == 0 {
            println!("Rank 0, time step: {}", step);
        }

        // Exchange boundary data with neighbors
        if rank > 0 {
            world.process_at_rank(rank - 1).send(&u_local[1]); // Send left boundary to left neighbor
            let received = world.process_at_rank(rank - 1).receive::<f64>().0;
            u_local[0] = received; // Receive left boundary from left neighbor
        }
        if rank < size - 1 {
            world.process_at_rank(rank + 1).send(&u_local[local_n]); // Send right boundary to right neighbor
            let received = world.process_at_rank(rank + 1).receive::<f64>().0;
            u_local[local_n + 1] = received; // Receive right boundary from right neighbor
        }

        // Local computation (update the interior points)
        let mut u_new = vec![0.0; local_n + 2];
        for i in 1..=local_n {
            u_new[i] = u_local[i] + alpha * (u_local[i - 1] - 2.0 * u_local[i] + u_local[i + 1]);
        }

        u_local.copy_from_slice(&u_new);

        // Synchronization point (optional, helps debug communication issues)
        world.barrier();
    }

    // Gather results to the root process
    let mut final_result = vec![0.0; points + 1];
    let local_start = (rank as usize) * local_n;
    let local_end = local_start + local_n;

    if rank == 0 {
        final_result[local_start..local_end].copy_from_slice(&u_local[1..local_n + 1]);
        for i in 1..size {
            world.process_at_rank(i).receive_into(&mut final_result[(i as usize) * local_n..(i as usize + 1) * local_n]);
        }
    } else {
        world.process_at_rank(0).send(&u_local[1..local_n + 1]);
    }

    if rank == 0 {
        (final_result, start_time.elapsed().as_micros())
    } else {
        (vec![], start_time.elapsed().as_micros()) // Return an empty vector for non-root ranks
    }
}