use eframe::egui;
mod st_heat_solver;
mod mt_heat_solver;
mod mpi_heat_solver;

struct MyApp {
    length: f64,
    temp_right: f64,
    points: usize,
    dt: f64,
    time_steps: usize,
    result: Option<Vec<f64>>,
    compute_time: Option<u128>
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            length: 10.0,
            temp_right: 100.0,
            points: 100000,
            dt: 0.01,
            time_steps: 800,
            result: None,
            compute_time: None
        }
    }
}

impl MyApp {
    fn solve_lab_1(&mut self) {
        let (solution, time) = st_heat_solver::solve_heat_equation(
            self.length,
            self.temp_right,
            self.points,
            self.dt,
            self.time_steps,
        );
        self.result = Some(solution);
        self.compute_time = Some(time);
    }

    fn solve_lab_2(&mut self) {
        let (solution, time) = mt_heat_solver::solve_heat_equation(
            self.length,
            self.temp_right,
            self.points,
            self.dt,
            self.time_steps,
        );
        self.result = Some(solution);
        self.compute_time = Some(time);
    }

    fn solve_lab_3(&mut self) {
        let length = self.length;
        let temp_right = self.temp_right;
        let points = self.points;
        let dt = self.dt;
        let time_steps = self.time_steps;

        // Create a separate thread for the MPI computation
        let handle = std::thread::spawn(move || {
            mpi_heat_solver::solve_heat_equation_mpi(length, temp_right, points, dt, time_steps)
        });

        // Wait for the thread to finish and collect the result
        match handle.join() {
            Ok((solution, time)) => {
                self.result = Some(solution);
                self.compute_time = Some(time);
            }
            Err(_) => {
                // Handle the error case
                eprintln!("Error occurred during MPI computation.");
            }
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Виджет панели управления
        egui::Window::new("Параметры задачи")
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                // Поля для ввода значений
                ui.add(egui::Slider::new(&mut self.temp_right, 10.0..=500.0).text("Температура справа (T)"));
                ui.add(egui::Slider::new(&mut self.points, 10..=100000).text("Количество точек (N)"));
                ui.add(egui::Slider::new(&mut self.dt, 0.001..=0.1).text("Шаг по времени (dt)"));
                ui.add(egui::Slider::new(&mut self.time_steps, 10..=10000).text("Количество временных шагов"));

                // Кнопка для расчёта
                if ui.button("Однопоточное решение").clicked() {
                    self.solve_lab_1();
                }

                if ui.button("Многопоточное решение").clicked() {
                    self.solve_lab_2();
                }

                if ui.button("MPI решение").clicked() {
                    self.solve_lab_3();
                }
            });

        // Гистограмма
        // egui::CentralPanel::default().show(ctx, |ui| {
        //     if let Some(result) = &self.result {
        //         let max_value = result.iter().cloned().fold(f64::NAN, f64::max);
        //         let available_size = ui.available_size();

        //         let bar_width = available_size.x / result.len() as f32;
        //         for (i, value) in result.iter().enumerate() {
        //             let height = value / max_value * available_size.y as f64;
        //             ui.painter().rect_filled(
        //                 egui::Rect::from_min_size(
        //                     egui::pos2(i as f32 * bar_width, available_size.y - height as f32),
        //                     egui::vec2(bar_width - 1.0, height as f32),  // Небольшой промежуток между столбцами
        //                 ),
        //                 0.0,
        //                 egui::Color32::from_rgb(245, 134, 141),
        //             );
        //         }
        //     }

        //     // Время расчёта
        //     if let Some(time) = self.compute_time {
        //         ui.label(format!("Время расчета: {} мкс", time));
        //         ui.label(format!("=~ {} мс", time / 1000));
        //         ui.label(format!("=~ {} с", time / 1000000));
        //     }
        // });

        // Линейный график
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(result) = &self.result {
                let max_value = result.iter().cloned().fold(f64::NAN, f64::max);
                let available_size = ui.available_size();

                // Вычисление ширины между точками на графике
                let point_spacing = available_size.x / (result.len() - 1) as f32;
                let mut previous_point = None;

                // Итерация по значениям массива, исключая последнюю точку
                for (i, value) in result.iter().take(result.len() - 1).enumerate() {
                    let normalized_value = value / max_value;
                    let current_point = egui::pos2(
                        i as f32 * point_spacing,
                        available_size.y - (normalized_value * available_size.y as f64) as f32,
                    );

                    if let Some(prev) = previous_point {
                        // Рисование линии между предыдущей и текущей точкой
                        ui.painter().line_segment(
                            [prev, current_point],
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(245, 134, 141)),
                        );
                    }

                    previous_point = Some(current_point);
                }
            }

            // Время расчёта
            if let Some(time) = self.compute_time {
                ui.label(format!("Время расчета: {} мкс", time));
                ui.label(format!("=~ {} мс", time / 1000));
                ui.label(format!("=~ {} с", time / 1000000));
            }
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Суперкомпьютерные вычисления Л1-3 ИДМ-23-08 Кочановский",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}