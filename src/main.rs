use eframe::egui;
mod st_heat_solver;

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
            points: 100,
            dt: 0.01,
            time_steps: 100,
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
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Введите параметры для Лабораторной 1:");

            // Поля для ввода значений
            ui.add(eframe::egui::Slider::new(&mut self.length, 1.0..=100.0).text("Длина стержня (L)"));
            ui.add(eframe::egui::Slider::new(&mut self.temp_right, 10.0..=500.0).text("Температура справа (T)"));
            ui.add(eframe::egui::Slider::new(&mut self.points, 10..=500).text("Количество точек (N)"));
            ui.add(eframe::egui::Slider::new(&mut self.dt, 0.001..=0.1).text("Шаг по времени (dt)"));
            ui.add(eframe::egui::Slider::new(&mut self.time_steps, 10..=1000).text("Количество временных шагов"));

            // Кнопка для расчёта
            if ui.button("Однопоточное решение").clicked() {
                self.solve_lab_1();
            }

            // Вывод гистограммы, если есть результат
            if let Some(result) = &self.result {
                let max_value = result.iter().cloned().fold(f64::NAN, f64::max);

                let available_size = ui.available_size();
                for (i, value) in result.iter().enumerate() {
                    let height = value / max_value * available_size.y as f64;
                    ui.painter().rect_filled(
                        eframe::egui::Rect::from_min_size(
                            eframe::egui::pos2(i as f32 * 5.0, available_size.y - height as f32),
                            eframe::egui::vec2(4.0, height as f32),
                        ),
                        0.0,
                        eframe::egui::Color32::from_rgb(100, 150, 250),
                    );
                }
            }

            // Вывод времени расчета, если оно доступно
            if let Some(time) = self.compute_time {
                ui.label(format!("Время расчета: {} мс", time));
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