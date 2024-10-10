use std::time::Instant;

pub fn solve_tridiagonal(a: &Vec<f64>, b: &Vec<f64>, c: &Vec<f64>, d: &mut Vec<f64>) {
  let n = d.len();
  let mut alpha = vec![0.0; n]; // Вектор для коэффициентов прогонки (альфа)
  let mut beta = vec![0.0; n];  // Вектор для коэффициентов прогонки (бета)

  // Прямой ход
  alpha[0] = -c[0] / b[0];  // Вычисление первого значения альфа
  beta[0] = d[0] / b[0];  // Вычисление первого значения бета

  for i in 1..n {
      let denom = b[i] + a[i] * alpha[i - 1]; // Вычисляем знаменатель для текущей строки
      alpha[i] = -c[i] / denom; // Вычисляем следующее значение альфа
      beta[i] = (d[i] - a[i] * beta[i - 1]) / denom;  // Вычисляем следующее значение бета
  }

  // Обратный ход
  for i in (0..n - 1).rev() {
      d[i] = alpha[i] * d[i + 1] + beta[i]; // Обратный ход для вычисления решения
  }
}

pub fn solve_heat_equation(length: f64, temperature: f64, points: usize, dt: f64, time_steps: usize) -> (Vec<f64>, u128) {
  let start_time = Instant::now();

  let dx = length / (points as f64);
  let alpha = dt / (dx * dx);

  let mut u = vec![0.0; points + 1];
  let mut u_new = vec![0.0; points + 1];
  u[points] = temperature;

  let mut a = vec![0.0; points - 1];
  let mut b = vec![0.0; points - 1];
  let mut c = vec![0.0; points - 1];
  let mut d = vec![0.0; points - 1];

  // Заполняем коэффициенты
  for i in 0..points - 1 {
      a[i] = -alpha;
      b[i] = 1.0 + 2.0 * alpha;
      c[i] = -alpha;
  }

  // Итерации по времени
  for _ in 0..time_steps {
      for i in 1..points {
          d[i - 1] = u[i];  // Правая часть — это текущее значение температуры
      }

       // Применяем граничные условия
      d[0] += alpha * 0.0;  // Граничное условие на левой границе (U(0, t) = 0)
      d[points - 2] += alpha * temperature;  // Граничное условие на правой границе (U(L, t) = T)

      // Решаем систему уравнений методом прогонки
      solve_tridiagonal(&a, &b, &c, &mut d);

      // Обновляем температуру на следующем шаге по времени
      for i in 1..points {
          u_new[i] = d[i - 1];  // Копируем значения из решения
      }

      u_new[points] = temperature;  // Граничное условие справа

      u.copy_from_slice(&u_new);  // Копируем новое решение на текущий шаг
  }

  (u, start_time.elapsed().as_micros())
}