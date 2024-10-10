# heat-solver
## Численное решение краевой задачи для уравнения теплопроводности неявной четырехточечной конечно-разностной схемой
### Задача
Дано одномерное уравнение теплопроводности для стержня
- x - переменная по длине стержня;
- t - переменная по времени;
- L - длина стержня (задается пользователем);
- T - температура (задается пользователем).

𝑈𝑡 = 𝑈𝑥𝑥  

начальное условие:  

𝑈(𝑥,𝑡 = 0) = 0;  

граничные условия:  

𝑈(𝑥 = 0,𝑡) = 0;  
𝑈(𝑥 = 𝐿,𝑡) = 𝑇;

Требуется написать программу численного решения краевой задачи для уравнения теплопроводности неявной четырехточечной конечно-разностной схемой:
- [x] Однопоточно
- [x] Многопоточно
- [ ] Через MPI (Message Passing Interface)

![image](https://github.com/user-attachments/assets/bf887ee7-aecd-4f3d-ae31-f53ab0cb9f1b)

### Как запустить
```bash
# Установка Rust и Cargo: https://www.rust-lang.org/tools/install

# Проверка версии Rust
rustc --version

# Компиляция проекта
cargo build
# или
cargo build --release

# Запуск проекта
cargo run
```

### Установка MPI
MS MPI: https://www.microsoft.com/en-us/download/details.aspx?id=100593  
- После установки добавить в путь MSMPI_INC (C:\Program Files (x86)\Microsoft SDKs\MPI\Include) и MSMPI_LIB64 (C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64)

LLVM: https://releases.llvm.org/
- После установки добавить в путь LIBCLANG_PATH (C:\Program Files\LLVM\bin)

```bash
# Запуск проекта
mpiexec -n 4 target/release/heat_solver
```

### Ресурсы
egui: https://github.com/emilk/egui  
egui documentation: https://docs.rs/egui/latest/egui/  

### Лицензия
Эта программа распространяется под лицензией MIT License. Пожалуйста, прочтите файл лицензии, чтобы узнать об условиях использования.
