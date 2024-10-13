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
- [x] Через MPI (Message Passing Interface)

![image](https://github.com/user-attachments/assets/09e8c687-3bc9-4225-a7b4-c357f3129037)

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

# Запуск проекта с MPI
mpiexec -n 1 cargo run
```

### Установка MPI
MS MPI: https://www.microsoft.com/en-us/download/details.aspx?id=100593  
- После установки добавить в путь MSMPI_INC (C:\Program Files (x86)\Microsoft SDKs\MPI\Include) и MSMPI_LIB64 (C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64)

LLVM: https://releases.llvm.org/
- После установки добавить в путь LIBCLANG_PATH (C:\Program Files\LLVM\bin)

```bash
# Проверка версии Clang
clang --version

# Компиляция проекта
cargo build

# Запуск проекта
mpiexec -n 1 cargo run
``` 

### Зависимости
egui: https://github.com/emilk/egui  
egui documentation: https://docs.rs/egui/latest/egui/  
rayon: https://github.com/rayon-rs/rayon  
rayon documentation: https://docs.rs/rayon/latest/rayon/  
rsmpi: https://github.com/rsmpi/rsmpi  
rsmpi documentation: https://rsmpi.github.io/rsmpi/mpi/index.html  

### Лицензия
Эта программа распространяется под лицензией MIT License. Пожалуйста, прочтите файл лицензии, чтобы узнать об условиях использования.
