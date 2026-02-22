# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

# Устанавливаем seed для воспроизводимости результатов
np.random.seed(42)

# Параметры задачи
n = 25  # Объем выборки
lambda_exp = 1.0  # Параметр экспоненциального распределения (масштаб = 1/lambda)

# Генерируем исходную выборку из экспоненциального распределения
# В numpy scale = 1/lambda. При lambda=1, scale=1.
sample = np.random.exponential(scale=1.0, size=n)

print("="*60)
print("ИСХОДНАЯ ВЫБОРКА (n=25):")
print(sorted(np.round(sample, 3)))
print("="*60)

# ============================================================
# ЗАДАНИЕ a) Точечные статистики: мода, медиана, размах, асимметрия
# ============================================================
print("\nЗАДАНИЕ a) ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ")

# Мода: для непрерывных данных и малой выборки понятие моды условно.
# Мы либо находим значение, встречающееся чаще всего (если есть повторения),
# либо, что чаще, строим гистограмму и ищем модальный интервал.
# Для простоты используем гистограмму позже.
# Здесь просто скажем, что модальный интервал мы увидим на гистограмме.
# Формально можно использовать argmax гистограммы, но это не точно.
print("Мода: (определяется по гистограмме - интервал с макс. частотой)")

# Медиана
median_value = np.median(sample)
print(f"Медиана: {median_value:.4f} (теоретическая ln(2)≈0.6931)")

# Размах
range_value = np.max(sample) - np.min(sample)
print(f"Размах: {np.min(sample):.2f} ... {np.max(sample):.2f} => {range_value:.4f}")

# Коэффициент асимметрии (Skewness)
# Используем метод moments из scipy для несмещенной оценки? 
# st.skew дает точечную оценку.
skew_value = st.skew(sample)
print(f"Коэффициент асимметрии: {skew_value:.4f} (теоретический = 2)")

# ============================================================
# ЗАДАНИЕ b) Эмпирическая функция распределения, гистограмма, boxplot
# ============================================================
print("\nЗАДАНИЕ b) ВИЗУАЛИЗАЦИЯ (графики сохранены в файлы)")

# Создаем фигуру с тремя подграфиками
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- 1. Эмпирическая функция распределения (ECDF) ---
# Сортируем данные
x_sorted = np.sort(sample)
# Эмпирическая вероятность (от 0 до 1)
y_ecdf = np.arange(1, n+1) / n

axes[0].step(x_sorted, y_ecdf, where='post', label='ECDF')
axes[0].set_title('Эмпирическая функция распределения')
axes[0].set_xlabel('x')
axes[0].set_ylabel('F(x)')
axes[0].grid(True, linestyle='--', alpha=0.7)
# Для сравнения нарисуем теоретическую функцию распределения экспоненциального закона
x_theor = np.linspace(0, np.max(sample)+1, 100)
y_theor = 1 - np.exp(-x_theor)  # CDF экспоненциального распределения
axes[0].plot(x_theor, y_theor, 'r--', label='Теоретическая CDF (exp(1))')
axes[0].legend()

# --- 2. Гистограмма ---
axes[1].hist(sample, bins=8, density=True, alpha=0.6, color='green', edgecolor='black', label='Гистограмма')
# Добавим теоретическую плотность для сравнения
x_pdf = np.linspace(0, np.max(sample)+1, 100)
y_pdf = np.exp(-x_pdf)  # PDF экспоненциального распределения
axes[1].plot(x_pdf, y_pdf, 'r-', label='Теоретическая PDF (exp(1))')
axes[1].set_title('Гистограмма')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Плотность')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.7)

# --- 3. Boxplot ---
axes[2].boxplot(sample, vert=True, patch_artist=True)
axes[2].set_title('Ящик с усами (Boxplot)')
axes[2].set_ylabel('Значения')
axes[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('task_b_plots.png', dpi=150)
plt.show()
print("Графики сохранены в task_b_plots.png")

# ============================================================
# ЗАДАНИЕ c) Сравнение ЦПТ и бутстрапа для среднего арифметического
# ============================================================
print("\nЗАДАНИЕ c) ОЦЕНКА ПЛОТНОСТИ СРЕДНЕГО: ЦПТ vs БУТСТРАП")

# --- Оценка по ЦПТ ---
# По ЦПТ: X_bar ~ N(mu, sigma^2/n). Мы знаем mu=1, sigma^2=1.
mu_theor = 1.0
sigma_theor = 1.0
std_error = sigma_theor / np.sqrt(n)  # стандартная ошибка среднего = 1/5 = 0.2

print(f"По ЦПТ: Среднее распределено нормально с мат. ожиданием {mu_theor} и стд. ошибкой {std_error:.3f}")

# --- Бутстраповская оценка ---
B = 5000  # Количество бутстрап-выборок
bootstrap_means = []

for _ in range(B):
    # Генерируем бутстрап-выборку (с возвращением) того же размера n
    boot_sample = np.random.choice(sample, size=n, replace=True)
    # Считаем среднее
    bootstrap_means.append(np.mean(boot_sample))

# Преобразуем в массив numpy
bootstrap_means = np.array(bootstrap_means)

print(f"Бутстрап: Среднее бутстраповских средних = {np.mean(bootstrap_means):.4f}")
print(f"Бутстрап: Стд. ошибка (стд. бутстраповских средних) = {np.std(bootstrap_means):.4f}")

# --- Визуализация сравнения ---
plt.figure(figsize=(10, 6))

# Гистограмма бутстраповских средних
plt.hist(bootstrap_means, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='black', label='Бутстрап (гистограмма)')

# Оценка плотности по бутстрапу (KDE)
sns.kdeplot(bootstrap_means, color='blue', linewidth=2, label='Бутстрап (KDE)')

# Теоретическая плотность по ЦПТ (нормальное распределение)
x_c = np.linspace(0, 2, 200)
y_cpt = st.norm.pdf(x_c, loc=mu_theor, scale=std_error)
plt.plot(x_c, y_cpt, 'r--', linewidth=2, label='ЦПТ (нормальное)')

# Вертикальная линия на выборочном среднем
sample_mean = np.mean(sample)
plt.axvline(sample_mean, color='green', linestyle=':', label=f'Среднее выборки = {sample_mean:.3f}')

plt.title('Сравнение оценок плотности распределения среднего')
plt.xlabel('Значение среднего')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('task_c_mean_comparison.png', dpi=150)
plt.show()
print("График сохранен в task_c_mean_comparison.png")

# ============================================================
# ЗАДАНИЕ d) Бутстрап для коэффициента асимметрии и P(skew=1)
# ============================================================
print("\nЗАДАНИЕ d) БУТСТРАП ДЛЯ КОЭФФИЦИЕНТА АСИММЕТРИИ")

bootstrap_skews = []

for _ in range(B):
    boot_sample = np.random.choice(sample, size=n, replace=True)
    # Считаем асимметрию для каждой бутстрап-выборки
    # bias=False дает несмещенную оценку (умножение на (n*(n-1))^0.5 / (n-2) )
    # Для согласованности с st.skew(sample) оставим как есть.
    bootstrap_skews.append(st.skew(boot_sample))

bootstrap_skews = np.array(bootstrap_skews)

print(f"Бутстрап: Среднее значение коэффициента асимметрии = {np.mean(bootstrap_skews):.4f}")
print(f"Бутстрап: Стд. ошибка = {np.std(bootstrap_skews):.4f}")

# --- Оценка "вероятности того, что skew = 1" ---
# Для непрерывной случайной величины вероятность точного равенства = 0.
# Обычно в таких задачах просят найти значение плотности в точке 1.
# Используем KDE для оценки плотности.

# Построим KDE по бутстраповским значениям
kde_skew = st.gaussian_kde(bootstrap_skews)

# Значение плотности в точке x=1
density_at_1 = kde_skew.evaluate(1.0)[0]
print(f"Значение плотности распределения асимметрии в точке 1: {density_at_1:.4f} (это не вероятность!)")
print("P(skew = 1) = 0, т.к. распределение непрерывно.")

# Альтернативный "игрушечный" ответ: доля значений, попавших в малую окрестность 1
epsilon = 0.05
prob_near_1 = np.sum(np.abs(bootstrap_skews - 1) < epsilon) / B
print(f"Доля значений в окрестности (1±{epsilon}): {prob_near_1:.4f}")

# --- Визуализация распределения бутстраповских skew ---
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_skews, bins=30, density=True, alpha=0.5, color='orange', edgecolor='black', label='Бутстрап skew')
sns.kdeplot(bootstrap_skews, color='red', linewidth=2, label='KDE')
plt.axvline(1, color='blue', linestyle='--', label='x = 1')
plt.axvline(skew_value, color='green', linestyle=':', label=f'Выборочный skew = {skew_value:.3f}')
plt.title('Бутстраповское распределение коэффициента асимметрии')
plt.xlabel('Коэффициент асимметрии')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('task_d_skew_bootstrap.png', dpi=150)
plt.show()
print("График сохранен в task_d_skew_bootstrap.png")

# ============================================================
# ЗАДАНИЕ e) Сравнение распределения медианы: теория (симуляция) vs бутстрап
# ============================================================
print("\nЗАДАНИЕ e) СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ МЕДИАНЫ")

# --- 1. Истинное распределение медианы (симуляция Монте-Карло) ---
# Мы можем приблизить его, сгенерировав много выборок напрямую из распределения.
N_sim = 10000  # Количество симулированных выборок
medians_sim = []

for _ in range(N_sim):
    # Генерируем выборку напрямую из экспоненциального распределения
    sim_sample = np.random.exponential(scale=1.0, size=n)
    medians_sim.append(np.median(sim_sample))

medians_sim = np.array(medians_sim)

print(f"Симуляция (N={N_sim}): Среднее значение медианы = {np.mean(medians_sim):.4f}")
print(f"Симуляция: Стд. ошибка = {np.std(medians_sim):.4f}")

# --- 2. Бутстраповская оценка распределения медианы (по одной выборке) ---
bootstrap_medians = []

for _ in range(B):
    boot_sample = np.random.choice(sample, size=n, replace=True)
    bootstrap_medians.append(np.median(boot_sample))

bootstrap_medians = np.array(bootstrap_medians)

print(f"Бутстрап (B={B}): Среднее значение медианы = {np.mean(bootstrap_medians):.4f}")
print(f"Бутстрап: Стд. ошибка = {np.std(bootstrap_medians):.4f}")

# --- Визуализация сравнения ---
plt.figure(figsize=(10, 6))

# Гистограмма симулированных медиан (приближенно "истинное" распределение)
plt.hist(medians_sim, bins=30, density=True, alpha=0.4, color='lightgreen', edgecolor='black', label='Симуляция (истинное)')

# Гистограмма бутстраповских медиан
plt.hist(bootstrap_medians, bins=30, density=True, alpha=0.4, color='lightcoral', edgecolor='black', label='Бутстрап (по одной выборке)')

# KDE для наглядности
sns.kdeplot(medians_sim, color='green', linewidth=2, label='KDE (симуляция)')
sns.kdeplot(bootstrap_medians, color='red', linewidth=2, label='KDE (бутстрап)')

# Вертикальная линия для выборочной медианы
plt.axvline(median_value, color='blue', linestyle='--', label=f'Выборочная медиана = {median_value:.3f}')

plt.title('Сравнение распределений медианы: симуляция vs бутстрап')
plt.xlabel('Медиана')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('task_e_median_comparison.png', dpi=150)
plt.show()
print("График сохранен в task_e_median_comparison.png")

print("\n" + "="*60)
print("РЕШЕНИЕ ВЫПОЛНЕНО. Все графики сохранены в PNG файлы.")
print("="*60)
