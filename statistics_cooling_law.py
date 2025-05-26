import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare
import pandas as pd


# Data directory
file_path = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Error_file/20_points_cooling_processed.csv'

# Open data
data = pd.read_csv(file_path)

avg_RMSE = np.array(data["Average error"].apply(eval).tolist())
avg_max = np.array(data["Maximum error"].apply(eval).tolist())


# Stored LOO-CV results
avg_RMSE_pinn = avg_RMSE[:, 0]
avg_RMSE_data = avg_RMSE[:, 1]
avg_RMSE_num = avg_RMSE[:, 2]

max_errors_pinn = avg_max[:, 0]
max_errors_data = avg_max[:, 1]
max_errors_num = avg_max[:, 2]

# Compute differences
diff_pinn_data = max_errors_pinn - max_errors_data
diff_pinn_numerical = max_errors_pinn - max_errors_num
diff_data_numerical = max_errors_data - max_errors_num

# Normality test on the differences
_, p_shapiro_pinn_data = shapiro(diff_pinn_data)
_, p_shapiro_pinn_numerical = shapiro(diff_pinn_numerical)
_, p_shapiro_data_numerical = shapiro(diff_data_numerical)

print(f"Shapiro-Wilk p-values: PINN vs Data: {p_shapiro_pinn_data:.5f}, "
      f"PINN vs Numerical: {p_shapiro_pinn_numerical:.5f}, "
      f"Data vs Numerical: {p_shapiro_data_numerical:.5f}")

stat, p_value = friedmanchisquare(max_errors_pinn, max_errors_data, max_errors_num)

print(f"Friedman Test p-value: {p_value:.5f}")



alpha = 0.05 / 3  # Bonferroni correction for 3 comparisons

p1 = wilcoxon(max_errors_pinn, max_errors_data).pvalue
p2 = wilcoxon(max_errors_pinn, max_errors_num).pvalue
p3 = wilcoxon(max_errors_data, max_errors_num).pvalue

print(f"Wilcoxon Signed-Rank Tests (Bonferroni-corrected alpha = {alpha:.5f}):")
print(f"PINN vs Data: p = {p1:.5f} {'(Significant)' if p1 < alpha else '(Not Significant)'}")
print(f"PINN vs Numerical: p = {p2:.5f} {'(Significant)' if p2 < alpha else '(Not Significant)'}")
print(f"Data vs Numerical: p = {p3:.5f} {'(Significant)' if p3 < alpha else '(Not Significant)'}")


