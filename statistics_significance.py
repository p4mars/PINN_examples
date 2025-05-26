import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

np.random.seed(42)

class StatisticEvaluation:
    def __init__(self, sim_data, exp_data, time_data):
        self.simulation_data = sim_data
        self.experimental_data = exp_data
        self.time_data = time_data
        
    def Metric_calc(self, x, y, metric):
        if metric == 'RMS':
            return np.sqrt(np.mean((x - y)**2))
        elif metric == 'MSE':
            return np.mean((x - y)**2)
        elif metric == 'MAE':
            return np.mean(np.abs(x - y))
        elif metric == 'MAPE':
            return np.mean(np.abs(x - y) / y) * 100
        
        
    def Bootstrap(self, n_iterations, metric):
        rms_normal = self.Metric_calc(self.simulation_data, self.experimental_data, metric)
        
        dictionary = {
            'time' : self.time_data,
            'data' : self.experimental_data
            }
        
        panda_data = pd.DataFrame(dictionary)
        
        bootstrap_RMS_lst = []
        
        #plt.figure()
        
        for _ in range(n_iterations):
          
            # Standard bootstrap: sampling with replacement
            sample_data = panda_data.sample(n=len(panda_data), replace=True, random_state=np.random.randint(0, len(panda_data))).sort_values(by='time')
            
            # Plot sampled data for visualization
            #plt.plot(sample_data['time'], sample_data['data'], 'o', alpha=0.5)  # Plotting with time axis
            bootstrap_RMS_lst.append(self.Metric_calc(self.simulation_data, sample_data['data'], metric))
        
        rms_lower = np.percentile(bootstrap_RMS_lst, 2.5)
        rms_upper = np.percentile(bootstrap_RMS_lst, 97.5)
        
        # Visualize Bootstrap Results
        plt.figure()
        plt.hist(bootstrap_RMS_lst, bins=50, alpha=0.7, label="Bootstrap RMS")
        plt.axvline(rms_normal, color='red', linestyle='--', label='Observed RMS')
        plt.axvline(rms_lower, color='green', linestyle='--', label='95% CI Lower')
        plt.axvline(rms_upper, color='green', linestyle='--', label='95% CI Upper')
        plt.legend()
        plt.xlabel("RMS Error")
        plt.ylabel("Frequency")
        plt.title("Bootstrap RMS Distribution")
        plt.show()
        
        return rms_normal, rms_lower, rms_upper

# Test example
time = np.linspace(-5, 5, 5000)
model = time**2 + 2*time + 7
measured_data = model + np.random.normal(0, 0.1, time.shape)
model_2 = model + 2

# Visualization
plt.figure()
plt.plot(time, model, label="True Model")
plt.plot(time, measured_data, 'o', label="Measured Data")
plt.legend()
plt.show()

# Statistical evaluation
stat_eval = StatisticEvaluation(model, measured_data, time)
rms_normal, rms_lower, rms_upper = stat_eval.Bootstrap(10000, 'MAPE')
print(f"Observed RMS: {rms_normal}")
print(f"95% Confidence Interval: [{rms_lower}, {rms_upper}]")
