# Imported modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def RMSE(y1, y2):
    return np.sqrt(np.mean((y1 - y2)**2))

# Import data dir
data_dir_t80 = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Ladson/CL_CD_data/alpha_cl_cd_M_015_t80.txt'
data_dir_t120 = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Ladson/CL_CD_data/alpha_cl_cd_M_015_t120.txt'
data_dir_t180 = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Ladson/CL_CD_data/alpha_cl_cd_M_015_t180.txt'
data_dir_cfd = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Ladson/Cl_cd_data_CFD/n0012clcd_cfl3d_sst.dat'

# Obtain data
data_t80 = np.genfromtxt(data_dir_t80)
data_t120 = np.genfromtxt(data_dir_t120)
data_t180 = np.genfromtxt(data_dir_t180)
data_cfd = np.genfromtxt(data_dir_cfd)


alpha_t80 = data_t80[:, 0]
cl_t80 = data_t80[:, 1]
cd_t80 = data_t80[:, 2]

alpha_t120 = data_t120[:, 0]
cl_t120 = data_t120[:, 1]
cd_t120 = data_t180[:, 2]

alpha_t180 = data_t180[:, 0]
cl_t180 = data_t180[:, 1]
cd_t180 = data_t180[:, 2]

alpha_cfd = data_cfd[:, 0]
cl_cfd = data_cfd[:, 1]
cd_cfd = data_cfd[:, 2]

# Interploation of 180 data
interp_cl_180 = interp1d(alpha_t180, cl_t180, kind='cubic', fill_value="extrapolate")

alpha_0 = interp_cl_180(0.0169)
alpha_10 = interp_cl_180(10.0254)
alpha_15 = interp_cl_180(15.0299)

diff_0 = RMSE(alpha_0, cl_cfd[0])
diff_10 = RMSE(alpha_10, cl_cfd[1])
diff_15 = RMSE(alpha_15, cl_cfd[2])

# Plot results
plt.plot(alpha_t80, cl_t80, label='t80')
plt.plot(alpha_t120, cl_t120, label='t120')
plt.plot(alpha_t180, cl_t180, label='t180')
plt.plot(alpha_cfd, cl_cfd, label='CFD')
plt.legend()
plt.show()