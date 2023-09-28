import torch
import pickle
import matplotlib
import numpy as np
from deepcfd.models.UNetEx import UNetEx
from matplotlib import pyplot as plt

matplotlib.use('tkagg')

model_filename = "PATH/model.pt"
data_x_filename = "PATH/dataX.pkl"
data_y_filename = "PATH/dataY.pkl"

index = 771
kernel_size = 5
filters = [8, 16, 32, 32]
bn = False
wn = False

extent = [0, 200, 0, 200]

min_u_x = -0.2
max_u_x = 1.5
min_u_y = -0.3
max_u_y = 0.5
min_u_x_error = 0
max_u_x_error = 0.05
min_u_y_error = 0
max_u_y_error = 0.05
min_p = 1.222
max_p = 1.2235
min_p_error = 0
max_p_error = 0.0075

model = UNetEx(
    3,
    3,
    filters=filters,
    kernel_size=kernel_size,
    batch_norm=bn,
    weight_norm=wn
)

x = pickle.load(open(data_x_filename, "rb"))
y = pickle.load(open(data_y_filename, "rb"))

print(x.shape)

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

truth = y[index:(index + 1)].cpu().detach().numpy()
inputs = x[index:(index + 1)].cpu().detach().numpy()

plt.figure()
fig = plt.gcf()
fig.set_size_inches(15, 10)
fig.canvas.manager.window.wm_attributes('-topmost', 0)

def visualize_results():
    results = torch.load((model_filename), map_location=torch.device('cpu'))

    # Extract the state dictionary for the model
    state_dict = {k: v for k, v in results.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(state_dict)
    out = model(x[index:(index + 1)]).detach().numpy()
    error = abs(out - truth)
    print(truth.shape)

    fig.suptitle('Results from the last saved epoch: ')

    plt.subplot(3, 3, 1)
    plt.ylabel('Ux [m/s]', fontsize=18)
    plt.title('bluffFOAM', fontsize=18)
    plt.imshow(truth[0, 0, :, :].T, cmap='jet', vmin = min_u_x, vmax = max_u_x, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')
    plt.subplot(3, 3, 2)
    plt.title('TurbulentDeepCFD', fontsize=18)
    plt.imshow(out[0, 0, :, :].T, cmap='jet', vmin = min_u_x, vmax = max_u_x, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')
    plt.subplot(3, 3, 3)
    plt.title('error', fontsize=18)
    plt.imshow(error[0, 0, :, :].T, cmap='jet', vmin = min_u_x_error, vmax = max_u_x_error, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')

    plt.subplot(3, 3, 4)
    plt.ylabel('Uy [m/s]', fontsize=18)
    plt.imshow(truth[0, 1, :, :].T, cmap='jet', vmin = min_u_y, vmax = max_u_y, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')
    plt.subplot(3, 3, 5)
    plt.imshow(out[0, 1, :, :].T, cmap='jet', vmin = min_u_y, vmax = max_u_y, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')
    plt.subplot(3, 3, 6)
    plt.imshow(error[0, 1, :, :].T, cmap='jet', vmin = min_u_y_error, vmax = max_u_y_error, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')

    plt.subplot(3, 3, 7)
    plt.ylabel('p [m2/s2]', fontsize=18)
    plt.imshow(truth[0, 2, :, :].T, cmap='jet', vmin = min_p, vmax = max_p, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')
    plt.subplot(3, 3, 8)
    plt.imshow(out[0, 2, :, :].T, cmap='jet', vmin = min_p, vmax = max_p, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')
    plt.subplot(3, 3, 9)
    plt.imshow(error[0, 2, :, :].T, cmap='jet', vmin = min_p_error, vmax = max_p_error, origin='lower', extent=extent)
    plt.colorbar(orientation='vertical')

    plt.draw()
    plt.show()

visualize_results()
