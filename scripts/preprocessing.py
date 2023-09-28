#Author: Alvaro Pay Lozano

matplotlib.pyplot as plt
import numpy as np
import os
from interpolation import read_vtu, interpolate
import pickle
from stl import mesh
import glob

def generateBluff():

  bluff = []
  edge = np.arange(0,5,1)
  angle = np.arange(0,10,1)
  mach = [0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
  shape = [3, 4]
  ratio = [2, 3, 5, 6, 8]
  
  for s in shape:
      for ar in ratio:
          for a in angle:
              for e in edge:
                  for m in mach:
                      bluff.append(str(s) + str(0) + str(ar) + str(a) + str(e) + '_' + str(m))
  return bluff    
      
inp = generateBluff()

#Preprocessing inputs
x_range = (-1, 1.5)
y_range = (-2, 2)
xmin = x_range[0] 
xmax = x_range[1]
ymin = y_range[0]
ymax = y_range[1]
k = 5
resolution = 200
gpu_id = 4
x_center, y_center = (x_range[0]+x_range[1]) / 2, (y_range[0]+y_range[1]) / 2
width = x_range[1]-x_range[0]
height = 1

output_path = "PATH/dataX.pxl"
output_path_2 = "PATH/dataY.pkl"

#Geometry input
Ns, Nc, Nx, Ny = len(inp), 3, resolution, resolution
geo = []
cfd = []

for i, bluff in enumerate(inp):

  # Case
  bluff_stl = bluff.split('_')[0]
  stl_path = f"/local/disk/apay/test/input/{bluff}/constant/triSurface/bluff{bluff_stl}.stl"
  matching_folders = [f for f in glob.glob(f"/local/disk/apay/test/input/{bluff}/{bluff}/{bluff}_*") if os.path.isdir(f)]
  
  if not matching_folders:
    print(f"No matching folders found for {bluff}")
    continue
  vtu_path = os.path.join(matching_folders[0], "internal.vtu")
  
  if not os.path.exists(stl_path):
      print(f"File not found: {stl_path}")
      continue
      
  try:
      data = read_vtu(vtu_path)
  except ValueError:
        print(f"Diverged case detected for {bluff}. Skipping...")
        continue

  #STL input
  # read point coordinates from file
  points = mesh.Mesh.from_file(stl_path)[:, 0:3]
  
  # remove duplicate points of side surface at x=-1
  points = points[points[:, 2] > 0][:, 0:2]
  points = points[0::2]
  
  # loop for identifying end of first points loop
  for t in range(len(points) - 1):
      for j in range(t + 1, len(points)):
          if np.array_equal(points[t], points[j]):
                  idx = t  

  #remove extra
  points = points[idx:, :]

  #CFD data input
  mach = float(bluff.split("_")[-1])
        
  cfd_data, sdf_geo, flow_reg = interpolate(xmin, xmax, ymin, ymax, k, resolution, gpu_id, points, data, mach)
    
  # Append to the lists 
  geo_current = np.zeros((Nc, Nx, Ny))
  geo_current[0, :, :] = sdf_geo
  geo_current[1, :, :] = flow_reg
  geo_current[2, :, :] = flow_reg
  geo.append(geo_current)

  cfd_current = np.zeros((Nc, Nx, Ny))
  cfd_current[0, :, :] = cfd_data[:, 2].reshape(Nx, Ny)
  cfd_current[1, :, :] = cfd_data[:, 3].reshape(Nx, Ny)
  cfd_current[2, :, :] = cfd_data[:, 4].reshape(Nx, Ny)
  cfd.append(cfd_current)

# Convert the lists back to numpy arrays 
geo = np.array(geo)
cfd = np.array(cfd)

#Export to pickle
with open(output_path, 'wb') as f:
    pickle.dump(geo, f)
with open(output_path_2, 'wb') as f:
    pickle.dump(cfd, f)

