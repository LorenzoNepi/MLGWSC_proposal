import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

def data_reader(data_path):
    return

test_path = f"/home/lorenzo02/MLGWSC_proposal/data"
background_path = os.path.join(test_path, "background.hdf")
foreground_path = os.path.join(test_path, "foreground.hdf")
injections_path = os.path.join(test_path, "injections.hdf")

print("BACKGROUND HDF5 FILE\n")

"""
with h5py.File(background_path) as f:
    print(f"Root group keys: {f.keys()}")
    f_attr=dict(f.attrs.items())
    print("Root group attributes:")
    print(*f_attr.items(), sep="\n")

    H1_group = f["H1"]
    print(f"\nH1 group keys: {H1_group.keys()}")
    print(f"H1 group attributes: {H1_group.attrs.keys()}")

    L1_group = f["L1"]
    print(f"\nL1 group keys: {L1_group.keys()}")
    print(f"L1 group attributes: {L1_group.attrs.keys()}")

print("\nFOREGROUND HDF5 FILE\n")
"""

with h5py.File(foreground_path) as f:
    print(f"Root group keys: {f.keys()}")
    f_attr=dict(f.attrs.items())
    print("Root group attributes:")
    print(*f_attr.items(), sep="\n")

    H1_group = f["H1"]
    print(f"\nH1 group keys: {H1_group.keys()}")
    print(f"H1 group attributes: {H1_group.attrs.keys()}")

    L1_group = f["L1"]
    print(f"\nL1 group keys: {L1_group.keys()}")
    print(f"L1 group attributes: {L1_group.attrs.keys()}")

    data = np.array(L1_group['1238205077'])
    time_interval = 1/float(f.attrs['sample_rate'])
    time = np.arange(0, 300, time_interval)


"""
print("\nINJECTIONS HDF5 FILE\n")

with h5py.File(injections_path) as f:
    print(f"Root group keys: {f.keys()}")
    f_attr=dict(f.attrs.items())
    print("Root group attributes:")
    #print(f_attr, sep="\n")
    print(f.attrs.keys())
    print(len(f["chirp_distance"][()]))
    print(len(f["tc"][()]))
"""

f = h5py.File(foreground_path)

L1_group = f["L1"]
data = np.array(L1_group['1238205077'])
time_interval = 1/float(f.attrs['sample_rate'])
time = np.arange(0, 300, time_interval)
plt.plot(time, data)
plt.show()

f.close()