import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from recap.trajectory import to_numpy

file_path = os.path.join(os.path.dirname(__file__), f"../examples/motions/amass/h1/amass_h1_dance_waltz10_poses.pkl")
with open(file_path, "rb") as f:
    motion = pickle.load(f)
motion = to_numpy(motion)

data_dt = motion["dt"]
pos = motion["transforms"]["imu"]["position"]
vel = motion["transforms"]["imu"]["linear_velocity"]
integrated_pos = pos[0, :2] + np.cumsum(vel[:, :2] * data_dt, axis=0)

plt.plot(pos[:, 0], pos[:, 1], label="positions")
plt.plot(integrated_pos[:, 0], integrated_pos[:, 1], label="integrated")
plt.legend()


def quat_to_euler(quat):
    w, x, y, z = (
        quat[:, 0],
        quat[:, 1],
        quat[:, 2],
        quat[:, 3],
    )
    ysqr = y * y
    X = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + ysqr))
    Y = -np.pi / 2 + 2 * np.arctan2(np.sqrt(1 + 2.0 * (w * y - z * x)), np.sqrt(1 - 2.0 * (w * y - z * x)))
    Z = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (ysqr + z * z))
    return np.vstack([X, Y, Z]).T


quat = motion["transforms"]["imu"]["quaternion"]
angvel = motion["transforms"]["imu"]["angular_velocity"]

euler_angles = quat_to_euler(quat)
integrated_angles = euler_angles[0] + np.cumsum(angvel * data_dt, axis=0)


t = np.arange(0, euler_angles.shape[0] * data_dt, data_dt)
for i, name in enumerate(["roll", "pitch", "yaw"]):
    plt.figure()
    plt.plot(t, euler_angles[:, i], label=name)
    plt.plot(t, integrated_angles[:, i], label=f"{name} int")
    plt.legend()
plt.show()
