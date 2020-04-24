from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import mujoco_py
import os
import numpy as np
import cv2


def powerset(n):
    powerset = []
    for i in range(1 << n):
        bitset = [0 for i in range(n)]
        for j in range(n):
            bitset[j] = 1 if (i & (1 << j)) else 0
        powerset.append(bitset)
    return powerset
for i in [5, 7, 9]:
    fullpath = os.path.join(os.path.dirname(__file__), 'assets', "arena_v2_"+str(i)+".xml")
    if not os.path.exists(fullpath):
        raise IOError('File {} does not exist'.format(fullpath))
    model = mujoco_py.load_model_from_path(fullpath)
    sim = mujoco_py.MjSim(model)
    for config in powerset(i):
        sim.model.light_active[:] = np.array(config)
        img = sim.render(width=32, height=32, camera_name="birdview")
        im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./data/{}/{}.png".format(i, "".join([str(_) for _ in config])), im_rgb)