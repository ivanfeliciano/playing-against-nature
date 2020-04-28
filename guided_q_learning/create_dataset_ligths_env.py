import os
import sys

import numpy as np 
import cv2

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import mujoco_py

from utils.lights_env_helper import powerset

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    for i in [5, 7, 9]:
        fullpath = os.path.join(os.path.dirname(__file__), 'env/assets', "arena_v2_" + str(i) + ".xml")
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))
        model = mujoco_py.load_model_from_path(fullpath)
        sim = mujoco_py.MjSim(model)
        for config in powerset(i):
            sim.model.light_active[:] = np.array(config)
            img = sim.render(width=32, height=32, camera_name="birdview")
            im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("./data/{}/{}.png".format(i, "".join([str(_) for _ in config])), im_rgb)