import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
import os

path = r"C:\\Users\\fastdaq\\Desktop\\04-15-2023\\fine/"
name = [i.name for i in os.scandir(path) if "img" in i.name]
img = np.hstack([np.load(path + i) for i in name])
img_i = simpson(img, axis=-1)

# %%
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(img_i[2:], cmap="cividis")
ax.set_aspect('equal')
plt.tight_layout()
