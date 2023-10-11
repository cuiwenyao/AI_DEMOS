import torch
import numpy as np
import matplotlib.pyplot as plt
import os
pwd_path=os.path.dirname(__file__)+"/"
T, D=1024, 4096
pe = torch.zeros(T, D)
position = torch.arange(0., T).unsqueeze(1)
div_term = torch.exp(torch.arange(0., D, 2) *-(np.log(10000.0) / D))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
print(pe.shape)
plt.figure(figsize=(12,8))
plt.pcolormesh(pe, cmap='viridis')
plt.xlabel('Embedding Dimensions')
plt.xlim((0, D))
plt.ylim((T,0))
plt.ylabel('Token Position')
plt.colorbar()
plt.show()
plt.savefig(os.path.join(pwd_path, "Sinusoidal.png"))