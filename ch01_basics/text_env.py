import torch
import numpy as np
import matplotlib

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("GPU可用:", torch.cuda.is_available())