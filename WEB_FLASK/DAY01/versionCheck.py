import torch
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as ma
import torchvision
import torchaudio
from importlib.metadata import version
import flask


print(f"pytorch          {torch.__version__}")
print(f"torchvision      {torchvision.__version__}")
print(f"torchaudio       {torchaudio.__version__}")
print(f"pandas           {pd.__version__}")
print(f"numpy            {np.__version__}")
print(f"sklearn          {sk.__version__}")
print(f"matplotlib       {ma.__version__}")
print(f"flask            {version('flask')}")
