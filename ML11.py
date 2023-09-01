import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


x=np.arange(-10,10,0.01)
y=[]
for i in x:
    y.append(i**2+2*i-1)
y=np.array(y)
plt.plot(x,y)
