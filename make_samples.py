from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
X, y = mnist.data[60000:], mnist.target[60000:].astype(int)  # test split only

Path("samples").mkdir(exist_ok=True)
for i in range(10):
    plt.imsave(f"samples/test_{i}_label{y[i]}.png", X[i].reshape(28, 28), cmap="gray")
print("Saved 10 images in ./samples")