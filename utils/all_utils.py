import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def prepare_data(df, target_col="y"):
  X = df.drop(target_col, axis=1)
  y = df[target_col]
  return X,y


def save_plot(df, model, filename="plot.png", plot_dir="plots"):

  def _create_base_plot(df):
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="coolwarm")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)

    figure = plt.gcf()
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classifier, resolution=0.02):
    colors = ("cyan", "lightgreen")
    cmap = ListedColormap(colors=colors)

    X = X.values # as an array
    X1 = X[:, 0]
    X2 = X[:, 1]

    X1_min, X1_max = X1.min() - 1, X1.max() + 1
    X2_min, X2_max = X2.min() - 1, X2.max() + 1

    XX1, XX2 = np.meshgrid(np.arange(X1_min, X1_max, resolution),
                           np.arange(X2_min, X2_max, resolution)
                           )
    
    yhat = classifier.predict(np.array([XX1.ravel(), XX2.ravel()]).T)
    yhat = yhat.reshape(XX1.shape)

    plt.contourf(XX1, XX2, yhat, alpha=0.3, cmap=cmap)
    plt.xlim(XX1.min(), XX1.max())
    plt.ylim(XX2.min(), XX2.max())

    plt.plot()

  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  os.makedirs(plot_dir, exist_ok=True)
  plot_path = os.path.join(plot_dir, filename)
  plt.savefig(plot_path)