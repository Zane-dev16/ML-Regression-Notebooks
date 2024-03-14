import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

xs = np.linspace(0, 3, 100).reshape(-1 , 1)

def plot_model(X, y,model_class, polynomial, alphas, **model_kwargs):
    for alpha, style, in zip(alphas, ("b:", "g--", "r-")):
        if alpha > 0:
            model = model_class(alpha, **model_kwargs)
        else:
            model = LinearRegression()
        if polynomial:
            model = make_pipeline(
                PolynomialFeatures(degree=10, include_bias=False),
                StandardScaler(),
                model,
            )
        model.fit(X, y)
        ys = model.predict(xs)
        plt.plot(xs, ys, style, linewidth=2, label=f"alpha = {alpha}")
        plt.axis([0, 3, 0, 3.5])
    plt.legend(loc='upper left')
    