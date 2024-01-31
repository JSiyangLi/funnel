import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def plot(df):
    # transform BF from natural log to log10
    scaling = np.log10(np.exp(1))
    df['BF'] = df['BF'].apply(lambda x: x * scaling)
    df['fp'] = df['fp'].apply(lambda x: np.log10(x))

    # colours for dots
    Om9 = (1, 0, 0)
    Om10 = (0, 1, 0)
    Om11 = (0, 0, 1)

    # create a new variable to classify and colour the dots
    def Om_class(OmegaP):
        if OmegaP == 9:
            return Om9
        elif OmegaP == 10:
            return Om10
        else:
            return Om11

    df['Omn'] = df['OmegaP'].apply(Om_class)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Bayes factor for models with phase transition vs. without phase transition')

    # overall plot
    ax1.scatter(df['fp'], df['BF'], c = df['Omn'], alpha = 0.9)
    ax1.legend([r"$Omega_p=10^{-9}$", r"$Omega_p=10^{-10}$", r"$Omega_p=10^{-11}$"], loc="upper left")
    rect = Rectangle(xy = (-5, -2), width = 5, height = 4, alpha = 0.4, color="tab:green")
    ax1.gca().add_patch(rect)
    ax1.set_xticks([-5, -4, -3, -2, -1, 0])  # Set the tick positions
    ax1.set_xlabel(r"$\log(f_p)$", labelpad=5)  # Set the tick labels
    ax1.set_ylabel("log Bayes factor", labelpad=5)

    # detailed plot near 0
    ax2.scatter(df['fp'], df['BF'], c=df['Omn'], alpha=0.9)
    ax2.ylim(-15, 15)
    [ax2.axhline(i, color='k', linestyle='--') for i in [-2, 2]]
    ax2.axhline(0, color='k')

    #
    plt.show()

data = pd.read_csv("LISA_1pse.csv").transpose()
plot(data)