import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plotUnitData(data, title, xlabel, ylabel,color):
    plt.plot(data,color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
def removeOutliers(data,threshold):
    differences = np.abs(np.diff(data, prepend=data[0]))

    # Detect outliers where the jump is greater than the threshold
    outliers = differences > threshold

    # Replace outliers with interpolated values (average of neighbors)
    cleaned_data = data.copy()
    for i in range(1, len(data) - 1):
        if outliers[i]:
            cleaned_data[i] = (cleaned_data[i - 1] + cleaned_data[i + 1]) / 2
    return cleaned_data
def smoothData(data,sigma):
    smoothed_data = gaussian_filter1d(data, sigma=sigma)
    return smoothed_data