import os

import math
import numpy as np
import matplotlib.pyplot as plt
import bisect


def rotate(center, point, angle):
    """
    helper function to rotate point on to x axis (anticlockwise)
    center: point to rotate around (x, y)
    point: point that will be rotated (x,y)
    angle: angle that the point will be rotated around the center (in radians)
    returns: x and y coordinates of rotated point
    """
    ox, oy = center
    px, py = point

    newx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    newy = oy + math.cos(angle) * (py - oy) + math.sin(angle) * (px - ox)
    return newx, newy

pointsx = [3, 2, 1]
pointsy = [3, 2, 1]
coeffs_linear_fit = np.polyfit(pointsx, pointsy, deg=1)

zero_crossing = (coeffs_linear_fit[1] * -1)  / coeffs_linear_fit[0]
print("coeffs: ", coeffs_linear_fit)
print("zero crossing: ", zero_crossing)

# compute angle
rad = math.atan(coeffs_linear_fit[0])
print("angle: ", math.degrees(rad))

new_x = []
new_y = []

# print(data_selectionx.shape)
for x, y in zip(pointsx, pointsy):
        newx, newy = rotate(center=(zero_crossing, 0), point=(x, y), angle=rad * -1)
        print("new point for ", x, y, " is ", newx, newy)
        new_y.append(newy)
        new_x.append(newx)

plt.scatter(pointsx, pointsy, label="old")
plt.plot(pointsx, coeffs_linear_fit[0] * np.asarray(pointsx) \
            + coeffs_linear_fit[1], color='red')
plt.scatter(zero_crossing, 0, label="zero crossing")
plt.scatter(new_x, new_y, label="new")
plt.legend()
plt.show(block=True)
