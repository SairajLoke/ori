import numpy as np

z = np.load("episode_0_queries_50.npz")
pred = z["predicted_actions"]

boundary_diff = np.abs(
    pred[:-1, -1, :] - pred[1:, 0, :]
)

# Largest individual jump
boundary, joint = np.unravel_index(
    np.argmax(boundary_diff),
    boundary_diff.shape
)

print("Largest jump:")
print("boundary:", boundary, "->", boundary + 1)
print("joint:", joint)
print("difference:", boundary_diff[boundary, joint])

print("\nPrevious horizon final:")
print(pred[boundary, -1, joint])

print("Next horizon first:")
print(pred[boundary + 1, 0, joint])