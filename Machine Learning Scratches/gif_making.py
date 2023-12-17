import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Generate some example data
np.random.seed(42)
X_train = np.random.rand(50, 1)
y_train = 2 * X_train.squeeze() + 1 + 0.1 * np.random.randn(50)
print(X_train.shape,y_train.shape)
# Assume lr.y_pred is a list of predicted lines during training
# For simplicity, let's generate some random lines
num_lines = 10
y_pred = [2 * X_train.squeeze() + np.random.normal(0, 0.5, size=X_train.shape[0]) for _ in range(num_lines)]
print(len(y_pred))
fig, ax = plt.subplots()
scatter = ax.scatter(X_train, y_train)
line, = ax.plot([], [], label='Prediction')
ax.set_title('Linear Regression Training Animation')
ax.set_xlabel('X train')
ax.set_ylabel('Y train')
ax.legend()

def update(frame):
    line.set_data(X_train, y_pred[frame])
    return scatter, line

num_frames = len(y_pred)
ani = FuncAnimation(fig, update, frames=range(num_frames), blit=True)
# Save the animation as a GIF
ani.save('linear_regression_training.gif', writer='imagemagick', fps=2)  # You can try 'pillow' if 'imagemagick' doesn't work
plt.show()
