import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from data import classification_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
def gif_plotter(X_train,y_train,lines):
    # Assuming lr.y_pred is a list of regression lines for each frame
    # Adjust this based on your actual data
    y_pred_list = lines

    # Set up the figure and axis
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_train, y_train)
    line, = ax.plot([], [], color='red')

    # Function to update the plot in each animation frame
    def update(frame):
        line.set_data(X_train, y_pred_list[frame])
        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(y_pred_list), blit=True)
    # Save the animation as a GIF using PillowWriter
    ani.save('regression_animation.gif', writer=PillowWriter(fps=10))

    plt.show()
