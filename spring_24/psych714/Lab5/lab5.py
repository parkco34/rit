#!/usr/bin/env python
from psychopy import visual, core, event, monitors
import numpy as np

# Function to run a single trial of the compensatory tracking task
def run_trial(sr_compatibility='high'):
    # Create a window
    win = visual.Window([800, 600], pos=[300,0], fullscr=False, units="pix")

    # Initialize target and cursor
    target = visual.Circle(win, radius=10, fillColor='red', lineColor='red')
    cursor = visual.Circle(win, radius=10, fillColor='green', lineColor='green')
    target.pos = [0, 0]  # Start target at center

    # Variables for target movement
    speed = 2  # Pixels per frame
    direction = [speed, 0]  # Move horizontally initially

    # Variables for tracking accuracy
    distances = []

    # Clock to control the frame rate
    clock = core.Clock()

    # Run until key press
    while not event.getKeys():
        # Update target position
        target.pos += direction

        # Get current mouse position
        mouse_pos = event.Mouse(win=win).getPos()

        # Invert mouse X position for low S-R compatibility
        if sr_compatibility == 'low':
            mouse_pos[0] = -mouse_pos[0]

        cursor.pos = mouse_pos

        # Calculate and record the distance between cursor and target
        distance = np.linalg.norm(np.array(cursor.pos) - np.array(target.pos))
        distances.append(distance)

        # Draw target and cursor
        target.draw()
        cursor.draw()

        # Flip the display
        win.flip()

        # Bounce target off the edges
        if abs(target.pos[0]) >= win.size[0] / 2 or abs(target.pos[1]) >= win.size[1] / 2:
            direction = -np.array(direction)

    # Close the window
    win.close()

    # Calculate average deviation
    avg_deviation = np.mean(distances)
    print(f"Average deviation for {sr_compatibility} S-R compatibility: {avg_deviation:.2f} pixels")

# Main experiment
if __name__ == "__main__":
    # Run trials for both high and low S-R compatibility
    run_trial('high')
    core.wait(2)  # Wait for 2 seconds between trials
    run_trial('low')


