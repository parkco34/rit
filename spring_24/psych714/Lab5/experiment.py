#!/usr/bin/env python
from textwrap import dedent
from psychopy import visual, core, event, monitors
import numpy as np

def run_experiment(sr_compatibility="high"):
    """
    Description:
    """
    win = visual.Window([800, 600], pos=[300, 0], fullscr=False, units="pix")
    target = visual.Circle(win, radius=10, fillColor="red", lineColor="red")
    cursor = visual.Circle(win, radius=10, fillColor="green", lineColor="green")

    # Variables representing target movemnet
    speed = 2 # (px per frame)
    direction = [speed, 0] # (initially moving horizontally)

    # Distance tracking list
    distances = []
    # Temporal componnet
    clock = core.Clock()

    # Run until keypress
    while not event.getKeys():
        # Update target position
        target.pos += direction
        # Get current mouse position
        mouse_pos = event.Mouse(win=win).getPos()
        # INverts the mouse's X-posittion
        if sr_compatibility == "low":
            mouse_pos[0] = -mouse_pos[0] # numpy array

        cursor.pos = mouse_pos

        distance = np.linalg.norm(np.array(cursor.pos) - np.array(target.pos))
        distances.append(distance)

        # Lookout Picasso...
        target.draw()
        cursor.draw()
    
        # Synchronizes the visual display with experimental control logic
        win.flip()

        # Collision with edges
        if abs(target.pos[0])  >= win.size[0] / 2 or abs(target.pos[1]) >= \
        win.size[1] / 2:
            direction = -np.array(direction)

    win.close()

    # Get AVG_DEV
    avg_dev = np.mean(distance)
    print(
f"""Average deviation for {sr_compatibility} S-R compatibility:
{avg_dev:.2f} pixels"""
    )

if __name__ == "__main__":
    run_experiment("high")
    core.wait(2)
    run_experiment("low")








