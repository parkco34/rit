#!/usr/bin/env python
import os
# Set the environment variable for Pyglet shadow window
from psychopy import visual, core, event
import random
from psychopy.monitors import Monitor
my_monitor = Monitor(name='myMonitor', width=53.0, distance=60.0)  # Adjust these values as necessary
my_monitor.setSizePix((1920, 1080))  # Adjust resolution as necessary

# Replaces the function below this one
def returnRandomPositions():
    """
    Returns random coordinates on window of the signal.
    """
    posX = random.uniform(-1, 1)
    posY = random.uniform(-1, 1)
    return [posX, posY]

# Functions to handle key press events
def onKeyPressed(key, if_dep_present, results):
    if if_dep_present == (key == 'y'):
        results['hits' if key == 'y' else 'correct_rejections'] += 1
    else:
        results['false_alarms' if key == 'y' else 'misses'] += 1
    return True

# Replaces the function below this one
def drawObject(window, size, probability, blemishColor, weaves):
    if_dep_present = random.random() < probability
    weave = random.choice(weaves)
    img = visual.ImageStim(window, image=weave, units='pix')
    img.size = window.size
    img.draw()

    if if_dep_present:
        x, y = returnRandomPositions()
        angle = random.randint(0, 360)
        blemish = visual.Line(window, start=(x, y), end=(x+0.2, y), lineColor=blemishColor, ori=angle, units='norm')
        blemish.draw()

    return if_dep_present

# Main experiment function
def run_experiment(num_trials, time_per_slide, size, blemishColor, probability):
    results = {'hits': 0, 'misses': 0, 'false_alarms': 0, 'correct_rejections': 0}
    weaves = ['bg1.png', 'bg2.png', 'bg3.png', 'bg4.png', 'bg5.png', 'bg6.png']

    for i in range(num_trials):
        window = visual.Window(fullscr=True, color=(0, 0, 0))
        if_dep_present = drawObject(window, size, probability, blemishColor, weaves)
        window.flip()

        # Wait for a response or time out
        start_time = core.getTime()
        responded = False
        while core.getTime() - start_time < time_per_slide and not responded:
            for key in event.getKeys():
                if key in ['y', 'n']:
                    responded = onKeyPressed(key, if_dep_present, results)
                    break

        window.close()

    return results

# Parameters for the experiment
num_trials = 10
time_per_slide = 10 # seconds
size = 1 # relative size of the blemish
blemishColor = 'black' # color of the blemish
probability = 0.6 # probability of blemish occurring

# Run the experiment
experiment_results = run_experiment(num_trials, time_per_slide, size, blemishColor, probability)

# Output the results
print(experiment_results)
