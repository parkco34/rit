#!/usr/bin/env python
#!/usr/bin/env python
import os
from psychopy import visual, core, event
import random

# Function to calculate random positions for the blemish
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

# Function to draw the blemish based on probability
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

    window = visual.Window(fullscr=True, color=(0, 0, 0))

    results_file = open('results.txt', "a")
    summary = open('summary.txt', 'a')

    for i in range(num_trials):
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

        s = f"{i},{if_dep_present},{'y' if responded and key == 'y' else 'n' if responded else 'none'},{core.getTime() - start_time}\n"
        results_file.write(s)

    window.close()

    t_hits = results['hits']
    t_fa = results['false_alarms']
    t_cr = results['correct_rejections']
    t_miss = results['misses']
    number_of_signals = t_hits + t_miss

    summary.write(f"{t_hits/number_of_signals if number_of_signals > 0 else 0},{t_fa/number_of_signals if number_of_signals > 0 else 0},{t_cr/number_of_signals if number_of_signals > 0 else 0},{t_miss/number_of_signals if number_of_signals > 0 else 0}")

    results_file.close()
    summary.close()

    return results

# Parameters for the experiment
num_trials = 10
time_per_slide = 10  # seconds
size = 1  # relative size of the blemish
blemishColor = 'black'  # color of the blemish
probability = 0.6  # probability of blemish occurring

# Run the experiment
experiment_results = run_experiment(num_trials, time_per_slide, size, blemishColor, probability)

# Output the results
print(experiment_results)
