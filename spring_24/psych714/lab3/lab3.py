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

# Function to draw the blemish based on probability
def drawObject(window, size, probability, blemishColor, weaves):
    if_dep_present = random.random() < probability
    weave = random.choice(weaves)
    img = visual.ImageStim(window, image=weave, units='pix')
    img.size = window.size
    img.draw()
    if if_dep_present:
        x, y = returnRandomPositions()

        # Draw the upside-down smiley face
        face_size = size * 0.1
        face_color = blemishColor
        face_opacity = 0.2  # Adjust the opacity to make it more difficult to see

        # Draw the face outline
        face = visual.Circle(window, radius=face_size, edges=32, lineColor=None, fillColor=face_color, pos=(x, y), opacity=face_opacity)
        face.draw()

        # Draw the eyes
        eye_size = face_size * 0.2
        eye_offset = face_size * 0.4
        left_eye = visual.Circle(window, radius=eye_size, edges=32, lineColor=None, fillColor='white', pos=(x - eye_offset, y + eye_offset), opacity=face_opacity)
        right_eye = visual.Circle(window, radius=eye_size, edges=32, lineColor=None, fillColor='white', pos=(x + eye_offset, y + eye_offset), opacity=face_opacity)
        left_eye.draw()
        right_eye.draw()

        # Draw the mouth using visual.ShapeStim
        mouth_width = face_size * 0.6
        mouth_height = face_size * 0.2
        mouth_vertices = [
            (x - mouth_width/2, y - face_size*0.3),
            (x + mouth_width/2, y - face_size*0.3),
            (x, y - face_size*0.3 - mouth_height)
        ]
        mouth = visual.ShapeStim(window, vertices=mouth_vertices, lineColor=None, fillColor='white', opacity=face_opacity)
        mouth.draw()

    return if_dep_present

# Main experiment function
def run_experiment(num_trials_per_condition, time_per_slide, size, blemishColor, probability, results_file, condition_name):
    weaves = ['pngs/img1.png', 'pngs/img2.png', 'pngs/img3.png', 'pngs/img4.png', 'pngs/img5.png', 'pngs/img6.png']
    window = visual.Window(fullscr=True, color=(0, 0, 0))
    feedback_text = visual.TextStim(window, text="", pos=(0, -0.8), color='white')

    num_signal_present = 0
    num_signal_absent = 0
    trial_data = []
    results = {'hits': 0, 'misses': 0, 'false_alarms': 0, 'correct_rejections': 0}

    while num_signal_present < num_trials_per_condition // 2 or num_signal_absent < num_trials_per_condition // 2:
        if_dep_present = drawObject(window, size, probability, blemishColor, weaves)
        window.flip()

        # Wait for a response or time out
        start_time = core.getTime()
        responded = False
        while core.getTime() - start_time < time_per_slide and not responded:
            for key in event.getKeys():
                if key in ['y', 'n']:
                    responded = True
                    feedback_text.text = "You pressed: " + key
                    feedback_text.draw()
                    window.flip()
                    core.wait(0.5)  # Display feedback for 0.5 seconds
                    break

        if if_dep_present:
            num_signal_present += 1
            if responded:
                if key == 'y':
                    results['hits'] += 1
                else:
                    results['misses'] += 1
            else:
                results['misses'] += 1
        else:
            num_signal_absent += 1
            if responded:
                if key == 'y':
                    results['false_alarms'] += 1
                else:
                    results['correct_rejections'] += 1
            else:
                results['correct_rejections'] += 1

        trial_data.append(f"{if_dep_present},{'y' if responded and key == 'y' else 'n' if responded else 'none'},{core.getTime() - start_time}\n")

    # Write the trial data to the results file
    with open(results_file, "a") as file:
        file.write(f"Condition: {condition_name}\n")
        for i, data in enumerate(trial_data[:num_trials_per_condition]):
            file.write(f"{i},{data}")

    return window, results

# Function to calculate hits, false alarms, misses, and correct rejections from results file
def calculate_results(file_path, summary_file):
    results = {'hits': 0, 'misses': 0, 'false_alarms': 0, 'correct_rejections': 0}
    current_condition = None

    with open(file_path, 'r') as file, open(summary_file, 'w') as summary:  # Open summary file in write mode
        for line in file:
            if line.startswith("Condition:"):
                if current_condition is not None:
                    summary.write(f"Condition: {current_condition}\n")
                    summary.write(f"Hits: {results['hits']}, False Alarms: {results['false_alarms']}, Misses: {results['misses']}, Correct Rejections: {results['correct_rejections']}\n\n")
                    results = {'hits': 0, 'misses': 0, 'false_alarms': 0, 'correct_rejections': 0}  # Reset counts for each condition
                current_condition = line.strip().split(": ")[1]
            else:
                trial, if_dep_present, response, _ = line.strip().split(',')
                if_dep_present = True if if_dep_present == 'True' else False
                if if_dep_present and response == 'y':
                    results['hits'] += 1
                elif if_dep_present and response == 'n':
                    results['misses'] += 1
                elif not if_dep_present and response == 'y':
                    results['false_alarms'] += 1
                elif not if_dep_present and response == 'n':
                    results['correct_rejections'] += 1

        # Write the results for the last condition
        if current_condition is not None:
            summary.write(f"Condition: {current_condition}\n")
            summary.write(f"Hits: {results['hits']}, False Alarms: {results['false_alarms']}, Misses: {results['misses']}, Correct Rejections: {results['correct_rejections']}\n\n")

    return results

# Parameters for the experiment
num_trials_per_condition = 10  # Specify the desired number of trials per condition
size = 0.5  # relative size of the blemish (decreased from 1)
blemishColor = 'black'  # color of the blemish
probability = 0.3  # probability of blemish occurring (decreased from 0.5)

# Conditions for stimulus duration
conditions = [
    {'name': 'Short', 'time_per_slide': 1},  # decreased from 2
    {'name': 'Medium', 'time_per_slide': 1.75},  # decreased from 4
    {'name': 'Long', 'time_per_slide': 2}
]

# Clear the results file before starting the experiment
open('resultz.txt', 'w').close()
open('summary.txt', 'w').close()

# Run the experiment for each condition
for condition in conditions:
    print(f"Running condition: {condition['name']}")
    window, experiment_results = run_experiment(num_trials_per_condition, condition['time_per_slide'], size, blemishColor, probability, 'resultz.txt', condition['name'])
    window.close()
    print(f"Results for {condition['name']} condition: {experiment_results}")

# Calculate and print the final results from the results file
final_results = calculate_results('resultz.txt', 'summary.txt')
print(f"Final results: {final_results}")

