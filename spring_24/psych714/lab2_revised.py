#!/usr/bin/env python
"""
Lab 2: Visual Search Program
Independent variables that can be changed:
    @param number_of_objects: Sets the number of objects displayed
    @param dependentText: Sets the text of the dependent variable
    @param time_per_slide: Sets the time for each slide to be shown
    @param number_of_screens: Sets the number of stills to be displayed
"""

from psychopy import visual, core, event
import random

def run_experiment(condition, number_of_objects, dependentText):
    event.globalKeys.clear()
    global_event_key_yes = 'y'
    global_event_key_no = 'n'
    if_dep_present = ""
    key_pressed = ""
    found = 0
    results_file_name = f'results_{condition}.txt'

    def returnRandomPositions():
        posX = random.randint(-800, 800)
        posY = random.randint(-800, 800)
        return [posX / 1000, posY / 1000]

    def onEventKeyPressed(key):
        nonlocal found, key_pressed
        found = 1
        key_pressed = key

    event.globalKeys.add(key=global_event_key_yes, func=lambda: onEventKeyPressed('y'))
    event.globalKeys.add(key=global_event_key_no, func=lambda: onEventKeyPressed('n'))
    event.globalKeys.add(key='q', modifiers=['ctrl'], func=core.quit)

    window = visual.Window(units="norm", fullscr=False, monitor="myMonitor")
    clock = core.Clock()

    with open(results_file_name, "a") as results_file:
        for _ in range(10):  # number_of_screens
            found = 0
            startingTime = clock.getTime()

            # Draw objects with random positions
            for _ in range(number_of_objects):
                pos = returnRandomPositions()
                visual.TextStim(window, text=dependentText if random.random() < 0.5 else 'O', pos=pos, height=0.05, color=[1, 1, 1]).draw()

            window.flip()
            while clock.getTime() - startingTime < 10:  # time_per_slide
                if found:
                    break

            reaction_time = clock.getTime() - startingTime
            results_file.write(f"{dependentText},{if_dep_present},{key_pressed},{reaction_time}\n")

            window.flip()  # Clear the window after each trial
            core.wait(1.0)  # Wait a second before the next trial to prevent immediate overlap

if __name__ == "__main__":
    conditions = {
        "ef": {"condition": "easy_few", "number_of_objects": 10, "dependentText": "X"},
        "em": {"condition": "easy_many", "number_of_objects": 20, "dependentText": "X"},
        "hf": {"condition": "hard_few", "number_of_objects": 10, "dependentText": "∅"},
        "hm": {"condition": "hard_many", "number_of_objects": 20, "dependentText": "∅"}
    }

    user_input = input("Select: \nEasy few: ef\nEasy many: em\nHard few: hf\nHard many: hm\n").lower()
    if user_input in conditions:
        run_experiment(**conditions[user_input])
    else:
        print("Invalid selection. Please try again.")


