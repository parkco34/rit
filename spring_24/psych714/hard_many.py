#!/usr/bin/env python
'''
    Lab 2 : Visual Search Program
    Independant variables that can be changed :
        @param      number_of_objects       Sets the number of objects that are displayed
        @param      independantText         Sets the text of the independant objects
        @param      dependantText           Sets the text of the dependant variable
        @param      global_event_key        Sets the key that is to be pressed when the dependant is found
        @param      time_per_slide          Sets the time for each slide to be shown
        @param      number_of_screens       Sets the number of stills to be displayed
        @param      size                    Sets size of the independant text
        @color      color                   Sets color of the independant
'''

from psychopy import visual,core,event
from psychopy.preferences import prefs
import random

#Clear the set of global event keys incase of any spurious keys
event.globalKeys.clear()


global_event_key_yes = 'y'
global_event_key_no = 'n'
if_dep_present = ""
key_pressed = ""

'''
    This function calculates random positions for X and Y coordinates
    @return     posX        X Coordinate
    @return     posY        Y Coordinate
'''

def returnRandomPositions():
    #posX = random.randint(200,1800)
    #posY = random.randint(-400,400)

    #return [posX/1000,posY/1000]

    posX = random.randint(-800,800)
    posY = random.randint(-800,800)
    return [posX/1000,posY/1000]


'''
    Every event key must be tied to a function for action
'''

def onEventKeyPressedY():
    global found, key_pressed
    #print ('found')
    found = 1
    key_pressed = 'y'

def onEventKeyPressedN():
    global found, key_pressed
    #print ('found')
    found = 1
    key_pressed = 'n'


'''
    This function draws the desired number of objects and also the dependant
'''

def drawObject(window, number_of_objects = 10, dependantText = 'O',
               independantText = 'X', dependantSize = 1, dependantColor = [0,0,0], independantSize = 1, independantColor = [0,0,1]):

    global if_dep_present
    independantSize = independantSize * 0.05
    dependantSize = dependantSize * 0.05
    drawnStims = []
    pos = []
    for _ in range(number_of_objects):

        if len(drawnStims) == 0:
            pos = returnRandomPositions()
        else:
            flag = 0

            while(True):
                pos = returnRandomPositions()
                for stims in drawnStims:
                    if not stims.contains(pos):
                        flag = 1
                    else:
                        flag = 0

                if flag == 1:
                    break

        # visual.TextStim is used to create the Text that is seen on the screen
        # The window itself is passed, then the text, and then the position the text is to be displayed
        independantTextStim = visual.TextStim(window, text = independantText,pos=pos,
                                    height = dependantSize, color = dependantColor)


        # When autoDraw is set to true, the object can be drawn on ever frame
        # If you want to choose when a stimuli is diplayed, use the draw() function instead
        independantTextStim.autoDraw = True


    if(random.randint(0,10) <5):
        dependantTextStim = visual.TextStim(window,units = 'norm', text = dependantText, pos=returnRandomPositions(), height = independantSize, color = independantColor)
        dependantTextStim.autoDraw = True
        if_dep_present = 'y'

    else:
        if_dep_present = 'n'
    window.flip()


# This method is used to add the key for any action, where key is the key that can be pressed
# And func is the action that is to be performed when the event occurs
event.globalKeys.add(key = global_event_key_yes, func = onEventKeyPressedY)
event.globalKeys.add(key = global_event_key_no, func = onEventKeyPressedN)
event.globalKeys.add(key='q', modifiers=['ctrl'], func=core.quit)
found = 0


clock = core.Clock()

def main():

    results_file = open('results_hard2.txt',"a")
    number_of_objects = 20
    dependantText = '∅'
    independantText = 'O'
    dependantSize = 1
    dependantColor = [0.35,0.35,0.35]
    independantSize = 1
    independantColor = [1,1,1]
    number_of_screens = 10
    time_per_slide = 10
    global if_dep_present

    for i in range(number_of_screens):
        global found
        found = 0
        # Set starting time when each screen is loaded
        startingTime = clock.getTime()


        #This function creates the window that the graphics are drawn onL
        window = visual.Window(units="norm", pos=[0, 0], fullscr=False, monitor="myMonitor")

        # Draw objects with random positions
        drawObject(window, number_of_objects, dependantText, independantText, dependantSize, dependantColor, independantSize, independantColor)

        window.flip(clearBuffer = False)

        '''
            This snippet of code is to draw the clock and show the time left for each still
        '''
        while clock.getTime() - startingTime < time_per_slide:
                clockStim = visual.TextStim(window, text = int(clock.getTime() - startingTime), height = 0.05, pos = (-0.9,0.9))
                clockStim.autoDraw = True
                window.flip()
                clockStim.text = ""

                if found == 1:
                    s = dependantText + "," + if_dep_present + "," + key_pressed + "," + str(clock.getTime() - startingTime) + "\n"
                    results_file.write(s)
                    break

        #print (found)
        if found == 0:
            s = dependantText + "," + if_dep_present + "," + "none" + "," + str(10) + "\n"
            results_file.write(s)
        window.close()

    results_file.close()


if __name__ == "__main__":
    main()


