'''
    Lab 2 : Signal Detection Theory
    Independant variables that can be changed :
        @param      global_event_key        Sets the key that is to be pressed when the dependant is found
        @param      time_per_slide          Sets the time for each slide to be shown
        @param      number_of_screens       Sets the number of stills to be displayed
        @param      size                    Sets size of the blemish
        @param      blemishColor            Sets color of the blemish
        @param      probability             Probability of the blemish occuring
'''

from psychopy import visual,core,event
from psychopy.preferences import prefs
import random

#Clear the set of global event keys incase of any spurious keys
event.globalKeys.clear()


global_event_key_yes = 'y'
global_event_key_no = 'n'
hit = 0
false_alarm = 0
correct_rejection = 0
miss = 0
if_dep_present = 0
number_of_signals = 0
key_pressed = ""
is_knowledge_on = False
t_time = 0
knowledge_text = ""

weaves = ['bg1.png','bg2.png','bg3.png','bg4.png','bg5.png','bg6.png']

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
    posY = random.randint(-600,600)
    return [posX/1000,posY/1000]


'''
    Every event key must be tied to a function for action
'''

def onEventKeyPressedY():
    global found, key_pressed, if_dep_present, hit, miss
    #print ('found')
    found = 1
    key_pressed = 'y'

    if if_dep_present == 1:
        hit = 1
    else:
        miss = 1

def onEventKeyPressedN():
    global found, key_pressed, if_dep_present, correct_rejection, false_alarm
    #print ('found')
    found = 1
    key_pressed = 'n'
    if if_dep_present == 0:
        correct_rejection =  1
    else:
        false_alarm = 1


'''
    This function draws the desired number of objects and also the dependant
'''

def drawObject(window, size, probability, blemishColor):

    global weaves, if_dep_present, number_of_signals
    weave = random.randint(0,len(weaves))
    size = size*0.1

    # Choose from one of the various patterns to display and display it as an image
    img = visual.ImageStim(window, weaves[weave], size = (2,1.7))
    
    img.autoDraw = True

    # Line appears based on probability
    if(random.randint(0,100) <= probability*100):
        x,y = returnRandomPositions()
        angle = random.randint(0,360)

        # Draw the Blemish with a random position, and angle 
        blemish = visual.Line(window, start = (x,y), end = (x+0.2,y), lineColor = blemishColor, ori = angle)
        blemish.autoDraw = True
        if_dep_present = 1
        number_of_signals +=1
    else:
        if_dep_present = 0
    
    window.flip()

def turnKnowledgeOn():
    global knowledge_text, is_knowledge_on, if_dep_present, t_time
    is_knowledge_on = True
    if if_dep_present == 1:
        knowledge_text = "Present"
        t_time = t_time + 10
    else:
        knowledge_text = "Not Present"


def resetVars():
    global hit, false_alarm, correct_rejection, miss, is_knowledge_on, knowledge_text
    hit = 0
    false_alarm = 0
    correct_rejection = 0
    miss = 0
    is_knowledge_on = False
    knowledge_text = ""


# This method is used to add the key for any action, where key is the key that can be pressed
# And func is the action that is to be performed when the event occurs
event.globalKeys.add(key = global_event_key_yes, func = onEventKeyPressedY)
event.globalKeys.add(key = global_event_key_no, func = onEventKeyPressedN)
event.globalKeys.add(key = "c", func = turnKnowledgeOn)
event.globalKeys.add(key='q', modifiers=['ctrl'], func=core.quit)
found = 0


clock = core.Clock()

def main():

    t_hits = 0 
    t_fa = 0 
    t_cr = 0 
    t_miss = 0
    results_file = open('results.txt',"a")
    summary = open('summary.txt','a')
    number_of_screens = 10
    time_per_slide = 10
    blemishColor = (0.1, 0.1, 0.1)
    size = 1
    probability = 0.6

    global if_dep_present, hit, false_alarm, correct_rejection, miss, found, is_knowledge_on, t_time
    for i in range(number_of_screens):
        
        found = 0
        t_time = time_per_slide
        resetVars()
        # Set starting time when each screen is loaded
        startingTime = clock.getTime()


        #This function creates the window that the graphics are drawn on
        window = visual.Window(units="norm", monitor = "myMonitor", color=(0,0,0))

        # Draw objects with random positions
        drawObject(window, size, probability, blemishColor)

        window.flip(clearBuffer = False)

        '''
            This snippet of code is to draw the clock and show the time left for each still
        '''
        while clock.getTime() - startingTime < t_time:
                clockStim = visual.TextStim(window, text = int(clock.getTime() - startingTime), height = 0.1, pos = (-0.9,0.93), color = (1,1,1))
                clockStim.autoDraw = True
                window.flip()
                clockStim.text = ""
                
                if found == 1:
                    s = str(i) + "," + str(if_dep_present) + "," + key_pressed + "," + str(clock.getTime() - startingTime) + "\n"
                    results_file.write(s)
                    break

                if is_knowledge_on == True:
                    knowledgeStim = visual.TextStim(window, text = knowledge_text, height = 0.05, pos= (0, 0.93 ), color=(1,1,1))
                    knowledgeStim.autoDraw = True

        #print (found)
        if found == 0:
            s = str(i) + "," + str(if_dep_present) + "," + "none" + "," + str(time_per_slide) + "\n"
            results_file.write(s)
            pass

        
        t_hits += hit
        t_fa += false_alarm
        t_cr += correct_rejection
        t_miss += miss

        
        window.close()
    summary.write(str(t_hits/number_of_signals) + "," + str(t_fa/number_of_signals) + "," + str(t_cr/number_of_signals) + "," + str(t_miss/number_of_signals) )
    results_file.close()
    summary.close()


if __name__ == "__main__":
    main()