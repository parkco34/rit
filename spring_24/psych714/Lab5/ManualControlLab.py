import psychopy
from psychopy import gui

from DiscreteControl import discreteControl
from Demonstration import demonstration
from Compensatory import compensatory
from Pursuit import pursuit

CHOICE = 0

        

def MCL():

    global CHOICE
    
    while True:

        dlg = gui.Dlg(title = 'Choose Experiment')
        dlg.addField(label = 'Experiment', choices = ['Discrete Control', 'Continuous Control Demonstration', 'Compensatory Tracking', 'Pursuit Tracking'])    
        data = dlg.show()

        if data == None:
            exit()

        if data[0] == 'Discrete Control':
            discreteControl()

        elif data[0] == 'Continuous Control Demonstration':
            demonstration()

        elif data[0] == 'Compensatory Tracking':
            compensatory()

        else:
            pursuit()

if __name__ == "__main__":
    MCL()



