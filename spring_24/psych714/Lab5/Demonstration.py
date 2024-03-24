from psychopy import visual, core, event, monitors, gui
from Circle import Circle

GAIN = 0
CANCEL = False


def demonstration():

    global GAIN, CANCEL
    
    dlg = gui.Dlg(title = 'Continuous Control Demonstration', size = (300,300))
    dlg.addField(label = 'Gain', initial = '1')

    data = dlg.show()
    if data == None:
        return

    GAIN = float(data[0])
    monitor = monitors.Monitor('myMon')
    window = visual.Window(allowGUI = True,fullscr=True, monitor = monitor, units = 'norm',color = (-1,-1,-1))
    GAIN = float(GAIN)
    line = visual.Line(window,start = (0,1), end = (0,-1))
    line.autoDraw = True
    circle1 = Circle(window,(0,0.6),"red")
    circle2 = Circle(window,(0,0),"green")
    circle3 = Circle(window,(0,-0.6),"blue")
    
    

    mouse = event.Mouse()
    mouse.setPos(newPos = (0,0))
    start_pos = 0
    prev_pos = 0
    speed = 0
    acceleration = 0
    direction = 'NOT MOVING'
    while not event.getKeys():
        
        
        mouse.setPos(newPos= (mouse.getPos()[0],0))
        if mouse.getPressed()[0] == True:
            circle1.set((0,circle1.get()[1]),1)
            circle2.set((0,circle2.get()[1]),1)
            circle3.set((0,circle3.get()[1]),1)
            mouse.setPos( newPos = (0,0))
            start_pos = 0
            prev_pos = 0
            speed = 0
            acceleration = 0
            direction = 'NOT MOVING'
        circle1.set((mouse.getPos()[0],circle1.get()[1]), GAIN)
        
        

        if mouse.getPos()[0] != start_pos:
            speed = 0.1*(mouse.getPos()[0] - start_pos)
        
        circle2.set((circle2.get()[0]+speed, circle2.get()[1]),GAIN)
        

        if circle3.get()[0] >=1 and circle3.get()[0]<=-1:
                mouse.setPos(newPos = (prev_pos,0))
                acceleration = 0

        if mouse.getPos()[0] > prev_pos:
            direction = 'RIGHT'
            acceleration += 0.0002*(mouse.getPos()[0] - prev_pos)
        elif mouse.getPos()[0] < prev_pos:
            direction = 'LEFT'
            acceleration -= 0.0002*(mouse.getPos()[0] - prev_pos)
        else:
            direction = 'NOT MOVING'

        if direction == 'RIGHT':
            acceleration +=0.001
        elif direction == 'LEFT':
            acceleration -=0.001
        else:
            pass

        prev_pos = mouse.getPos()[0]
        pos = [circle3.get()[0] + acceleration,circle3.get()[1]]
        circle3.set(pos,GAIN)
        window.flip()

    window.close()

#demonstration()