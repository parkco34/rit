from psychopy import visual, core, monitors, event, gui
from Crosshair import Crosshair
from Rect import Rect
import random

ORDER = None
DISTANCE = None
GAIN = None
CAP_TIME = None
WIDTH = None
DISTUBRANCE = None
AXIS = None
CANCEL = False



def pursuit():
    global ORDER, WIDTH, CAP_TIME, DISTANCE, GAIN, CANCEL, DISTUBRANCE, AXIS
    
    dlg = gui.Dlg(title = 'Discrete Control', size = (300,300))
    dlg.addField(label = 'Width of target [0-1]', initial = '0.2' )
    dlg.addField(label = 'Amplitude [0-1.2]', initial = '1')
    dlg.addField(label = 'Gain', initial = '1')
    dlg.addField(label = 'Order', choices = ['0th Order','1st Order','2nd Order'])
    dlg.addField(label = 'Capture time', initial = '2.5')
    dlg.addField(label = 'Disturbance', choices = ['Low','Medium','High'])
    dlg.addField(label = 'Axis', choices = ['X','Y'])
    data = dlg.show()

    if data == None:
        return

    WIDTH = float(data[0])
    DISTANCE = float(data[1])
    GAIN = float(data[2])

    if data[3] == '0th Order':
        ORDER = 0
    elif data[3] == '1st Order':
        ORDER = 1
    else:
        ORDER = 2

    CAP_TIME = float(data[4])

    if data[5] == 'Low':
        DISTUBRANCE = 0
    elif data[5] == 'Medium':
        DISTUBRANCE = 1
    else:
        DISTUBRANCE = 2

    if data[6] == 'X':
        AXIS = 0
    else:
        AXIS = 1
        
    monitor = monitors.Monitor('myMon')
    window = visual.Window(allowGUI = True,fullscr=True, monitor = monitor, units = 'norm',color = (-1,-1,-1))
    rect = Rect(window,(0.8,0),float(WIDTH))
    distance = float(DISTANCE)
    gain = float(GAIN)
    order = float(ORDER)
    capture_time = float(CAP_TIME)
    disturbanceArray = [0.002,0.004,0.006]
    disturbance = int(DISTUBRANCE)
    axis = int(AXIS)
    DISTANCE_FROM_CENTRE = -(rect.get()[0] - distance)
    entered_box = False


    crosshair = Crosshair(window,rect.get()[0] - distance)

    window.flip()

    mouse = event.Mouse(visible = True, win = window)
    mouse.setPos(newPos = (rect.get()[0] - distance,0))
    
    mouse.setVisible(False)

    start_time = core.getTime()
    start_pos = crosshair.get(DISTANCE_FROM_CENTRE)
    prev_pos = mouse.getPos()
    accelerationX = 0
    accelerationY = 0
    directionX = 'NOT MOVING'
    directionY = 'NOT MOVING'
    speedX = 0
    speedY = 0
    dist = disturbanceArray[disturbance]
    randX = 6
    while not event.getKeys():
        current_time = core.getTime()

        if current_time %10 == randX:
            randX = random.randint(1,10)
        elif current_time %10 < randX:
            distX = -disturbanceArray[disturbance]
        else:
            distX = disturbanceArray[disturbance]
            
        if current_time %5 <=3:
            distY = -disturbanceArray[disturbance]
        else:
            distY = disturbanceArray[disturbance]

        if order == 0:
            
            if axis == 0:
                mouse.setPos (newPos=(mouse.getPos()[0],0))
                
                rect.set((rect.get()[0] + distX, 0))
            else:
                rect.set((rect.get()[0] + distX, rect.get()[1] + distY))

            crosshair.setWithGain(mouse.getPos(),gain,DISTANCE_FROM_CENTRE)
        
        if order == 1:
            if axis == 0:
                mouse.setPos(newPos = (mouse.getPos()[0],0))

            if start_pos[0] != mouse.getPos()[0]:
                speedX = 0.01*(mouse.getPos()[0] - start_pos[0])
            if start_pos[1] != mouse.getPos()[1]:
                speedY = 0.01*(mouse.getPos()[1] - start_pos[1])

            if axis == 0:
                rect.set((rect.get()[0] + distX, 0))
                pos = [crosshair.get(DISTANCE_FROM_CENTRE)[0] + speedX,0]
            else:
                pos = [crosshair.get(DISTANCE_FROM_CENTRE)[0] + speedX, crosshair.get(DISTANCE_FROM_CENTRE)[1] + speedY]
                rect.set((rect.get()[0] + distX, rect.get()[1] + distY))
            crosshair.setWithGain(pos,gain,DISTANCE_FROM_CENTRE)

        if order == 2:
            if axis == 0:
                mouse.setPos(newPos = (mouse.getPos()[0],0))

            if crosshair.get(DISTANCE_FROM_CENTRE)[0] >=0.999 and crosshair.get(DISTANCE_FROM_CENTRE)[0]<=-0.999 and crosshair.get(DISTANCE_FROM_CENTRE)[1] >=0.999 and crosshair.get(DISTANCE_FROM_CENTRE)[1]<=-0.999:
                mouse.setPos(newPos = crosshair.get(DISTANCE_FROM_CENTRE))
                #accelerationX = 0
                #accelerationY = 0

            if mouse.getPos()[0] > prev_pos[0]:
                directionX = 'RIGHT'
                accelerationX += 0.0002*(mouse.getPos()[0] - prev_pos[0])
            elif mouse.getPos()[0] < prev_pos[0]:
                directionX = 'LEFT'
                accelerationX -= 0.0002*(mouse.getPos()[0] - prev_pos[0])
            else:
                directionX = 'NOT MOVING'

            if directionX == 'RIGHT':
                accelerationX +=0.001
            elif directionX == 'LEFT':
                accelerationX -=0.001
            else:
                pass
                
            if mouse.getPos()[1] > prev_pos[1]:
                directionY = 'RIGHT'
                accelerationY += 0.0002*(mouse.getPos()[1] - prev_pos[1])
            elif mouse.getPos()[1] < prev_pos[1]:
                directionY = 'LEFT'
                accelerationY -= 0.0002*(mouse.getPos()[1] - prev_pos[1])
            else:
                directionY = 'NOT MOVING'

            if directionY == 'RIGHT':
                accelerationY +=0.001
            elif directionY == 'LEFT':
                accelerationY -=0.001
            else:
                pass

            prev_pos = mouse.getPos()
            if axis == 0:
                rect.set((rect.get()[0] + distX, 0))
                pos = [crosshair.get(DISTANCE_FROM_CENTRE)[0] + accelerationX,0]
            else:
                rect.set((rect.get()[0] + distX, rect.get()[1] + distY))
                pos = pos = [crosshair.get(DISTANCE_FROM_CENTRE)[0] + accelerationX, crosshair.get(DISTANCE_FROM_CENTRE)[1] + accelerationY]
            crosshair.setWithGain(pos,gain,DISTANCE_FROM_CENTRE)

        if crosshair.get(DISTANCE_FROM_CENTRE)[0]>=rect.getXBounds()[0] and crosshair.get(DISTANCE_FROM_CENTRE)[0]<=rect.getXBounds()[1] and crosshair.get(DISTANCE_FROM_CENTRE)[1]>=rect.getYBounds()[0] and crosshair.get(DISTANCE_FROM_CENTRE)[1]<=rect.getYBounds()[1]:
            if not entered_box:
                entered_box = True
                time_entered = core.getTime()
        else:
            entered_box = False
            time_entered = 0

        if entered_box:
            if (current_time - time_entered > capture_time):
                        total_time = core.getTime() - start_time - capture_time
                        f1 = open('Result.txt', 'r')
                        
                        lines = f1.readlines()
                        f1.close()
                        f2 = open('Result.txt','a')
                        print (lines)
                        if len(lines) < 1:
                            f2.write(str(0) + "," + str(total_time) + "\n")
                        else:
                            f2.write(str(int(lines[-1].split(',')[0]) + 1) + "," + str(total_time) + "\n")
                        f2.close()
                        break

        
                
        window.flip()
    window.close()


#pursuit()
