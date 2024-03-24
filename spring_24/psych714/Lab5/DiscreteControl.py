from psychopy import visual, core, monitors, event, gui
from Crosshair import Crosshair
from Rect import Rect

ORDER = None
DISTANCE = None
GAIN = None
CAP_TIME = None
WIDTH = None
CANCEL = False



def discreteControl():
    global ORDER, WIDTH, CAP_TIME, DISTANCE, GAIN, CANCEL
    
    
    
    dlg = gui.Dlg(title = 'Discrete Control', size = (300,300))
    dlg.addField(label = 'Width of target [0-1]', initial = '0.2' )
    dlg.addField(label = 'Amplitude [0-1.2]', initial = '1')
    dlg.addField(label = 'Gain', initial = '1')
    dlg.addField(label = 'Order', choices = ['0th Order','1st Order','2nd Order'])
    dlg.addField(label = 'Capture time', initial = '2.5')
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


        
    
    if CANCEL == True:
        return
    monitor = monitors.Monitor('myMon')
    window = visual.Window(allowGUI = True,fullscr=True, monitor = monitor, units = 'norm',color = (-1,-1,-1))
    rect = Rect(window,(0.8,0),float(WIDTH))
    distance = float(DISTANCE)
    gain = float(GAIN)
    order = float(ORDER)
    capture_time = float(CAP_TIME)
    
    DISTANCE_FROM_CENTRE = -(rect.get()[0] - distance)
    entered_box = False


    crosshair = Crosshair(window,rect.get()[0] - distance)

    window.flip()

    mouse = event.Mouse(visible = True, win = window)
    mouse.setPos(newPos = (rect.get()[0] - distance,0))
    
    mouse.setVisible(False)

    start_time = core.getTime()
    start_pos = crosshair.get(DISTANCE_FROM_CENTRE)
    prev_pos = crosshair.get(DISTANCE_FROM_CENTRE)[0]
    acceleration = 0
    direction = 'NOT MOVING'

    while not event.getKeys():
        current_time = core.getTime()
        if order == 0:
            mouse.setPos (newPos=(mouse.getPos()[0],0))

            crosshair.setWithGain(mouse.getPos(),gain,DISTANCE_FROM_CENTRE)
        
        if order == 1:
            mouse.setPos(newPos = (mouse.getPos()[0],0))

            if start_pos[0] != mouse.getPos()[0]:
                speed = 0.01*(mouse.getPos()[0] - start_pos[0])

            pos = [crosshair.get(DISTANCE_FROM_CENTRE)[0] + speed,0]
            crosshair.setWithGain(pos,gain,DISTANCE_FROM_CENTRE)

        if order == 2:

            if crosshair.get(DISTANCE_FROM_CENTRE)[0] >=1 and crosshair.get(DISTANCE_FROM_CENTRE)[0]<=-1:
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
            pos = [crosshair.get(DISTANCE_FROM_CENTRE)[0] + acceleration,0]
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


#discreteControl()
