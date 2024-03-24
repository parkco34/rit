from psychopy import visual,core

class Crosshair:

    line1 = None
    line2 = None
    def __init__(self,window,pos):
        self.line1 = visual.Line(window,units = 'norm', start = (pos-0.03,0), end = (pos+0.03,0), lineColor = 'green')
        self.line1.autoDraw = True
        self.line2 = visual.Line(window, units = 'norm', start = (pos,-0.03),end = (pos,0.03), lineColor = 'green')
        self.line2.autoDraw = True

    def set(self, pos, DISTANCE_FROM_CENTRE):
        if pos[0]>=-1 and pos[0]<=1 and pos[1]>=-1 and pos[1]<=1:
            self.line1.pos = (pos[0] + DISTANCE_FROM_CENTRE,pos[1])
            self.line2.pos = (pos[0] + DISTANCE_FROM_CENTRE,pos[1])

    def get(self,DISTANCE_FROM_CENTRE):
        return (self.line1.pos[0] - DISTANCE_FROM_CENTRE, self.line1.pos[1])

    def setWithGain(self, pos,gain, DISTANCE_FROM_CENTRE):
        if pos[0]*gain>=-1 and pos[0]*gain<=1 and pos[1]*gain>=-1 and pos[1]*gain<=1:
            self.line1.pos = ((pos[0])*gain+DISTANCE_FROM_CENTRE, pos[1]*gain)
            self.line2.pos = ((pos[0])*gain+DISTANCE_FROM_CENTRE, pos[1]*gain)
