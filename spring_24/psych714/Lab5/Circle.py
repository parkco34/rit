from psychopy import visual,core

class Circle:

    circle = None
    def __init__(self, window, position, color):

        self.circle = visual.Circle(window, radius = 0.1, pos = position, units = "norm", edges = 64, fillColor = color, lineColor = color)
        self.circle.autoDraw = True

    def get(self):
        return self.circle.pos

    def set(self,position,gain):
        if position[0]*gain>=1 or position[0]*gain<=-1 or position[1]*gain>=1 or position[1]*gain<=-1:
            return
        
        self.circle.pos = (position[0]*gain, position[1])