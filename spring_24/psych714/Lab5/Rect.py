from psychopy import visual,core

class Rect:

    rect = None
    def __init__(self,window,pos,width):
        self.rect = visual.Rect(window,pos = pos,width = width, lineColor = 'red')
        self.rect.autoDraw = True

    def get(self):
        return self.rect.pos

    def set(self,pos):
        self.rect.pos = pos
    def getWidth(self):
        return self.rect.width

    def getHeight(self):
        return self.rect.height

    def getXBounds(self):
        pos = self.get()[0]

        return (pos - self.getWidth()/2,pos + self.getWidth()/2)

    def getYBounds(self):
        pos = self.get()[1]

        return (pos - self.getHeight()/2,pos + self.getHeight()/2)