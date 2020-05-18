from tkinter import *
from PIL import Image
import shelve
import numpy as np
import os
import random
HEIGHT = 20
WIDHT = 20
NUM_LABELS = 10
INPUT_LAYER_SIZE = 400
HIDDEN_LAYER_SIZE = 25
class MLDraw(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.pack()
        self.canvas = Canvas(self, bg='white', height=115, width=115)
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', lambda event: self.freedraw(event))
        self.canvas.bind('<Button-1>', self.freedraw)
        self.thick = 2
        self.count = len(dbase.keys())
        self.aim = random.randint(0, 9)
        self.aimlabel = Label(self, text=self.aim)
        self.aimlabel.pack(side=BOTTOM)
        self.makebuttons()

    def freedraw(self, event):
        #print('x= ', event.x, 'y= ', event.y)
        self.canvas.create_oval(event.x - self.thick, event.y - self.thick, event.x + self.thick,
                                event.y + self.thick, fill='black', width=0)

    def makebuttons(self):
        butframe = Frame(self)
        butframe.pack(side=TOP, expand=YES, fill=X)
        Button(butframe, text='Read', command=self.save).pack(side=LEFT)
        Button(butframe, text='Clear', command=self.clear).pack(side=RIGHT)

    def save(self):
        print(self.count)
        self.canvas.postscript(file='temp.jpg')
        im = Image.open('temp.jpg')
        rez_im = im.resize((HEIGHT,WIDHT))
        rez_im.save('temp1.jpg')
        self.data = np.asarray(rez_im)
        self.Y = np.zeros((10,1))
        self.Y[self.aim] = 1
        self.X = np.zeros((HEIGHT, WIDHT))
        for y in range(HEIGHT):
            for x in range(WIDHT):
                self.X[x][y] = 1 if max(self.data[x][y]) == 255 else 0
        self.X = self.X.reshape((INPUT_LAYER_SIZE, 1))
        dbase[str(self.count)] = (self.X, self.Y)
        self.count += 1
        self.clear()



    def clear(self):
        self.canvas.delete(ALL)
        self.aim = random.randint(0, 9)
        self.aimlabel.config(text=self.aim)



if __name__ == '__main__':
    dbase = shelve.open('Test_set.db')
    root = Tk()
    MLDraw(root).pack()
    root.mainloop()
    dbase.close()