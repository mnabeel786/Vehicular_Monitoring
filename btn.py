from tkinter import *
import os
import sys

window=Tk()

window.title("Vehicular Monitoring System")
window.geometry('550x200')

def run():
    os.system('python main.py')

B = Button(window,height = 2, text ="Click Me!For License Plate",command=run)
def run1():
	os.system('python helmet.py')

b = Button(window,height = 2, text = "Click Me! For Helmet Detection",command=run1)

B.pack(pady=20)
b.pack(pady=20)
window.mainloop()