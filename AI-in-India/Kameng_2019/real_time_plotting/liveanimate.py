# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
# %matplotlib qt

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("/home/Desktop/freq_time.txt","r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y = eachLine.split(',')
            xar.append(int(x))
            yar.append(int(y))
    ax1.clear()
    ax1.plot(yar,xar)
ani = animation.FuncAnimation(fig, animate)
plt.show()
