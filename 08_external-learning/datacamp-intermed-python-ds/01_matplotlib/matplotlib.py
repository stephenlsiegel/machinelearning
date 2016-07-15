########## Notes on matplotlib ##########

### Lines and Scatter Plots

import matplotlib.pyplot as plt

year = [1950,1970,1990,2010]
pop = [2.519,3.692,5.263,6.972]

# plt.plot(year, pop)
# plt.show() # required to actually display the plot

# plot tells python what to plot and how to plot it
# show actually reveals the plot

plt.scatter(year, pop)
plt.show()

plt.xscale('log') makes the x axis on a logarithmic scale


### Histograms

import matplotlib.pyplot as plt

# We will use the plt.hist function
values = [0,0.6,1.4,1.6,2.2,2.5,2.6,3.2,3.5,3.9,4.2,6]
plt.hist(values, bins=3)


### Customizations

# Label axis
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('Plot Title')
plt.yticks([list of numbers to put tick marks at], [list of display names of ticks])

# Change size, color, opacity of dots in scatter plot
plt.scatter(x, y, s=[list of sizes], c=[list of colors], alpha=numeric between 0 and 1 where 0 is transparent and 1 is opaque)

# Add text
plt.text(xpos, ypos)