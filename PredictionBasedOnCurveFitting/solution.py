
import scipy as sp
import matplotlib.pyplot as plt

data = sp.genfromtxt('web_traffic.tsv', delimiter='\t')
print data.shape
print data[:10]

x = data[:, 0]
y = data[:, 1]
print sp.sum(sp.isnan(y))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

plt.scatter(x, y, s=10)
plt.title('Page views in last month')
plt.xlabel('Time')
plt.ylabel('PV/hour')
# set x ticks by week count
plt.xticks([w*7*24 for w in range(10)], ['week %d'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')
plt.show()
