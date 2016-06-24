
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
# plt.show()


def error(f, x, y):
    return sp.sum(sp.power(f(x) - y, 2))

fp1 = sp.polyfit(x, y, 1)
print fp1  # [   2.59619213  989.02487106]
f1 = sp.poly1d(fp1)
print type(f1)  # <class 'numpy.lib.polynomial.poly1d'>
print f1  # 2.596 x + 989
print error(f1, x, y)  # (317389767.34+0j)

fx = sp.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), color='green', linewidth=4)
plt.legend(['d = %d' % f1.order], loc='upper left')
plt.show()
