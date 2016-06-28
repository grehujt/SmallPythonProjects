
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from scipy.optimize import fsolve

data = np.genfromtxt('web_traffic.tsv', delimiter='\t')
print data.shape
print data[:10]

x = data[:, 0]
y = data[:, 1]
print np.sum(np.isnan(y))

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

plt.scatter(x, y, s=10)
plt.title('Page views over last month')
plt.xlabel('Time')
plt.ylabel('PV/hour')
# set x ticks by week count
plt.xticks([w*7*24 for w in range(10)], ['week %d'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')
# plt.show()


def error(f, x, y):
    return np.sum(np.power(f(x) - y, 2))

fp1 = np.polyfit(x, y, 1)
print fp1  # [   2.59619213  989.02487106]
f1 = np.poly1d(fp1)
print type(f1)  # <class 'numpy.lib.polynomial.poly1d'>
print f1  # 2.596 x + 989
print error(f1, x, y)  # (317389767.34+0j)

# fx = np.linspace(0, x[-1], 1000)
fx = np.linspace(0, 6*7*24, 1000)
plt.ylim(0, 10000)
plt.plot(fx, f1(fx), linewidth=4)
# plt.legend(['d = %d' % f1.order], loc='upper left')
# plt.show()

fp2 = np.polyfit(x, y, 2)
print fp2  # [  1.05322215e-02  -5.26545650e+00   1.97476082e+03]
f2 = np.poly1d(fp2)
print type(f2)  # <class 'numpy.lib.polynomial.poly1d'>
print f2  # 0.01053 x^2 - 5.265 x + 1975
print error(f2, x, y)  # (179983507.878+0j)

plt.plot(fx, f2(fx), linewidth=4)
# plt.legend(['d = %d' % f.order for f in [f1, f2]], loc='upper left')
# plt.show()

fp3 = np.polyfit(x, y, 3)
f3 = np.poly1d(fp3)
print error(f3, x, y)
plt.plot(fx, f3(fx), linewidth=4)

fp10 = np.polyfit(x, y, 10)
f10 = np.poly1d(fp10)
print error(f10, x, y)
plt.plot(fx, f10(fx), linewidth=4)

fp50 = np.polyfit(x, y, 50)
f50 = np.poly1d(fp50)
print error(f50, x, y)
plt.plot(fx, f50(fx), linewidth=4)

fs = [f1, f2, f3, f10, f50]
plt.legend(['d = %d' % f.order for f in fs], loc='upper left', prop={'size': 10})

table = plt.table(cellText=[['%.2e' % error(f, x, y) for f in fs]],
    colWidths=[0.13] * len(fs),
    rowLabels=['error'],
    colLabels=['order %d' % f.order for f in fs],
    loc='upper right', zorder=100)
table.scale(1, 1.5)
# plt.show()


def gen_model(x, y, degree):
    fp = np.polyfit(x, y, degree)
    return np.poly1d(fp)

plt.figure()
plt.scatter(x, y, s=10)
plt.title('Page views over last month')
plt.xlabel('Time')
plt.ylabel('PV/hour')
# set x ticks by week count
plt.xticks([w*7*24 for w in range(10)], ['week %d'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')
plt.ylim(0, 10000)
y_lastweek = y[x > 3.5*7*24]
x_lastweek = x[x > 3.5*7*24]
degrees = [1, 2, 3, 10, 50]
fs = [gen_model(x_lastweek, y_lastweek, i) for i in degrees]
for f in fs:
    plt.plot(fx, f(fx), linewidth=4)
plt.legend(['d = %d' % f.order for f in fs], loc='upper left', prop={'size': 10})
table = plt.table(cellText=[['%.2e' % error(f, x, y) for f in fs]],
    colWidths=[0.13] * len(fs),
    rowLabels=['error'],
    colLabels=['order %d' % f.order for f in fs],
    loc='upper right', zorder=100)
table.scale(1, 1.5)
# plt.show()


result = {}
for train, test in KFold(y_lastweek.shape[0], 5):
    # print train.shape, test.shape
    x_train, x_test = x_lastweek[train], x_lastweek[test]
    y_train, y_test = y_lastweek[train], y_lastweek[test]
    for d in [1, 2, 3, 10, 50]:
        if d not in result:
            result[d] = []
        f = gen_model(x_train, y_train, d)
        result[d].append(error(f, x_test, y_test))

for k, v in result.iteritems():
    print k, '\t'.join('%.2e' % x for x in v), '\t%.2e' % np.mean(v)


bestF = gen_model(x_lastweek, y_lastweek, 2) - 10000
roots = fsolve(bestF, x0=800) / (7.0 * 24)
print roots  # [ 5.0628393]

plt.figure()
plt.xticks([w*7*24 for w in range(10)], ['week %d'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')
plt.ylim(0, 10000)
plt.plot(fx, bestF(fx), linewidth=4)
plt.show()
