import time, datetime
dt = "Thu Mar 19 19:52:18 +0000 2020"
dts = dt.split()
dtf = dts[:4]
dtf.append(dts[5])
dt = ' '.join(dtf)
print ((datetime.datetime.strptime(dt, "%a %b %d %H:%M:%S %Y")).timestamp())
