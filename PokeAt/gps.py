
import os
import json
import datetime
import random
import re
from Tkinter import *
import pyautogui

pyautogui.PAUSE = 0.1
random.seed()

content = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxx="http://www.garmin.com/xmlschemas/GpxExtensions/v3" xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1" creator="mapstogpx.com" version="1.1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd http://www.garmin.com/xmlschemas/GpxExtensions/v3 http://www.garmin.com/xmlschemas/GpxExtensionsv3.xsd http://www.garmin.com/xmlschemas/TrackPointExtension/v1 http://www.garmin.com/xmlschemas/TrackPointExtensionv1.xsd">
    <wpt lat="%s" lon="%s"><name>WP%d</name><time>%s</time></wpt>
</gpx>
'''


def parse_time(s):
    year = int(s[:4])
    month = int(s[5:7])
    day = int(s[8:10])
    hour = int(s[11:13])
    minute = int(s[14:16])
    second = int(s[17:19])
    return datetime.datetime(year, month, day, hour, minute, second)


def make_time(d):
    return '%4d-%02d-%02dT%02d:%02d:%02dZ' % tuple(x for x in d.timetuple())[:-3]


def update_gpx(gpx, lat, lng, ts):
    # print 'update gpx file', lat, lng, ts
    n = random.randint(0, 2 << 20)
    dt = datetime.timedelta(seconds=random.randint(0, 9))
    ts = make_time(parse_time(ts) + dt)
    with open(gpx, 'wb') as fout:
        fout.write(content % (lat, lng, n, ts))
    return ts


def update_xcode(seq):
    # pyautogui.moveTo(x1, y1)
    # pyautogui.click(x1, y1)
    # pyautogui.moveTo(x2, y2)
    # pyautogui.click(x2, y2)
    for i in range(0, len(seq), 2):
        # print seq[i], seq[i+1]
        # pyautogui.moveTo(seq[i], seq[i+1])
        pyautogui.click(seq[i], seq[i+1])


conf = json.loads(open('gps_conf.json').read())
gpx = conf['gpx_file']
ts = conf['ts']
last_lat = float(conf['last_lat'])
last_lng = float(conf['last_lng'])
speed1 = float(conf['speed1'])
speed2 = float(conf['speed2'])
click_locs = conf['click_locs']


if not os.path.exists(gpx):
    print 'no gpx file found, init one'
    ts = update_gpx(gpx, last_lat, last_lng, ts)
    print 'done init gpx file'
else:
    print 'found gpx file, load last records'
    gpx_content = open(gpx).read()
    last_lat = float(re.findall(r'lat="(.*?)"', gpx_content)[-1])
    last_lng = float(re.findall(r'lon="(.*?)"', gpx_content)[-1])
    ts = re.findall(r'<time>(.*?)</time>', gpx_content)[-1]
    print last_lat, last_lng, ts


def key(event):
    global last_lat, last_lng, ts
    c = event.char

    if c == 'j' or c == 'J':
        r1 = random.random() * 0.01 * speed1
        r2 = random.random() * 0.01 * speed1
        last_lng -= speed1+r1 if c == 'j' else speed2+r1
        last_lat -= r2
        last_lng = round(last_lng, 7)
        last_lat = round(last_lat, 7)
        ts = update_gpx(gpx, last_lat, last_lng, ts)
        print 'left:\t', last_lat, last_lng, ts
        update_xcode(click_locs)
    elif c == 'l' or c == 'L':
        r1 = random.random() * 0.01 * speed1
        r2 = random.random() * 0.01 * speed1
        last_lng += speed1+r1 if c == 'l' else speed2+r1
        last_lat += r2
        last_lng = round(last_lng, 7)
        last_lat = round(last_lat, 7)
        ts = update_gpx(gpx, last_lat, last_lng, ts)
        print 'right:\t', last_lat, last_lng, ts
        update_xcode(click_locs)
    elif c == 'i' or c == 'I':
        r1 = random.random() * 0.01 * speed1
        r2 = random.random() * 0.01 * speed1
        last_lat += speed1+r1 if c == 'i' else speed2+r1
        last_lng += r2
        last_lng = round(last_lng, 7)
        last_lat = round(last_lat, 7)
        ts = update_gpx(gpx, last_lat, last_lng, ts)
        print '  up:\t', last_lat, last_lng, ts
        update_xcode(click_locs)
    elif c == 'k' or c == 'K':
        r1 = random.random() * 0.01 * speed1
        r2 = random.random() * 0.01 * speed1
        last_lat -= speed1+r1 if c == 'k' else speed2+r1
        last_lng -= r2
        last_lng = round(last_lng, 7)
        last_lat = round(last_lat, 7)
        ts = update_gpx(gpx, last_lat, last_lng, ts)
        print 'down:\t', last_lat, last_lng, ts
        update_xcode(click_locs)


def callback(event):
    frame.focus_set()
    # print "clicked at", event.x, event.y


root = Tk()
# frame = Frame(root, width=700, height=700)
frame = Frame(root, width=300, height=300)
frame.bind("<Key>", key)
frame.bind("<Button-1>", callback)
frame.pack()
root.mainloop()
