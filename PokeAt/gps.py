
import os
import json
import time
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

conf = json.loads(open('gps_conf.json').read())
gpx = conf['gpx_file']
last_lat = float(conf['last_lat'])
last_lng = float(conf['last_lng'])
speed1 = float(conf['speed1'])
speed2 = float(conf['speed2'])
click_locs = conf['click_locs']


def _make_time():
    d = datetime.datetime.now()
    return '%4d-%02d-%02dT%02d:%02d:%02dZ' % tuple(x for x in d.timetuple())[:-3]


def _update_gpx(gpx, lat, lng):
    # print 'update gpx file', lat, lng
    n = random.randint(0, 2 << 20)
    lng = round(lng, 7)
    lat = round(lat, 7)
    with open(gpx, 'wb') as fout:
        fout.write(content % (lat, lng, n, _make_time()))


def _update_xcode(seq):
    for i in range(0, len(seq), 2):
        pyautogui.click(seq[i], seq[i+1])


def update(c):
    global last_lat, last_lng
    r1 = random.random() * 0.1 * speed1
    r2 = random.random() * 0.1 * speed1
    if c == 'j' or c == 'J':
        last_lng -= speed1+r1 if c == 'j' else speed2+r1
        last_lat -= r2
    elif c == 'l' or c == 'L':
        last_lng += speed1+r1 if c == 'l' else speed2+r1
        last_lat += r2
    elif c == 'i' or c == 'I':
        last_lat += speed1+r1 if c == 'i' else speed2+r1
        last_lng += r2
    elif c == 'k' or c == 'K':
        last_lat -= speed1+r1 if c == 'k' else speed2+r1
        last_lng -= r2
    print c, last_lat, last_lng
    _update_gpx(gpx, last_lat, last_lng)
    _update_xcode(click_locs)


def key(event):
    c = event.char
    if c == 'r':
        while True:
            tmp = random.choice(['i', 'j', 'k', 'l'])
            startT = datetime.datetime.now()
            while (datetime.datetime.now() - startT).seconds < 30:
                update(tmp)
                time.sleep(3)
    else:
        update(c)


def callback(event):
    frame.focus_set()
    # print "clicked at", event.x, event.y


if not os.path.exists(gpx):
    print 'no gpx file found, init one'
    _update_gpx(gpx, last_lat, last_lng)
    print 'done init gpx file'
else:
    print 'found gpx file, load last records'
    gpx_content = open(gpx).read()
    last_lat = float(re.findall(r'lat="(.*?)"', gpx_content)[-1])
    last_lng = float(re.findall(r'lon="(.*?)"', gpx_content)[-1])
    print last_lat, last_lng

root = Tk()
# frame = Frame(root, width=700, height=700)
frame = Frame(root, width=300, height=300)
frame.bind("<Key>", key)
frame.bind("<Button-1>", callback)
frame.pack()
root.mainloop()
