import os
import glob
import re
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input_xml_dir', type=str, default='./annot', help='path to root dir of xmls')
ap.add_argument('-s', '--subfolder', type=str, default='images', help='name of image subfolder')
args = vars(ap.parse_args())

xmls = glob.glob(os.path.join(args['input_xml_dir'], '*xml'))
print('found %d xmls.' % len(xmls))
subfolder = args['subfolder'] if not args['subfolder'].endswith('/') else args['subfolder'][:-1]
print('image sub folder:', subfolder)

pattern1 = r'<filename>(.*?)</filename>'
pattern2 = r'<folder>.*?</folder>'
pattern3 = r'<path>.*?</path>'
for xml in xmls:
    with open(xml, 'r') as fin:
        s = fin.read()
    filename = re.findall(pattern1, s)[0]
    s = re.sub(pattern2, '<folder>%s</folder>' % args['subfolder'], s)
    s = re.sub(pattern3, '<path>%s/%s/%s</path>' % (os.getcwd(), args['subfolder'], filename), s)
    with open(xml, 'wb') as fout:
        fout.write(s)
