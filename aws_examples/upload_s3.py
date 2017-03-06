import os
import glob
import date
import boto3

site = 'site'


def upload_s3(files):
    os.environ['AWS_ACCESS_KEY_ID'] = 'AWS_ACCESS_KEY_ID'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'AWS_SECRET_ACCESS_KEY'
    bucket = 'bucket'
    s3 = boto3.resource('s3')
    for f in glob.glob(files):
        outF = 'refs/%s/%s/%s' % (site, date.today(), os.path.basename(f))
        with open(f, 'rb') as fin:
            s3.Bucket(bucket).put_object(Key=outF, Body=fin)
        # print '%s --> s3:%s' % (f, outF)
