import requests
import pandas as pd
from PIL import Image, ImageOps
import boto3
import io

def open_save(i, url, bucketname, destination='local'):
    """ 
    Support function: this function opens the image url and saves it.
    If destination = 'local', it saves locally, in the current folder.
    If destination = 'aws', it saves to a temporary file and uploads to AWS S3.
    """

    # get response
    response = requests.get(url, stream=True)
    # open image
    im = Image.open(response.raw)
    
    if destination == 'local':
        im.save()

    elif destination == 'aws':
        # save image to temp file
        in_mem_file = io.BytesIO() # create temporary file to save resized image to
        im.save(in_mem_file, format=im.format)
        in_mem_file.seek(0) # seek beginning of saved file

        # use s3 in boto3
        s3 = boto3.resource('s3')

        # define bucket and name of file to save to
        bucket_name = bucketname
        im_name = str(str(i)+"."+im.format)
        content_type = str("image/"+im.format)
        
        # check to make sure S3 access is OK
        for bucket in s3.buckets.all():
            if bucket.name == bucket_name:
                good_to_go = True

            if not good_to_go:
                print('Not seeing your s3 bucket, might want to double check permissions in IAM')

        # upload to s3
            s3.Bucket(bucket_name).put_object(Key=im_name, Body=in_mem_file, ContentType=content_type)
            print(i)

    else:
        print("Please input valid destination - only 'local' or 'aws'")


def main():
    # arguments to change
    csv_filename = 'items_clean.csv'
    bucketname = ''

    # import links to download
    links = pd.read_csv(csv_filename)
    all_links = list(links['link'])

    # run function for links
    start_from = 0 # default should be 0
    for i, url in enumerate(all_links, start=start_from):
        try:
            open_save(i, url, bucketname)
        
        except (IOError, OSError):
            print(i, " - OSError")
            pass

        except ValueError:
            print(i, " - ValueError")

if __name__ == '__main__':
    main()