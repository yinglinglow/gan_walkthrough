"""
This function takes a csv file of links, downloads all the images in the links
and saves it in the current folder or uploads it to AWS.
To run: 
python3 1_2_downloading_wiki_pics.py --filename=items.csv --local=True
"""

def main(csv_filename, local, bucketname='NA'):
    import requests
    import pandas as pd
    from PIL import Image, ImageOps
    import boto3
    import io

    # import links to download
    links = pd.read_csv(csv_filename)
    all_links = list(links['link'])

    # run function for links
    start_from = 0 # default should be 0
    for i, url in enumerate(all_links, start=start_from):
        try:
            response = requests.get(url, stream=True) # get response
            im = Image.open(response.raw) # open image
            
            if local == True:
                im.save()

            else:
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
        
        except (IOError, OSError):
            print(i, " - OSError")
            pass

        except ValueError:
            print(i, " - ValueError")

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", required=True, help="Name of csv file to read links from")
    parser.add_argument("--local", "-l", required=True, help="True to save locally (False for AWS)")
    parser.add_argument("--bucketname", "-b", required=False, help="Name of bucket to save into")
    args = parser.parse_args()

    csv_filename = args.filename
    local = args.local
    try:
        bucketname = args.bucketname
    except:
        pass

    main(csv_filename, local, bucketname)