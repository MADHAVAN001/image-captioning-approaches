import concurrent.futures
import os
import sys
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np

sys.path.append("..")
from preprocessing.utils import make_directories

IMAGE_FORMAT = ".jpg"
GOOGLE_CAPTIONS_FILE = "/dataset/google-cc/train_GCC-training.tsv"
NUM_SAMPLES = 10000
WORKSPACE = "/dataset/google-cc/" + "10k" + "/images"


def download_image(url):
    file_name = str(uuid.uuid1())
    file_address = os.path.join(WORKSPACE, file_name + IMAGE_FORMAT)

    print(file_address)

    try:
        response = urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError as e:
        print("{}:{}".format(url, e.reason))
        return None, None
    except:
        print("{}:General Error".format(url))
        return None, None
    else:
        try:
            image = np.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_address, image)

            return url, file_name
        except cv2.error as e:
            print("{}:cv2.error".format(url))
            return None, None


def main():
    file = open(GOOGLE_CAPTIONS_FILE, 'r')

    # Ensure workspace directory exists
    make_directories(WORKSPACE)

    index_file = open(os.path.join(WORKSPACE, "index_file.txt"), 'a')
    count = 0

    future_list = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        for line in file:
            count += 1
            future_list.append(executor.submit(download_image, line.strip().split("\t")[-1]))
            if count >= NUM_SAMPLES:
                break

        for future in concurrent.futures.as_completed(future_list):
            try:
                url, file_name = future.result()
            except Exception as exc:
                print("Executor Service exception..!")
            else:
                if url is not None:
                    index_file.write(url + " " + file_name + "\n")


if __name__ == "__main__":
    main()
