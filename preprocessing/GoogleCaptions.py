import requests
import cv2
import uuid
import os

WORKSPACE = "/dataset/google-cc/"
IMAGE_FORMAT = ".jpg"
GOOGLE_CAPTIONS_FILE = "/dataset/google-cc/train_GCC-training.tsv"
NUM_SAMPLES = 1000


def download_image(url):
    file_name = str(uuid.uuid1())
    file_address = os.path.join(WORKSPACE, file_name + IMAGE_FORMAT)

    image_data = requests.get(url).content
    with open(file_address, 'wb') as handler:
        handler.write(image_data)

    image = cv2.imread(file_address)
    image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_AREA)
    cv2.imwrite(file_address, image)

    return url, uuid


def main():

    file = open(GOOGLE_CAPTIONS_FILE, 'r')
    index_file = open(os.path.join(WORKSPACE, "index_file.txt"), 'a')
    count = 0
    for line in file:
        url, uuid = download_image(line.strip().split("\t")[-1])
        index_file.write(url+","+uuid+"\n")
        count += 1
        if count >= NUM_SAMPLES:
            break


if __name__ == "__main__":
    main()
