# Image Captioning Approaches
Build and experiment scalable Image Captioning techniques.

## Dataset Loaders
Data preparation and loading is complete for the following Image captioning datasets
1. Flickr8k
2. Google Conceptual Captions

### Download Conceptual Captions
1. The main dataset is available for download at `https://ai.google.com/research/ConceptualCaptions/download`

Steps to download images:
1. Download the training split from the dataset page
1. Set the `GOOGLE_CAPTIONS_FILE` url path in `preprocessing\GoogleCaptions.py`.
1. Choose the number of samples to download by setting `NUM_SAMPLES`
1. Run the script using `python3 GoocleCaptions.py`
1. The images would be downloaded in the set directory

## Keras Loaders
1. The keras data loaders are written for `Flickr8k` and `Google Conceptual Captions` at 
`datasets\flickr8k` and `datasets\googlecc.py`.
1. Currently, there is no separate usage of Validation set and a part of the training data is used for 
Validation.

## Training
 

