## Introduction
We use the UTKface dataset (https://susanqq.github.io/UTKFace/) for race classification task. Then, we use a real scenario where we crowdsource new images using Amazon Mechanical Turk (AMT). We design a task by asking a worker to find new face images of a certain demographic (e.g., slice = White Women) from any website. We pay 4 cents per image found, employ workers from all possible countries, and collected images during 8 separate time periods. 

## Dataset
The file name of each image is formatted as [race][gender]\_[image_number].png
* [race] denotes White, Black, Asian, and Indian.
* [gender] is either male or female.
* [image number] is the image number in the slice.

## Post-processing
* Some workers make mistakes and collect incorrect images that do not fit in the specified demographic. Hence, we filter out obvious errors manually and remove exact duplicates.
* The aligned and cropped images are obtaind by Google Cloud Vision API.

## License Issue
* UTKface crowdsourcing data is only available for non-commercial purposes.
* The copyright belongs to the original owners. If any of the images belongs to you, please let us know and we will remove it immediately.
