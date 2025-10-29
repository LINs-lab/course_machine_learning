# Project Road Segmentation -- Extract Roads from Satellite Images

For this choice of project task, we provide a set of satellite images acquired from GoogleMaps. 
We also provide ground-truth images where each pixel is labeled as road or background. 
Your goal is to train a classifier to segment roads in these images, i.e. assign a label {road=1, background=0} to each pixel.

### Dataset
* training.zip - the training set consisting of images with their ground truth
* test_set_images.zip - the test set
* sampleSubmission.csv - a sample submission file in the correct format
* mask_to_submission.py - script to make a submission file from a binary image
* submission_to_mask.py - script to reconstruct an image from the sample submission file

The sample submission file contains two columns:
* The first column corresponds to the image id followed by the x and y top-left coordinate of the image patch (16x16 pixels)
* The second column is the label assigned to the image patch

### Evaluation 
Your algorithm is evaluated according to the following criterion:
* [F1 score](https://en.wikipedia.org/wiki/F1_score) (this combines the two numbers of precision and recall)

### Submission
Please use the following procedure to submit your predicted solutions & codes:
- Unzip `test_set_images.zip` and extract the test images.
- Run your model on the images to generate prediction masks.
- Use `mask_to_submission.py` (a script likely provided in the competition) to convert the masks into a submission-ready `solution.csv` file.

# Project Tips
Obtain the python notebook `segment_aerial_images.ipynb` from the github 
folder, to see example code on how to extract the images as well as corresponding labels of each pixel.
* The notebook shows how to use `scikit learn` to generate features from each pixel, and finally train a linear classifier to predict whether each pixel is road or background. Or you can use your own code as well. 
* Our example code here also provides helper functions to visualize the images, labels and predictions. 
* In particular, the two functions `mask_to_submission.py` and `submission_to_mask.py` help you to convert from the submission format to a visualization, and vice versa.
