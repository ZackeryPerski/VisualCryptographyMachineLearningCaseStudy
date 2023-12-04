## VisualCryptographyMachineLearningCaseStudy
# What This Is:
This repository contains the project files to build and run a FFN and a CNN that is capable of classifying overlayed images that are utilizing a visual cryptography scheme. These models utilize a binary classification scheme that essentially answers a yes or no question, that question being 'does the hidden image show up'?

If you're not sure what visual cryptography is, here's a quick link: https://en.wikipedia.org/wiki/Visual_cryptography

To run this project, you will first need to run the ImagePreparer.py file to generate the testing and training sets. These are dynamically created utilizing the MNIST Hand Written Digits dataset and are currently not available anywhere else for download.

You will need at least 1.5 GB of local storage to download and utilize this project, the majority of the space required being taken up by the produced image data set.

# Planned Future Improvements: 
The FFN is currently based on hand written code and manual backpropogation. For proper testing and comparison, it will be rewritten to utilize tensorflow.keras as well.

Additional classification levels is also planned. Given that the underlying dataset that was converted is MNIST, we plan to eventually build and test a digit classifier on top of the the binary image revealed classifier model.

