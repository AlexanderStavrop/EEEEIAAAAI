import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

############################################################
# Calculate the aspect ratio of the digit in the image     #
#                                                          #
# @param image: A 1D numpy array representing the image.   #
# @return: The aspect ratio of the digit in the image.     #
############################################################
def aspect_ratio(image):
  try:
    # Extract image data and reshape it (assuming data is in a column named 'image')
    img = image.values.reshape(28, 28)

    # Find non-zero foreground pixels
    nonzero_pixels = np.nonzero(img)

    # Check if there are any foreground pixels
    if nonzero_pixels[0].size == 0:
      return np.nan  # Return NaN if no foreground pixels found

    ########################################################################################################################## Added code starts here
    # Get minimum and maximum coordinates of foreground pixels
    min_col, min_row = np.min(nonzero_pixels[1]), np.min(nonzero_pixels[0])
    max_col, max_row = np.max(nonzero_pixels[1]), np.max(nonzero_pixels[0])

    # Calculate bounding box dimensions
    width = max_col - min_col
    height = max_row - min_row

    # Calculate aspect ratio
    aspect_ratio = width / height
    ########################################################################################################################## Added code ends here
    return aspect_ratio

  except (KeyError, ValueError) as e:
    print(f"Error processing image in row {image.name}: {e}")
    return np.nan  # Return NaN for rows with errors


############################################################
# Calculate the number of foreground pixels in the image   #
#                                                          #
# @param image: A 1D numpy array representing the image.   #
# @return: The number of foreground pixels in the image.   #
############################################################
def foreground_pixels(image):
    """
    Calculate the pixel density of the image, defined as the
    count of non-zero pixels

    Args:
    image (np.array): A 1D numpy array representing the image.

    Returns:
    int: The pixel density of the image.
    """
    try:
        # Extract image data and reshape it (assuming data is in a column named 'image')
        img = image.values.reshape(28, 28)

        ###################################################################################################################### Added code starts here
        # Find non-zero foreground pixels
        nonzero_pixels = np.count_nonzero(img)
        if nonzero_pixels == 0:
            print(f"Warning: Couldn't find nonzero pixels on  {image.name}")
            return np.nan  # Return NaN if no foreground pixels found
        return nonzero_pixels
        ###################################################################################################################### Added code ends here
    except (KeyError, ValueError) as e:
        print(f"Error processing image in row  {image.name}: {e}")
        return np.nan  # Return NaN for rows with errors


############################################################
# Calculate the centroid of the digit in the image         #
#                                                          #
# @param image: A 1D numpy array representing the image.   #
# @return: The centroid of the digit in the image.         #
############################################################
def calculate_centroid(image):
    """
    Calculate the normalized centroid (center of mass) of the image.

    Returns:
    tuple: The (x, y) coordinates of the centroid normalized by image dimensions.
    """
    # Extract image data and reshape it (assuming data is in a column named 'image')
    img = image.values.reshape(28, 28)
    rows, cols = img.shape
    ###################################################################################################################### Added code starts here
    total_mass = np.sum(img)
    x_center = np.sum(np.sum(img, axis=0) * np.arange(cols)) / total_mass
    y_center = np.sum(np.sum(img, axis=1) * np.arange(rows)) / total_mass
    # Create a single scalar as a centroid feature using x+(y * w) where w is the width of the image
    centroid = x_center + (y_center * cols)
    ###################################################################################################################### Added code ends here

    return centroid


############################################################
# Min-max scaling of the feature set                       #
#                                                          #
# @param X: The feature set to scale.                      #
# @param min_val: The minimum value of the range.          #
# @param max_val: The maximum value of the range.          #
# @return: The scaled feature set.                         #   
############################################################
def min_max_scaling(X, min_val=-1, max_val=1):
    """Scales features to a range between min_val and max_val."""
    ########################################################################################################################## Added code starts here
    X_normalized = (X - X.min()) / (X.max() - X.min())      # Min-max normalization
    X_scaled = X_normalized * (max_val - min_val) + min_val # Scaling to the desired range
    ########################################################################################################################## Added code ends here
    return X_scaled


############################################################
# Calculate the bounding box around the digit in an image  #
#                                                          #
# @param image: A 1D numpy array representing the image.   #
# @return: The bounding box around the digit in the image. #
############################################################
def calculate_bounding_box(image):
    # Find non-zero foreground pixels
    nonzero_pixels = np.nonzero(image)
    # Check if there are any foreground pixels
    if nonzero_pixels[0].size == 0:
        return np.nan  # Return NaN if no foreground pixels found

    # Get minimum and maximum coordinates of foreground pixels
    ########################################################################################################################## Added code starts here
    min_col, min_row = np.min(nonzero_pixels[1]), np.min(nonzero_pixels[0])
    max_col, max_row = np.max(nonzero_pixels[1]), np.max(nonzero_pixels[0])
    ########################################################################################################################## Added code ends here

    return min_col-1, min_row-1, max_col+1, max_row+1


########################################################
# Visualize the bounding box around the digit in an image
#
# @param image: A 1D numpy array representing the image.
# @param color: The color of the bounding box.
########################################################
def visualize_bounding_box(image, color='red'):
    """Visualizes the bounding box around the digit in an image."""
    bbox = calculate_bounding_box(image)

    # Create a drawing object
    sample_image_img = Image.fromarray(image.astype(np.uint8)).convert('RGB')
    scaling = 10
    sample_image_XL = sample_image_img.resize((28 * scaling, 28 * scaling), resample=Image.NEAREST)

    draw = ImageDraw.Draw(sample_image_img)
    # Draw the rectangle with desired fill color and outline (optional)
    draw.rectangle(bbox, outline=color, width=1)

    sample_image_XL.show()
    sample_image_XL_bbox = sample_image_img.resize((28 * scaling, 28 * scaling), resample=Image.NEAREST)
    sample_image_XL_bbox.show()


###########################################################################
# Naive Bayes Classifier for Gaussian distributions                       #
#                                                                         #
# @param class_priors: The prior probabilities for each class.
# @param class_stats: The mean and covariance for each class.
###########################################################################
class MyBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.class_stats = {}

    def train(self, X, y):
        """
        Train the classifier under the assumption of Gaussian distributions:
        calculate priors and Gaussian distribution parameters for each class.

        Args:
        X (pd.DataFrame): DataFrame with features.
        y (pd.Series): Series with target class labels.
        """
        ########################################################################################################################## Added code starts here
        self.classes_ = np.unique(y)
        for class_label in self.classes_:
            # Filter data by class
            X_class = X[y == class_label]
            # Calculate prior probability for the class
            self.class_priors[class_label] = len(X_class) / len(X)
        ########################################################################################################################## Added code ends here

        ########################################################################################################################## Added code starts here
            # Calculate mean and covariance for the class
            # Adding a small value to the covariance for numerical stability
            self.class_stats[class_label] = {
                'mean': X_class.mean(),
                'cov': X_class.cov() + 1e-4 * np.identity(X.shape[1])
            }
        ########################################################################################################################## Added code ends here

    def predict(self, X):
        """
        Predict class labels for each test sample in X.

        Args:
        X (pd.DataFrame): DataFrame with features to predict.

        Returns:
        np.array: Predicted class labels.
        """
        ########################################################################################################################## Added code starts here
        # For each sample in X, predict the class
        predictions = X.apply(self._predict_instance, axis=1)
        ########################################################################################################################## Added code ends here
        return np.array(predictions)

    def _predict_instance(self, x):
        """
        Private helper to predict the class for a single instance.

        Args:
        x (pd.Series): A single data point's features.

        Returns:
        The predicted class label.
        """
        posteriors = []

        # Calculate the posterior probability for each class
        ########################################################################################################################## Added code starts here
        for class_label in self.classes_:
            prior = self.class_priors[class_label]
            mean = self.class_stats[class_label]['mean']
            cov = self.class_stats[class_label]['cov']
            likelihood = self._calculate_likelihood_1D(x, mean, cov)
            posterior = prior * likelihood
            posteriors.append(posterior)

        # Choose the class with the highest posterior probability
        prediction = self.classes_[np.argmax(posteriors)]
        ########################################################################################################################## Added code ends here
        return prediction

    def _calculate_likelihood_1D(self, x, mean, cov):
        """
        Calculate the Gaussian likelihood of the data x given class statistics.

        Args:
        x (pd.Series): Features of the data point.
        mean (pd.Series): Mean features for the class.
        cov (pd.DataFrame): Covariance matrix of the features for the class.

        Returns:
        float: The likelihood value.
        """
        ########################################################################################################################## Added code starts here
        likelihood = multivariate_normal.pdf(x, mean=mean, cov=cov)
        ########################################################################################################################## Added code ends here
        return likelihood


###########################################################################
######    MAIN - CREATE FEATURES - TRAIN (NAIVE) BAYES CLASSIFIER    ######
###########################################################################
def main(test_case = 0):
    # Read the training samples from the corresponding file
    nTrainSamples = 10000 # specify 'None' if you want to read the whole file
    df_train = pd.read_csv('data/mnist_train.csv', delimiter=',', nrows=nTrainSamples)
    df_train = df_train[df_train['label'].isin([1, 2])] # Get samples from the selected digits only
    target_train = df_train.label
    data_train = df_train.iloc[:, 1:]

    # Read the test samples from the corresponding file
    nTestSamples = 1000 # specify 'None' if you want to read the whole file
    df_test = pd.read_csv('data/mnist_test.csv', delimiter=',', nrows=nTestSamples)
    
    df_test = df_test[df_test['label'].isin([1, 2])] # Get samples from the selected digits only
    target_test = df_test.label
    data_test = df_test.iloc[:, 1:]

    ############################# Create the features #############################
    # if test_case == 0:
    #     # Calculate aspect ratio as the first feature
    #     df_train['aspect_ratio'] = data_train.apply(aspect_ratio, axis=1)
    #     df_train['aspect_ratio'] = df_train['aspect_ratio']
    
    # elif test_case == 1:
    # Calculate aspect ratio as the first feature
    df_train['aspect_ratio'] = data_train.apply(aspect_ratio, axis=1)
    df_train['aspect_ratio'] = min_max_scaling(df_train['aspect_ratio'])

    if test_case == 1 or test_case == 2:
        # Calculate the number of non-zero pixels as the second feature
        df_train['fg_pixels'] = data_train.apply(foreground_pixels, axis=1)
        df_train['fg_pixels'] = min_max_scaling(df_train['fg_pixels'])
    
    if test_case == 2:
        # Calculate the centroid feature as the third feature
        df_train['centroid'] = data_train.apply(calculate_centroid, axis=1)
        df_train['centroid'] = min_max_scaling(df_train['centroid'])
    
    # Draw 10 sample images from the training data to make sure aspect ratio is correct
    # for sample in range (2):
    #     sample_image = data_train.iloc[sample].values.reshape(28, 28)
    #     visualize_bounding_box(sample_image)

    # Define the features to use for both train and test in this experiment
    if test_case == 0:
        features = ["aspect_ratio"]
    elif test_case == 1:
        features = ["aspect_ratio", "fg_pixels"]
    elif test_case == 2:
        features = ["aspect_ratio", "fg_pixels", "centroid"]
    
    ###############################################################################
    trainData = df_train[features]
    
    # Create the Classifier object and train the Gaussian parameters (prior, mean, cov)
    classifier = MyBayesClassifier()
    # Train the classifier
    classifier.train(trainData,target_train)

    ##############################################################################
    # Create the respective features for the test samples
    df_test['aspect_ratio'] = data_test.apply(aspect_ratio, axis=1)
    df_test['aspect_ratio'] = min_max_scaling(df_test['aspect_ratio'])
  
    if test_case == 1 or test_case == 2:
        df_test['fg_pixels'] = data_test.apply(foreground_pixels, axis=1)
        df_test['fg_pixels'] = min_max_scaling(df_test['fg_pixels'])
    
    if test_case == 2:
        df_test['centroid'] = data_test.apply(calculate_centroid, axis=1)
        df_test['centroid'] = min_max_scaling(df_test['centroid'])
  
    # Predict on the test samples (for the given feature set)
    test_data = df_test[features]
    predictions = classifier.predict(test_data)

    # Calculate accuracy as an example of validation
    accuracy = accuracy_score(target_test, predictions)
    print("Classification accuracy:", accuracy)


###########################################################
###########################################################
if __name__ == "__main__":
    print("Test case 1: Aspect ratio only")
    main(0)
    print("\nTest case 2: Aspect ratio and number of foreground pixels")
    main(1)
    print("\nTest case 3: Aspect ratio, number of foreground pixels, and centroid")
    main(2)
