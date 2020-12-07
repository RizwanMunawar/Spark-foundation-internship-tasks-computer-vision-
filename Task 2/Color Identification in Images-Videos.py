#!/usr/bin/env python
# coding: utf-8

# ### THE SPARK FOUNDATION INTERNSHIP
# 
# #### Name: Muhammad Rizwan Munawar
# 
# #### Domain:Computer Vision & IOT

# I have used KMeans Clustering Algorithm for colors Detection in images

# #### Step-1: Importing Libraries

# In[7]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_cie76


# #### Step-2: Reading image

# In[8]:


image = cv2.imread('sample_image.jpg')
plt.imshow(image)
plt.show()


# #### Step-3: Checking Type and Shape of image Data

# In[9]:


print("The type of this input is {}".format(type(image)))
print("Shape: {}".format(image.shape))


# #### NOTE

# We can see that the image has different colors as compared to the original image. This is because by default OpenCV reads the images in the color order "Blue-Green-RED" i.e. BGR. Thus, we need to convert it into "RED GREEN BLUE" i.e. RGB.

# #### Step-4: Converting BGR-To-RGB

# In[10]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()


# #### Note:

# Now we need to convert RGB image to grayscale becuase in RGB image we have 3 channel so computer need more time and memory to understand what's inside image but with grayscale we only left with 2d array so it's easy for computer to understand

# #### Step-5: Converting Image to GrayScale

# In[11]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.show()


# #### Note

# We need to resize the image to a certain size becuase we have images that are huge in size with different dimensions.

# #### Step-6: Resizing Images

# In[13]:


resized_image = cv2.resize(image, (1200, 600))
plt.imshow(resized_image)
plt.show()


# #### Step-7:Let us start color Identification

# First, we will define a function that can give us the `hex` values of our the colors that we will identify.

# In[14]:


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# #### Note

# KMeans expects flattened array as input during its fit method. 
# Thus, we need to reshape the image using numpy. 
# Then, we can apply KMeans to first fit and then predict on the image to get the results. 
# Then, the cluster colors are identified an arranged in the correct order. We plot the colors as a pie chart.
# I have combined all the steps in two method.

# #### Step-8: Defining method for getting images & Conversion from BGR-TO-RGB

# In[18]:


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# #### Step-9: Defining method along with K_Means Algorithm

# First we need to resize data into same size, then converting into array on which i fitted model

# In[19]:


def get_colors(image, number_of_colors, show_chart): 
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    
    return rgb_colors


# #### Step-10: Calling Above Two Functions inside each other

# In[25]:


get_colors(get_image('sample_image.jpg'), 8, True)


# #### Step-11: Search images using Color

# From the model above, we can extract the major colors. 
# This create the opportunity to search for images based on certain colors. We can select a color and if it's hex matches or is close to the hex of the major colors of the image, we say it's a match.
# We first get all the images and store them in the "images" variable.

# In[21]:


IMAGE_DIRECTORY = 'images'

COLORS = {'GREEN': [0, 128, 0],'BLUE': [0, 0, 128],'YELLOW': [255, 255, 0]}

images = []

for file in os.listdir(IMAGE_DIRECTORY):
    if not file.startswith('.'):
        images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))


# #### Step-12: Visualization of Data

# In[23]:


plt.figure(figsize=(20, 5))
for i in range(len(images)):
    plt.subplot(1, len(images), i+1)
    plt.imshow(images[i])
    plt.axis('off')


# #### Step-13: Function for finding matches by using top 10 colors in images

# I define the function below. I will try to match with the top 10 colors of the image. It is highly possible that there will be no extact match for the hex codes, thus we calculate the similarity between the chosen color and the colors of the image. We keep a threshold value such that if the difference between the chosen color and any of the selected colors is less than that threshold, we declare it as a match. Hex values or RGB values cannot be directly compared so I first convert them to a device independant and color uniform space. We use "RGB2LAB" to convert the values and then find the difference using "deltaE_cie76". The method calculates the difference between all top 5 colors of the image and the selected color and if atleast one is below the threshold, we show the image.

# In[27]:


def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    
    for i in range(number_of_colors):
        
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        
        if (diff < threshold):
            select_image = True
    
    return select_image


# #### Step-14: Function for selection of images (Match/Mismatch)

# I call the above method for all the images in our set and show relevant images out of the same that approximately match our selected color.

# In[35]:


def show_selected_images(images, color, threshold, colors_to_match):
    index = 1
    
    for i in range(len(images)):
        selected = match_image_by_color(images[i],
                                        color,
                                        threshold,
                                        colors_to_match)
        if (selected):
            plt.subplot(1, 5, index)
            plt.imshow(images[i])
            plt.axis('off')
            index += 1


# #### Step-15:Calling above methods and visualizing results

# Finding GREEN COLOR

# In[36]:


plt.figure(figsize = (20, 8))
show_selected_images(images, COLORS['GREEN'], 60, 5)


# Finding Blue Color

# In[38]:


plt.figure(figsize = (20, 10))
show_selected_images(images, COLORS['BLUE'], 60, 5)


# Finding Yellow Color

# In[39]:


plt.figure(figsize = (20, 10))
show_selected_images(images, COLORS['YELLOW'], 60, 5)


# #### Conclusion

# I have used KMeans Clustering Algorithm to extract majority colors from images. 
# 
# then I used the RGB Values of Colors to identify images from a collection that have that color in them.

# #### References & Special Thanks

# 1 - https://github.com/kb22/Color-Identification-using-Machine-Learning

# 2- https://towardsdatascience.com/color-identification-in-images-machine-learning-application-b26e770c4c71
