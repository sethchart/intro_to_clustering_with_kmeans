
# Introduction to Clustering: $k$-means

### By the end of this lecture, students will have:

- **Assessed** what scenarios could use $k$-means

- **Articulated** the methodology used by $k$-means

- **Applied** KMeans from sklearn.cluster to a relevant dataset

- **Selected** the appropriate number of clusters using $k$-means and the elbow method

- **Practiced** applying kmeans to an image color reduction problem


## Scenario

>You work for the marketing department within a large company that manages a customer base. 
For each customer you have a record of average purchase cost and time since last purchase.<br> 
You know that if you want to retain your customers you cannot treat them the same. You can use targeted marketing ads towards groups that demonstrate different behavior, but how will you divide the customers into groups?

## **Part 1**: Concept introduction
#### Import libraries and download dataset

We are continuing to use Scikit Learn as our main library.
The specific documentation for k-means can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).


```python
import os
import sys
module_path = os.path.abspath(os.pardir)
if module_path not in sys.path:
    sys.path.append(module_path)
```


```python
# Required packages for today
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import datasets

# Familiar packages for plotting, data manipulation, and numeric functions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import Alison's code for the demo clusters
from src.demo_images import *

# Have plots appear in notebook
%matplotlib inline

# Default plot params
plt.style.use('seaborn')
cmap = 'tab10'
```

## Clustering!   Finding **GROUPS**

How many groups do you see?

![img](../img/initialscenario.png)


## Wait - How is clustering different from classification?

>In _classification_ you **know** what groups are in the dataset and the goal is to _**predict**_ class membership accurately.

>In _clustering_ you **do not** know which groups are in the dataset and you are trying to _**identify**_ the groups.

### So what do you do with clustering results?

Clustering is often an *informing* step in your analysis. Once clusters are identified, one can:
- Create strategies on how to approach each group differently
- Use cluster membership as an independent variable in a predictive model
- Use the clusters as the _**target label**_ in future classification models. How would you assign new data to the existing clusters?

## Explore the algorithm with an intuitive K means approach

### Observe the following four methods with a sample dataset:

### Method Questions:

- What do they have in common?
- What are the differences between them?
- How many groups are there in the end?
- Do you see any problems with this method?

#### Method 1

![left](../img/from-left.gif)

#### Method 2

![right](../img/from-right.gif)

#### Method 3

![top](../img/from-top.gif)

#### Method 4

![bottom](../img/from-bottom.gif)

### Review Method Questions:

- What do they have in common?
- What are the differences between them?
- How many groups are there in the end?
- Do you see any problems with this method?

In common:
- Green dots starts at points
- Calculates distance
- Moves dots
- Re-measures distance
- Moves dots as needed


Differences:
- Dots start in different places and groups settle in different places

Groups:
- There are four groups

Problem with this method?
- Too variable!

### K-means algorithm, at its core, in an optimization function

![minmax](../img/minmaxdata.png)

### Reassigns groups and adjusts centroids to...
![min](../img/min.png)

### And to...
![max](../img/max.png)

The steps of the KMeans algorithm are pretty straightforward:
  1. Initialize cluster centers.
  2. Calculate the distance of every point to in the data set to each cluster center, and assign each point to the closest center.
  3. Make new cluster centers assigned to the averge of a all points labeled to the cluster.
  4. Repeat until some criteria has been met (the clusters no longer move)

**Sci-kit Learn** documentation actually has some pretty good [documentation describing the algorithm](https://scikit-learn.org/stable/modules/clustering.html#k-mean) if you wish for more detail.

#### Data for the exercise

- This is a sample dataset. 



```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```


```python
dummy_dat = pd.read_csv("data/xclara.txt",
                        header=0,
                        index_col=0)
dummy_dat.reset_index(inplace=True)
dummy_dat.drop('index', axis=1, inplace=True)
```


```python
dummy_dat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.072345</td>
      <td>-3.241693</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17.936710</td>
      <td>15.784810</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.083576</td>
      <td>7.319176</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.120670</td>
      <td>14.406780</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23.711550</td>
      <td>2.557729</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummy_dat.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2995</th>
      <td>85.65280</td>
      <td>-6.461061</td>
    </tr>
    <tr>
      <th>2996</th>
      <td>82.77088</td>
      <td>-2.373299</td>
    </tr>
    <tr>
      <th>2997</th>
      <td>64.46532</td>
      <td>-10.501360</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>90.72282</td>
      <td>-12.255840</td>
    </tr>
    <tr>
      <th>2999</th>
      <td>64.87976</td>
      <td>-24.877310</td>
    </tr>
  </tbody>
</table>
</div>



#### EDA of variables


```python
dummy_dat.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3000.000000</td>
      <td>3000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.611358</td>
      <td>22.862141</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25.859054</td>
      <td>31.759714</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-22.495990</td>
      <td>-38.795500</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.462790</td>
      <td>-4.003494</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>41.552210</td>
      <td>13.827390</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>62.249480</td>
      <td>55.729100</td>
    </tr>
    <tr>
      <th>max</th>
      <td>104.376600</td>
      <td>87.313700</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
ax.scatter(dummy_dat['V1'], dummy_dat['V2']);
```


![png](index_files/index_28_0.png)


#### Introduction of `KMeans` class


```python
# fit a KMeans Model on the dummy data. Initialize with n_clusters = 3
model = None
```


```python
#__SOLUTION__
model = KMeans(n_clusters=3).fit(dummy_dat)
```

Notice the `init` and `n_init` parameters!


```python
# Inspect the cluster centers attribute
```


```python
#__SOLUTION__
model.cluster_centers_
```




    array([[  9.4780459 ,  10.686052  ],
           [ 40.68362784,  59.71589274],
           [ 69.92418447, -10.11964119]])




```python
fig, ax = plt.subplots()
ax.scatter(dummy_dat['V1'], dummy_dat['V2'])
for i in range(len(model.cluster_centers_)):
    ax.scatter(model.cluster_centers_[i][0],
                model.cluster_centers_[i][1]);
```


![png](index_files/index_35_0.png)



```python
# Use the predict method on a list of 2 x and y values

```


```python
#__SOLUTION__
model.predict([[60, -20]])
```




    array([2], dtype=int32)




```python
fig, ax = plt.subplots()
ax.scatter(dummy_dat['V1'], dummy_dat['V2'],
           c='#f30303');
```


![png](index_files/index_38_0.png)



```python
fig, ax = plt.subplots()
ax.scatter(dummy_dat['V1'], dummy_dat['V2'],
           c= model.labels_);
```


![png](index_files/index_39_0.png)



```python
labeled_df = pd.concat([dummy_dat, pd.DataFrame(model.labels_,
                        columns=['cluster'])], axis=1)
```


```python
labeled_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.072345</td>
      <td>-3.241693</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17.936710</td>
      <td>15.784810</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.083576</td>
      <td>7.319176</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.120670</td>
      <td>14.406780</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23.711550</td>
      <td>2.557729</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## **Part 2**: Cluster Validation: Choosing the appropriate number of $k$

#### Two metrics we can use: **elbow method** and the **silhouette coefficient**

### **Part 2A**: Elbow Method

Elbow method uses the sum of squared error calculated from each instance of $k$ to find the best value of $k$.

This is sometimes called the "inertia" of the model, and fitted sklearn $k$-means models have an `inertia_` attribute.

Fewer clusters seems better, but inertia will always decrease with _more_ clusters. Hence the idea of looking for an elbow in the plot of inertia vs. $k$.


```python
model.inertia_
```




    611605.8806933895



Inertia is the sum of squared distances between points and their cluster center.


```python
# Specifying the dataset and initializing variables
X = dummy_dat
distortions = []

# Calculate SSE for different K
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=301)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plot values of SSE
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Elbow curve')
ax.set_xlabel('k')
ax.plot(range(2, 10), distortions)
ax.grid(True)
```


![png](index_files/index_46_0.png)


### **Part 2B**: Silhouette Coefficient

![silo](../img/silo2.png)

> **a** refers to the average distance between a point and all other points in that cluster.

> **b** refers to the average distance between that same point and all other points in clusters to which it does not belong

It is calculated for each point in the dataset, then averaged across all points for one cumulative score.

The Silhouette Coefficient ranges between -1 and 1. The closer to 1, the more clearly defined are the clusters. The closer to -1, the more incorrect assignment.





Suppose:
- I have four points in a one-dimensional space: 0, 1, 9, and 10; and
- I put them into two clusters: {0, 1} and {9, 10}.

Then we would calculate the Silhouette Score as follows:

For Point 0:
- $a=1$
- $b=9.5$
- $s(0) = \frac{9.5 - 1}{9.5} = \frac{17}{19}$

For Point 1:
- $a=1$
- $b=8.5$
- $s(1) = \frac{8.5 - 1}{8.5} = \frac{15}{17}$

For Point 9:
- $a=1$
- $b=8.5$
- $s(9) = \frac{8.5 - 1}{8.5} = \frac{15}{17}$

For Point 10:
- $a=1$
- $b=9.5$
- $s(10) = \frac{9.5 - 1}{9.5} = \frac{17}{19}$

The full Silhouette Score would be the average of all of these individual scores:

$\large s = \frac{2\left(\frac{17}{19}\right) + 2\left(\frac{15}{17}\right)}{4}$


```python
# Generate silhouette coefficient for each k
X = dummy_dat
silhouette_plot = []
for k in range(2, 10):
    clusters = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusters.fit_predict(X)
    silhouette_avg = metrics.silhouette_score(X, cluster_labels)
    silhouette_plot.append(silhouette_avg)
```


```python
# Plot Silhouette coefficient
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Silhouette coefficients over k')
ax.set_xlabel('k')
ax.set_ylabel('silhouette coefficient')
ax.plot(range(2, 10), silhouette_plot)
ax.axhline(y=np.mean(silhouette_plot), color="red", linestyle="--")
ax.grid(True)
```


![png](index_files/index_50_0.png)


# Activity

Let's practice k-means clustering with an image of a piece of art. 


```python
# Our new clustering class
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
# Allows us to visualize images through matplotlib plot methods
import matplotlib.image as mpimg

# Old favorites
import pandas as pd
import numpy as np
```

Let's look at a colorful Miro painting with matplotlib.


```python
fig, ax = plt.subplots(figsize=(10,10))
img = mpimg.imread('data/miro.jpg')
imgplot = ax.imshow(img)
```


![png](index_files/index_55_0.png)



```python
# What is the shape of the image, and what does each component represent?

```


```python
# Code here
```


```python
#__SOLUTION__
img.shape
```




    (1200, 1445, 3)




```python
# Let's look at one pixel

```


```python
#__SOLUTION__
# Let's look at one pixel
img[0,0,:]
```




    array([234, 194, 158], dtype=uint8)




```python
# Flatten the image so that each row represents one RGB triad
img_reshape = img.reshape()
```


```python
#__SOLUTION__
# flatten
img_reshape = img.reshape(img.shape[0]*img.shape[1], 3)
```


```python
# Check the shape
img_reshape.shape
```




    (1734000, 3)




```python
# after clustering, we will restore the original shape
# the code below demonstrates that the original image is restored by reshaping
# to the original dimensions 

fig, ax = plt.subplots(figsize=(10,10))
img = mpimg.imread('./data/miro.jpg')
restored_image = img_reshape.reshape(img.shape[0],img.shape[1], 3)
imgplot = ax.imshow(restored_image)


```


![png](index_files/index_64_0.png)


# In pairs: 10 minute exercise

Now, in pairs, we will use the KMeans algorithm to reduce the number of colors in the photo.   

Start by reducing the number of colors to 2.  To do so we will have to pass an appropriate argument  when instantianting a KMeans object.  The number of clusters we initiate will determine the number of colors that the image is reduced to.

In order to visualize the groupings, we will replace the original pixel values with the cluster centers associated with the assigned label.


```python
# Reminder of our flattened image
img_reshape.shape
```




    (1734000, 3)




```python
# Instantiate a KMeans object with the argument n_clusters equal to 2
# code here
km = None
```


```python
#__SOLUTION__
# Start with three colors
km = KMeans(2)

```


```python
# Fit the km object to img_reshape
# code here
```


```python
#__SOLUTION__
new_image = km.fit(img_reshape)
```


```python
# view the assigned labels via the labels_ attribute
# code here
```


```python
#__SOLUTION__
km.labels_
```




    array([0, 0, 0, ..., 0, 0, 0], dtype=int32)




```python
# view the cluster centers via the cluster_centers_ attribute
# code here
```


```python
#__SOLUTION__
km.cluster_centers_
```




    array([[200.90532216, 160.77198195, 129.26744708],
           [ 65.46684407,  62.99902106,  77.05050634]])




```python
# create a list which stores the cluster center associated with each label in a list.  
# The list should be 1734000 elements long

label_centers = []
for label in km.labels_:
    None
```


```python
#__SOLUTION__
label_centers = []
for label in km.labels_:
    label_centers.append(list(km.cluster_centers_[label]))
len(label_centers)
```




    1734000




```python
# Convert list to array
centers_2 = np.array(label_centers)
```


```python
# check shape is (1734000, 3)
centers_2.shape
```




    (1734000, 3)




```python
# reshape to (1200, 1445, 3)
new_image_2 = None
new_image_2.shape
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-247-50e848fc5cd5> in <module>
          1 # reshape to (1200, 1445, 3)
          2 new_image_2 = None
    ----> 3 new_image_2.shape
    

    AttributeError: 'NoneType' object has no attribute 'shape'



```python
#__SOLUTION__
new_image_2 = centers_2.reshape(img.shape[0],img.shape[1], 3)
new_image_2.shape

```




    (1200, 1445, 3)




```python
# Run the cell below to plot the new image.  It should have only 2 colors
```


```python
fig, ax = plt.subplots(figsize=(10,10))
imgplot = ax.imshow(new_image_2.astype(int))
```


![png](index_files/index_83_0.png)


### Explain in your own words why the image looks like it does.

Write answer here


```python
#__SOLUTION__
'''By finding clusters of nearest pixels, the KMeans algorithm reduces the number of colors coded in the picture.  With 2 clusters specified, KMeans finds the RGB pixel values which split the pixels into two clusters. These two centers are chosen because they represent the minimum total distance of the points in each group from their respective centers.'''
```




    'By finding clusters of nearest pixels, the KMeans algorithm reduces the number of colors coded in the picture.  With 2 clusters specified, KMeans finds the RGB pixel values which split the pixels into two clusters. These two centers are chosen because they represent the minimum total distance of the points in each group from their respective centers.'



Now, try out different numbers of clusters and see their affect on the painting.


```python
#__SOLUTION__
def paint_picture(n_clusters):
    
    '''
    Plot a picture with reduced color saturation via kmeans clustering.
    '''
    
    km = KMeans(n_clusters)
    km.fit(img_reshape)
    
    label_centers = []
    for label in km.labels_:
        label_centers.append(list(km.cluster_centers_[label]))
        
    centers = np.array(label_centers)
    new_image = centers.reshape(img.shape[0],img.shape[1], 3)

    fig, ax = plt.subplots(figsize=(10,10))
    imgplot = ax.imshow(new_image.astype(int))

    paint_picture(50)    
```


![png](index_files/index_88_0.png)

