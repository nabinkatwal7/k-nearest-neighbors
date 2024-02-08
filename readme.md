# What is K-Nearest Neighbors?

It is an algorithm that works on a simple assumption: *similar objects are found near each other.* 
It is like when you are in a huge library looking for books on, let's say baking. If you don't have a guide, you'll probably just grab books randomly until you find a cooking book, and then start grabbing books nearby as you hope they are about baking because cookbooks are usually kept in same spot. 

## How does KNN work?

KNN is like the memory whiz of machine learning algorithms. Instead of learning patterns and making predictions like many others do, KNN remembers every single detail of the training data. So, when you throw a new piece of data at it, it digs through everything it remembers to find the data points that are most similar to this new one. These similar points are its ‘nearest neighbors.’

To figure out which neighbors are closest, the algorithm measures the distance between the new data and everything it knows using methods like Euclidean or Manhattan distance. The choice of method matters a lot because it can change how KNN performs. For example, Euclidean distance works great for continuous data, while Manhattan distance is a go-to for categorical data.

After measuring the distances, KNN picks the ‘k’ closest ones. The ‘k’ here is important because it’s a setting you choose, and it can make or break the algorithm’s accuracy. If ‘k’ is too small, the algorithm can get too fixated on the noise in your data, which isn’t great. But if ‘k’ is too big, it might consider data points that are too far away, which isn’t helpful either.

For classification tasks, K-Nearest Neighbors looks at the most common class among these ‘k’ neighbors and goes with that. It’s like deciding where to eat based on where most of your friends want to go. For regression tasks, where you’re predicting a number, it calculates the average or sometimes the median of the neighbors’ values and uses that as the prediction.

What’s unique about KNN is it’s a ‘lazy’ algorithm, meaning it doesn’t try to learn a general pattern from the training data. It just stores the data and uses it directly to make predictions. It’s all about finding the nearest neighbors based on how you define ‘closeness,’ which depends on the distance method you use and the value of ‘k’ you set.

### Implementing KNN

1. Calculate Distance(Euclidean, Manhattan or Minkowski).
2. Identify the nearest neighbors. 
3. Aggregate Nearest Neighbors.
4. Predict the outcome.

#### Choosing the right K value.

Choosing the right number of neighbors, or ‘k’, in the K-Nearest Neighbors (KNN) algorithm is so important, that could be considered as one of the algorithm's limitations, as a poor choice would likely lead to a poor performance. The perfect ‘k’ helps the model catch the real patterns in the data, while the wrong ‘k’ could lead to guesses that are off the mark. Fortunately, there are a few techniques we can use to better understand what ‘k’ to use.

* What's your data like?
* How big is your data?
* What's the distribution of your data?

Lastly, don’t marry the first metric you meet. Play the field, try different metrics, and see which one makes your model the happiest through cross-validation.