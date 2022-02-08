# Artificial-Neural-Networks

![1-Logo](Images/ANN-sales.png)

Pretend this data are measurements of some rare gem stones, with 2 measurement features and a sale price. Our final goal would be to try to predict the sale price of a new gem stone we just mined from the ground, in order to try to set a fair price in the market.

Test/Train Split

Normalizing/Scaling the Data

TensorFlow 2.0 Syntax

Model - as a list of layers or adding in layers one by one

For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')
              
Sample: one element of a dataset.

Example: one image is a sample in a convolutional network. Example: one audio file is a sample for a speech recognition model

Batch: a set of N samples. The samples in a batch are processed independently, in parallel. If training, a batch results in only one update to the model.A batch generally approximates the distribution of the input data better than a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take longer to process and will still result in only one update. For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluation/prediction).

Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.

When using validation_data or validation_split with the fit method of Keras models, evaluation will be run at the end of every epoch.

Within Keras, there is the ability to add callbacks specifically designed to be run at the end of an epoch. Examples of these are learning rate changes and model checkpointing (saving).

Evaluation

Compare final evaluation (MSE) on training set and test set

Further Evaluations

Predicting on brand new data

Saving and Loading a Model
