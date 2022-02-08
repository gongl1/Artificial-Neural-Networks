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


![2-Logo](Images/ANN-house.png)

EDA: isnull, sns.displot(df['price']), df.corr()['price'].sort_values(), sns.scatterplot(x='price',y='sqft_living',data=df), sns.boxplot(x='bedrooms',y='price',data=df), sns.scatterplot(x='long',y='lat',data=df,hue='price')

Let us only use bottom 99% of data based on price to cut off outliers: non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]

Working with Feature Data - Feature Engineering to remove features that are not important

Feature Engineering from Date: df['date'] = pd.to_datetime(df['date']), df['month'] = df['date'].apply(lambda date:date.month)

Feature Engineering needs to consider if should convert a number-like category column to dummies OR keep its continuous number values makes more sense!

Scaling and Train Test Split

Scaling

Creating a Model: Because we have 19 incoming features, 19 nerons in each layer

Training the Model: model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=400)

validation_data=(X_test,y_test) will not affect weights/bias of the network

Keras will only use training data to update weight/bias and see how it is doing on training and validation data!

Smaller batch size, longer training to take, less likely to overfit your data because you are not passing in your entire training. Instead, focusing smaller batches.

Shows losses on training set and validation data: Take the mse of predictions against true values and try to minimize that through optimizer

Predicting on Brand New Data: mean_absolute_error(y_test,predictions), plt.scatter(y_test,predictions), errors = y_test.reshape(6480, 1) - predictions

Predicting on a brand new house: predicted-287402.97, actual-221900.0000


![3-Logo](Images/ANN-cancer.png)

EDA: sns.countplot(x='benign_0__mal_1',data=df), sns.heatmap(df.corr()), df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')

Train Test Split

Scaling Data

Creating the Model: # For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
Training the Model: 

Example One: Choosing too many epochs and overfitting! Overfitting: Loss still going down, but orange is going up for epochs.

Example Two: Early Stopping. We obviously trained too much! Let's use early stopping to track the val_loss and stop training once it begins increasing too much!

Example Three: Adding in DropOut Layers. model.add(Dropout(0.5)), 0.5 percent neurons are going to be turned off for each batch. Better result, blue and orange are flattening out at same epoch!

Model Evaluation: predictions = model.predict_classes(X_test). print(classification_report(y_test,predictions)). print(confusion_matrix(y_test,predictions)).

