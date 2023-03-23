# 🏡🏠 Airbnb Home Prices Prediction 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction

With the aid of __machine learning__ and __data science__, it is possible to predict the prices of houses respectively. There are features such as __longitude__ and __latitude__ that help determine the prices along with other features such as neighborhood and the demand for the area. 

<img src = "https://github.com/suhasmaddali/GIF-files/blob/main/homegif.gif" />

## Metrics

The problem that we are trying to consider is the price prediction which is a __regression__ problem. Hence, we should choose the metrics that are important for the regression problem. Below are the metrics that were considered for this problem.

* [__Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
* [__Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Machine Learning Analysis 

Taking into consideration these features, __machine learning predictions__ could be built. It is important to perform __Exploratory Data Analysis__ to predict the demand for these houses as well. There could be __null__ values that must also be addressed before giving those values to the models. Steps must be taken to ensure that models don't __overfit__ the data or __underfit__ it. 

## Exploratory Data Analysis (EDA)

* It would be seen based on the results from the KDE plot that there are a few outliers in the prices of houses. Therefore, those prices had to be removed to reduce the mean squared error or mean absolute error of the models. __Removing outliers__ is a good idea as not all the models are robust to outliers. 
* A large portion of the users from Airbnb was willing to rent the entire apartment instead of a __private room__ or a __shared room__ respectively.
* A large number of houses taken into consideration were from __Manhattan__ followed by __Brooklyn__. 
* There were very few houses from __Staten Island__ compared to the other cities. 

## Visualizations

In this section, we will **delve** into the data to uncover valuable insights and visualize the relationships between different features. By exploring the data in depth and gaining a visual understanding of its structure, we can better understand the underlying patterns and make more accurate predictions. With this approach, we can unlock the full potential of our data and make informed decisions based on the insights we discover.

There are features such as **name** of the place along with other features such as **neighbourhood** and **room_type** that could be used to determine the overall price of the rooms. Total number of reviews can give us a good representation about the popularity of the place as compared to other rooms in the vicinity. 

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Data%20Image.png"/>

Decision trees are valued for their **interpretability**, as they allow us to understand the importance of different features in predicting outcomes. After selecting the **best depth** for the tree, the resulting model indicates that the neighbourhood of Woodside was the most influential factor in determining housing prices. This was followed by the longitude and neighbourhood of Midland Beach features, which also had a significant impact on prices.

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Data%20Description.png"/>

The data below shows the total number of missing values present. It could be seen that there are missing values in the data for features such as 'last_review' and 'reviews_per_month' features. These features impact the performance of models if not treated before training them. Therefore, steps can be taken to ensure that these features are either imputed or removed. 

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Dataset%20Information.png"/>

**Missingno** plots give a good understanding about the total missing values present in the data in the form of white strips (depends on color mapping). There are features that contain missing information such as 'last_review' and 'reviews_per_month'. Therefore, steps could be taken to ensure that these values are either removed or imputed. 

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Missingno%20plot.png"/>

**KDE Plots** give a good idea about the distribution of values present in the data. In our case, we take a look at the distribution of price feature and it's density across different regions. A large amount of prices are concentrated in the region between **100 - 500** respectively. There can be some outliers as well as depicted in the diagram. 

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Prices%20distribution.png"/>

The **majority** of the listings in the dataset comprise entire homes or apartments for hosting, while the number of shared rooms available is comparatively low. Private rooms are also available, but they constitute a smaller proportion of the listings in the dataset.

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Room%20type%20count.png"/>

This plot shows the **latitude** and **longitude** information about various locations used in the dataset. The data is taken from varions regions of New York. There are regions such as Brooklyn, Manhattan, Queens and others. More add could be added if the ML models are able to make predictions well on this dataset. 

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Location%20plot.png"/>

For most properties, the monthly number of reviews is around **100**, with only a few houses having significantly more. This distribution is illustrated in the accompanying plot.

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Reviews%20per%20month%20plot.png"/>

The plot displays the **distribution** of housing prices across different regions in New York, with **Manhattan** having the highest concentration of houses, followed by **Brooklyn**. Conversely, there are relatively few houses in **Staten Island**. As a result, the model's predictive accuracy for houses in Staten Island may be lower compared to those in Manhattan due to the smaller sample size. 

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Neighborhood%20plots.png"/>

Upon examining the overall distribution of housing prices across different regions, it becomes apparent that there are **outliers** in each category. These outliers have the potential to skew the results of **machine learning models** by introducing unexpected values. Therefore, it is recommended to remove these outliers from the dataset to improve the accuracy and reliability of the model's predictions.

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Neighborhood%20prices%20plot.png"/>

To optimize the performance of machine learning models, **hyperparameter tuning** is a crucial step. The accompanying plot displays the results of a decision tree regressor model trained using different values for the maximum depth hyperparameter. The graph illustrates the corresponding cross-validation error for each depth value tested. Notably, the model performs best with a maximum depth of 4, which results in the lowest cross-validation error.

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/Decision%20Tree%20Hyperparameter%20tuning%20(Max%20depth).png"/>

Decision trees are renowned for their **interpretability**, allowing us to gain valuable insights into the factors affecting the target variable. After optimizing the **tree depth** and analyzing the tree structure, we observed that the 'neighborhood_Woodside' feature played a crucial role in determining house prices. Subsequently, the 'longitude' and 'neighborhood_Midland_Beach' features also emerged as significant factors influencing **property** values.

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/DT%20Feature%20Importance.png"/>

This plot shows the performance of the **decision tree regressor** and computes the **mean squared error** for various values of hyperparameter "Min samples leaves". As the number of Min samples leaves increases, there is also an increase in mean squared error which is undesirable. Steps can be taken to choose the most optimum value for minimum number of samples to reduce the overall cross-validation error. 

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/DT%20Hyperparameter%20tuning%20(Min%20samples%20leaves).png"/> 

Having recognized the importance of the **minimum number of samples** as a hyperparameter, we diligently conducted hyperparameter tuning by testing a broader range of values for this parameter. We then plotted the results to visualize the overall performance and gain a clearer understanding. Based on the analysis, setting the minimum number of samples to 200 yielded the lowest mean squared error (cross-validation error), indicating optimal performance for this particular hyperparameter.

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/DT%20Hyperparameter%20tuning%20(Min%20samples%20split).png"/>

Performing the same set of actions on **Random Forest Regressor**, we tend to see that with the increase in the number of estimators (trees), there is an improvement in the performance of the model on the cross-validation data. However, this can be computationally intensive as it can only be run in a sequence. 

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/RF%20Hyperparameter%20tuning%20(no.%20of%20estimator%20values).png"/>

As the number of trees in the **Random Forest** increases, the mean squared error decreases. However, the decrease in the mean squared error with respect to the number of trees is minimal. Therefore, the best number of trees with relatively high performance is **50**.

<img src = "https://github.com/suhasmaddali/Airbnb-Home-Prices-Prediction/blob/main/images/RF%20Hyperparameter%20tuning%20(no.%20of%20variables).png"/>

## Machine Learning Models

There are many libraries from __sklearn__ which we might be using for our machine learning predictions. Below are the best performing __machine learning__ and __deep learning models__ that was used in the prediction of house prices.

* [__Decision Trees__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
* [__Random Forests__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [__Neural Networks__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

## Outcomes 

* Hosts who are using __Airbnb application__ could benefit by using these algorithms in setting up the price so that it ensures that there is maximum profit generated by them.
* This would also lead to growth and an increase in revenue for __Airbnb__. 

## Future Scope

* __Additional features__ such as __weather conditions__ and the __crime rate__ present in the localities also help determine the prices of houses which could be added to improve the __performance__ of machine learning models. 
* The output from the __best models__ should be integrated in __AirBnb application (app)__ so that hosts can determine the best price and this leads to their __higher engagement__ in the application. 
* The __best machine learning model__ could also be deployed in the cloud such as in __Amazon Web Services (AWS)__ so that we can host our product on websites so that it is accessible to users across different regions. 

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git which could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in the "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link to the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(18).png" width = "600" />

5. The link to the repository can be found when you click on "Code" (Green button) and then, there would be an HTML link just below. Therefore, the command to download a particular repository should be "Git clone HTML" where the HTML is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(20).png" width = "600" />

8. Later, open the Jupyter notebook by writing "Jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 

