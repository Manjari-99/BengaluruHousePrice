 
# Bengaluru Real Estate Price prediction model


## Abstract

Bengaluru is the fastest growing city in India with throbbing job opportunities and numerous educational centres. People from all over India are moving to the ‘Silicon City’ for a better livelihood. This is exactly where the project comes into play. 
GharDekho.com is a user friendly Bengaluru Real Estate price prediction model. As the name suggests, users may enter the constraints of their notion of a perfect home and estimate a budget. The price is governed by the location, number of BHK, number of bathrooms and of course, the square feet area. Thus, they may get a clear-cut idea about their new residence based on their budget or may start saving for the dream home. GharDekho.com believes in the moto – 
Cause your DREAM HOME IS PRICELESS. Practically speaking, it lets you have a peek into the otherwise finite price for your emotionally priceless home.
The project has been developed in Python programming language with innate use of packages like pandas, numpy and scikit learn. The machine learning model used here is Random Forest Regressor from ensemble because of its high level of accuracy. Concepts of pickle and flask are also used to import our model and connect it with our front end in the work flow. Other programming languages used to build the user interface are HTML, CSS and concepts of Jinja.
The best thing about this project is its utility in the real world and thus, it’s ‘quotient of benefit’ is quite high for the new city dwellers. 


## Introduction

With the increasing inflow of city dwellers in the ‘Silicon City’, Bengaluru has witnessed a significant leap in the count of its city dwellers. This predictive model helps the user to predict the price of a real estate property. The dataset is imported, cleaned, certain constraints are label encoded to find out the proper correlation and trained to get the model. The model is finally tested and used for predictions.
Python version 3 is used as the programming language because of its utility in machine learning coding. The other utility modules used are numpy, for scientific array computations; and pandas, for its efficiency with tabular data. 
Machine learning is the science of making a machine understand the patterns and act like humans by feeding information and without explicitly coding.
Scikit learn is a free software library tool that helps us with machine learning with python. The machine learning model used here is random forest regressor because occasionally it outperforms a decision tree. It is a method of ensemble learning. 
Matplotlib library is used for ease of visualization of data.

## Aim

The aim of this project is to serve its purpose of accurate predictions of updated house prices through a user friendly and aesthetically pleasing interface. The model can be used in a large scale as its practical benefits cannot be overlooked. The soul of this model lies in its performance in predictions. The project shall also be well structured to keep it ready for beneficial modifications in the future.
Technology and Concepts used

## Machine Learning

The idea of making a machine predict the output is closely related to making it learn, which is machine learning. Machine learning is the concept of making a computer learn and act intelligently by feeding information. Hence, there is no need for explicit and repetitive coding. As availability of computational capacity and data has increased, machine learning has become more and more popular with time. 

## Random Forest Regressor

Decision tree classifiers are attractive models if we care about interpretability. Like the name decision tree suggests, we can think of this model as breaking down based on a splitting criterion. However, decision trees are prone to overfitting and error.
Random forests have gained huge popularity in applications of machine learning during the last decade due to their good classification performance, scalability, and ease of use. Intuitively, a random forest can be considered as an ensemble of decision trees. The idea behind ensemble learning is to combine weak learners to build a more robust model, a strong learner, that has a better generalization error and is less susceptible to overfitting. It is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

## Pickle

Python pickle module is used for serializing and de-serializing a Python object structure. Any object in Python can be pickled so that it can be saved on disk. What pickle does is that it “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

## Flask

Flask is a web micro-framework that means it has little to no dependencies on external libraries. There are some pros and cons. Pros mean there are little dependencies to upgrade and to watch security bugs and cons means by introducing plugins we may increase the dependency. We can render request to a web page through flask.  Jinja2 is a template engine

## Methodology

### Creating the Model 
![alt text](/images/dataset.jpg)

## Acknowledgment

My sincere gratitude and thanks towards my project paper guide Sir Bandenawaz Bagwan.
It was only with his backing and support that I could complete the report. He provided me all sorts of help and corrected me if ever seemed to make mistakes. I acknowledge my dearest parents for being such a nice source of encouragement and moral support that helped me tremendously in this aspect. I also declare to the best of my knowledge and belief that the Project Work has not been submitted anywhere else.

## References

•	Hands-On Machine Learning with scikit-learn, keras and tensorflow. – Aurelien Geron
•	Python Machine Learning - Sebastian Raschka
•	Study Materials provided by my instructor.
•	https://scikit-learn.org/stable/

