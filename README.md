 
# Bengaluru Real Estate Price prediction model

 Click on the gif below to have a look at the user web page
![](/images/website_video.gif)

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
Let’s first look at the dataset
Bengaluru_House_Data.csv
![](/images/dataset.jpg)

We next import the dataset
We make a python file called House_Price_Predictor.csv in Google Colab. We import pandas and numpy modules. We also import matplotlib and seaborn for data visualization. We then create dataframe df from dataset.
The next step is to clean the data
It involves steps like, removing null values and duplicate rows.
We see that society column has a lot of null values.
We plot the society with respect to price. We see that there is a lot of variation. For the same society there are instances of varied price. Also, there are more than 5000 null valued tuples. Hence, we drop this column.
![](/images/society.png)

The next column that has a lot of null values is balcony.
We see that balcony column has considerably low correlation with price, all the more, the graph shows low variation of price with balcony number. Hence, it is wise to remove this column.
![](/images/balcony.png)

The next column with null values is bath. However, bath has moderate correlation, hence, removing the constraint shall be avoided. 73 is a small number with respect to the dataframe shape 13320. The next column considered is size. Size has only 16 null valued rows. Hence, we drop the null values rows.
We next drop the duplicate rows.
Data cleaning is complete.


We next remove unnecessary columns
We start with area type.
There is little to no variation in the plot. We label encode the area type values to find its correlation with price. Since the correlation is very low, we drop this column.
![](/images/areatype.png)

We next consider location column.
We see that there are a huge number of outliers, hence we remove it. We put all the locations with 10 and less value count in ‘Others’ value.
![](/images/location.png)

We next label encode the values so that it can be trained in the model.
We next move to the total square feet column. We see that this column contains non numerical values as well as ranges with a ‘-‘ character. Hence, we format it.
![](/images/formatting.png)

We next go to size column and rename it to BHK by extracting the numerical value. 

### Creating the Interface
We first import pandas and flask and dump pickle files.
We create a file called app.py that connects the html file with the imported pickle model. This serves as the backend.

app = Flask(__name__)
label_encoder = pickle.load(open('models/loc_encoder.pkl', 'rb'))
random_forest_reg = pickle.load(open('models/rfr.pkl', 'rb'))

We next copy all the unique locations from dataset df and past it in the code by creating a list of locations. Next, we sort the list.
We next create flask route and create two functions, index to send the request to form.html and predict to fetch the values and return the predicted price to form.html for user display.

@app.route('/')
def index():
    return render_template('form.html', loc=loc)
@app.route('/predict', methods=['post'])
def predict():
    location = request.form.get('location')
    total_sqft = request.form.get('total_sqft')
    bath = request.form.get('bath')
    bhk = request.form.get('bhk')
    location = label_encoder.transform(np.array([location]))
    # in case the user forgets to type in all the values

Case handling empty inputs

    if (location == "" or total_sqft == "" or bhk == "" or bath == ""):
        return render_template('form.html', response=-1, loc=loc)

Case handling improper inputs

    elif(int(bhk)<=0 or int(total_sqft)<=0 or int(bath)<=0):
       return render_template('form.html', response=-2,loc=loc)

For correct predictions we predict with the random forest regressor that we had made earlier and imported through pickle.

    else:
        X = np.array([location, total_sqft, bath, bhk]).reshape(1, 4)
        y = 0
        y = random_forest_reg.predict(X)
        y = y * 100000
        answer = "Estimated Price: Rs " + str(round(float(y),2))
        return render_template('form.html', response=answer, loc=loc)


if __name__ == "__main__":
    app.run(debug=True)
We now create form.html
To fetch location we use Jinja

                            {% for i in loc %}
                                <option value="{{i}}">{{i}}</option>
                            {% endfor %}
                        </select><br>
                        </div>

                    </div>
                     <div class="row">
                         <div class="form-group col-md-4" align="center">
                         <label><h5>BHK</h5></label><br>
                        <input type="number" class="form-control" name="bhk" placeholder="e.g.3">

                        </div>
                        <div class="form-group col-md-4" align="center">
                            <label><h5>Square Feet Area</h5></label><br>
                        <input type="number" class="form-control" name="total_sqft" placeholder="e.g.1200">
                        </div>
                        <div class="form-group col-md-4" align="center">
                         <label><h5>Number of Bathrooms</h5></label><br>
                        <input type="number" class="form-control" name="bath" placeholder="e.g.2">

                    </div>
                    </div>
                        <div align="center">
                            <input type="submit" class="btn btn-large btn-outline-info" value="Check House Price"><br><br>

                        </div>
                </form>
                    <div align="center" >
                         {% if response %}
                            {% if response == -1 %}
                              <h5 class="btn btn-danger btn-large">Enter All The Values</h5>
                            {% else %}
                                 {% if response ==-2 %}
                                    <h5 class="btn btn-danger btn-large">Enter Proper Values</h5>
                                {% else %}
                                     <h3 class="btn btn-success btn-large">{{ (response) }}</h3>
                                {% endif %}
                        {% endif %}
                        {% endif %}
                    </div>

            </div>
         </div>
      </div>
   </div><div align="center" class="mt-5" style="color:white;">@madeby Manjari Nandi Majumdar</div>
</body></html>


## Demo

 Click on the gif below.
![](/images/input1.gif)

 Click on the gif below.
![](/images/input2.gif)


## References

*•	Hands-On Machine Learning with scikit-learn, keras and tensorflow. – Aurelien Geron
*•	Python Machine Learning - Sebastian Raschka
*•	Study Materials provided by my instructor.
*•	https://scikit-learn.org/stable/

