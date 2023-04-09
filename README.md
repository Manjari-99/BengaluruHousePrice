 
Project Report
on
Bengaluru Real Estate Price prediction model
using
Random Forest Regressor.











Abstract

Bengaluru is the fastest growing city in India with throbbing job opportunities and numerous educational centres. People from all over India are moving to the ‘Silicon City’ for a better livelihood. This is exactly where the project comes into play. 
GharDekho.com is a user friendly Bengaluru Real Estate price prediction model. As the name suggests, users may enter the constraints of their notion of a perfect home and estimate a budget. The price is governed by the location, number of BHK, number of bathrooms and of course, the square feet area. Thus, they may get a clear-cut idea about their new residence based on their budget or may start saving for the dream home. GharDekho.com believes in the moto – 
Cause your DREAM HOME IS PRICELESS. Practically speaking, it lets you have a peek into the otherwise finite price for your emotionally priceless home.
The project has been developed in Python programming language with innate use of packages like pandas, numpy and scikit learn. The machine learning model used here is Random Forest Regressor from ensemble because of its high level of accuracy. Concepts of pickle and flask are also used to import our model and connect it with our front end in the work flow. Other programming languages used to build the user interface are HTML, CSS and concepts of Jinja.
The best thing about this project is its utility in the real world and thus, it’s ‘quotient of benefit’ is quite high for the new city dwellers. 










Introduction

With the increasing inflow of city dwellers in the ‘Silicon City’, Bengaluru has witnessed a significant leap in the count of its city dwellers. This predictive model helps the user to predict the price of a real estate property. The dataset is imported, cleaned, certain constraints are label encoded to find out the proper correlation and trained to get the model. The model is finally tested and used for predictions.
Python version 3 is used as the programming language because of its utility in machine learning coding. The other utility modules used are numpy, for scientific array computations; and pandas, for its efficiency with tabular data. 
Machine learning is the science of making a machine understand the patterns and act like humans by feeding information and without explicitly coding.
Scikit learn is a free software library tool that helps us with machine learning with python. The machine learning model used here is random forest regressor because occasionally it outperforms a decision tree. It is a method of ensemble learning. 
Matplotlib library is used for ease of visualization of data.

Aim

The aim of this project is to serve its purpose of accurate predictions of updated house prices through a user friendly and aesthetically pleasing interface. The model can be used in a large scale as its practical benefits cannot be overlooked. The soul of this model lies in its performance in predictions. The project shall also be well structured to keep it ready for beneficial modifications in the future.
Technology and Concepts used

Machine Learning

The idea of making a machine predict the output is closely related to making it learn, which is machine learning. Machine learning is the concept of making a computer learn and act intelligently by feeding information. Hence, there is no need for explicit and repetitive coding. As availability of computational capacity and data has increased, machine learning has become more and more popular with time. 

Random Forest Regressor

Decision tree classifiers are attractive models if we care about interpretability. Like the name decision tree suggests, we can think of this model as breaking down based on a splitting criterion. However, decision trees are prone to overfitting and error.
Random forests have gained huge popularity in applications of machine learning during the last decade due to their good classification performance, scalability, and ease of use. Intuitively, a random forest can be considered as an ensemble of decision trees. The idea behind ensemble learning is to combine weak learners to build a more robust model, a strong learner, that has a better generalization error and is less susceptible to overfitting. It is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

 


Pickle

Python pickle module is used for serializing and de-serializing a Python object structure. Any object in Python can be pickled so that it can be saved on disk. What pickle does is that it “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

Flask

Flask is a web micro-framework that means it has little to no dependencies on external libraries. There are some pros and cons. Pros mean there are little dependencies to upgrade and to watch security bugs and cons means by introducing plugins we may increase the dependency. We can render request to a web page through flask.  Jinja2 is a template engine

Methodology

Creating the model

Let’s first look at the dataset
Bengaluru_House_Data.csv
 
We next import the dataset
We make a python file called House_Price_Predictor.csv in Google Colab. We import pandas and numpy modules. We also import matplotlib and seaborn for data visualization. We then create dataframe df from dataset.
 

The next step is to clean the data
  

It involves steps like, removing null values and duplicate rows.
 
We see that society column has a lot of null values.
 
 
We plot the society with respect to price. We see that there is a lot of variation. For the same society there are instances of varied price. Also, there are more than 5000 null valued tuples. Hence, we drop this column.
 


The next column that has a lot of null values is balcony
 




 
We see that balcony column has considerably low correlation with price, all the more, the graph shows low variation of price with balcony number. Hence, it is wise to remove this column.
 
 

The next column with null values is bath. However, bath has moderate correlation, hence, removing the constraint shall be avoided. 73 is a small number with respect to the dataframe shape 13320. The next column considered is size. Size has only 16 null valued rows. Hence, we drop the null values rows.
 
Hence our dataframe is free from null values.
We next drop the duplicate rows.
 

Data cleaning is complete.






We next remove unnecessary columns
 
We start with area type
 
 

There is little to no variation in the plot. We label encode the area type values to find its correlation with price. Since the correlation is very low, we drop this column.
 

We next consider availability.
 

We drop the column since the correlation is very low.
We next consider location column.
 

We see that there are a huge number of outliers, hence we remove it. We put all the locations with 10 and less value count in ‘Others’ value.
 
We next label encode the values so that it can be trained in the model.
 
We next move to the total square feet column. We see that this column contains non numerical values as well as ranges with a ‘-‘ character. Hence, we format it.
 
 
We next look at bathrooms column.
 

We remove outliers.
 
We next go to size column and rename it to BHK by extracting the numerical value. 
 
 
We next remove outliers from the overall dataset
 
The data is now ready for Training 
 
We use train_test_split from sklearn model_selection with test size 0.3.
  
 
We now test the data
 
We get an overall 0.62 r2_score.

We now convert the models into a pickle file.
 
Model creation is complete.











Creating the Interface
 Click on the gif below to have a look at the user web page
 
The above web page has been created through the following steps.
We first import pandas and flask and dump pickle files.
We create a file called app.py that connects the html file with the imported pickle model. This serves as the backend.
import pickle

import numpy as np
from flask import Flask, render_template, request
import pickle

# a flask object app
app = Flask(__name__)

# importing pickle files
label_encoder = pickle.load(open('models/loc_encoder.pkl', 'rb'))
random_forest_reg = pickle.load(open('models/rfr.pkl', 'rb'))

We next copy all the unique locations from dataset df and past it in the code by creating a list of locations. Next, we sort the list.
# creating location list, from the dataset
loc = ['Electronic City Phase II', 'Chikka Tirupathi', 'Uttarahalli',
       'Lingadheeranahalli', 'Kothanur', 'Whitefield', 'Old Airport Road',
       'Rajaji Nagar', 'Marathahalli', 'Others', '7th Phase JP Nagar',
       'Gottigere', 'Sarjapur', 'Mysore Road', 'Bisuvanahalli',
       'Raja Rajeshwari Nagar', 'Kengeri', 'Binny Pete', 'Thanisandra',
       'Bellandur', 'Electronic City', 'Ramagondanahalli', 'Yelahanka',
       'Hebbal', 'Kasturi Nagar', 'Kanakpura Road',
       'Electronics City Phase 1', 'Kundalahalli', 'Chikkalasandra',
       'Murugeshpalya', 'Sarjapur  Road', 'HSR Layout', 'Doddathoguru',
       'KR Puram', 'Bhoganhalli', 'Lakshminarayana Pura', 'Begur Road',
       'Devanahalli', 'Varthur', 'Bommanahalli', 'Gunjur', 'Hegde Nagar',
       'Haralur Road', 'Hennur Road', 'Kothannur', 'Kalena Agrahara',
       'Kaval Byrasandra', 'ISRO Layout', 'Garudachar Palya', 'EPIP Zone',
       'Dasanapura', 'Kasavanhalli', 'Sanjay nagar', 'Domlur',
       'Sarjapura - Attibele Road', 'Yeshwanthpur', 'Chandapura',
       'Nagarbhavi', 'Ramamurthy Nagar', 'Malleshwaram', 'Akshaya Nagar',
       'Shampura', 'Kadugodi', 'LB Shastri Nagar', 'Hormavu',
       'Vishwapriya Layout', 'Kudlu Gate', '8th Phase JP Nagar',
       'Bommasandra Industrial Area', 'Anandapura',
       'Vishveshwarya Layout', 'Kengeri Satellite Town', 'Kannamangala',
       ' Devarachikkanahalli', 'Hulimavu', 'Mahalakshmi Layout',
       'Hosa Road', 'Attibele', 'CV Raman Nagar', 'Kumaraswami Layout',
       'Nagavara', 'Hebbal Kempapura', 'Vijayanagar',
       'Pattandur Agrahara', 'Nagasandra', 'Kogilu', 'Panathur',
       'Padmanabhanagar', '1st Block Jayanagar', 'Kammasandra',
       'Dasarahalli', 'Magadi Road', 'Koramangala', 'Dommasandra',
       'Budigere', 'Kalyan nagar', 'OMBR Layout', 'Horamavu Agara',
       'Ambedkar Nagar', 'Talaghattapura', 'Balagere', 'Jigani',
       'Gollarapalya Hosahalli', 'Old Madras Road', 'Kaggadasapura',
       '9th Phase JP Nagar', 'Jakkur', 'TC Palaya', 'Giri Nagar',
       'Singasandra', 'AECS Layout', 'Mallasandra', 'Begur', 'JP Nagar',
       'Malleshpalya', 'Munnekollal', 'Kaggalipura', '6th Phase JP Nagar',
       'Ulsoor', 'Thigalarapalya', 'Somasundara Palya',
       'Basaveshwara Nagar', 'Bommasandra', 'Ardendale', 'Harlur',
       'Kodihalli', 'Narayanapura', 'Bannerghatta Road', 'Hennur',
       '5th Phase JP Nagar', 'Kodigehaali', 'Billekahalli', 'Jalahalli',
       'Mahadevpura', 'Anekal', 'Sompura', 'Dodda Nekkundi', 'Hosur Road',
       'Battarahalli', 'Sultan Palaya', 'Ambalipura', 'Hoodi',
       'Brookefield', 'Yelenahalli', 'Vittasandra',
       '2nd Stage Nagarbhavi', 'Vidyaranyapura', 'Amruthahalli',
       'Kodigehalli', 'Subramanyapura', 'Basavangudi', 'Kenchenahalli',
       'Banjara Layout', 'Kereguddadahalli', 'Kambipura',
       'Banashankari Stage III', 'Sector 7 HSR Layout', 'Rajiv Nagar',
       'Arekere', 'Mico Layout', 'Kammanahalli', 'Banashankari',
       'Chikkabanavar', 'HRBR Layout', 'Nehru Nagar', 'Kanakapura',
       'Konanakunte', 'Margondanahalli', 'R.T. Nagar', 'Tumkur Road',
       'Vasanthapura', 'GM Palaya', 'Jalahalli East', 'Hosakerehalli',
       'Indira Nagar', 'Kodichikkanahalli', 'Varthur Road', 'Anjanapura',
       'Abbigere', 'Tindlu', 'Gubbalala', 'Parappana Agrahara',
       'Cunningham Road', 'Kudlu', 'Banashankari Stage VI', 'Cox Town',
       'Kathriguppe', 'HBR Layout', 'Yelahanka New Town',
       'Sahakara Nagar', 'Rachenahalli', 'Yelachenahalli',
       'Green Glen Layout', 'Thubarahalli', 'Horamavu Banaswadi',
       '1st Phase JP Nagar', 'NGR Layout', 'Seegehalli', 'BEML Layout',
       'NRI Layout', 'ITPL', 'Babusapalaya', 'Iblur Village',
       'Ananth Nagar', 'Channasandra', 'Choodasandra', 'Kaikondrahalli',
       'Neeladri Nagar', 'Frazer Town', 'Cooke Town', 'Doddakallasandra',
       'Chamrajpet', 'Rayasandra', '5th Block Hbr Layout', 'Pai Layout',
       'Banashankari Stage V', 'Sonnenahalli', 'Benson Town',
       '2nd Phase Judicial Layout', 'Poorna Pragna Layout',
       'Judicial Layout', 'Banashankari Stage II', 'Karuna Nagar',
       'Bannerghatta', 'Marsur', 'Bommenahalli', 'Laggere',
       'Prithvi Layout', 'Banaswadi', 'Sector 2 HSR Layout',
       'Shivaji Nagar', 'Badavala Nagar', 'Nagavarapalya', 'BTM Layout',
       'BTM 2nd Stage', 'Hoskote', 'Doddaballapur', 'Sarakki Nagar',
       'Thyagaraja Nagar', 'Bharathi Nagar', 'HAL 2nd Stage',
       'Kadubeesanahalli']
# we alphabetically sort the list
loc = sorted(loc)


We next create flask route and create two functions, index to send the request to form.html and predict to fetch the values and return the predicted price to form.html for user display.
# setting flask route to the html page, and sending the list of locations of display

@app.route('/')
def index():
    return render_template('form.html', loc=loc)


# after fetching the values from the user we will use our model to find out the price
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
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GharDekho.com</title>
   <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
   <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" integrity="sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV" crossorigin="anonymous"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@450&display=swap" rel="stylesheet">
</head>


We import all the links of bootsrap and fonts.

<body style="background: #70e1f5;  /* fallback for old browsers */
background: -webkit-linear-gradient(to right, #ffd194, #70e1f5);  /* Chrome 10-25, Safari 5.1-6 */
background: linear-gradient(to right, #ffd194, #70e1f5); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
; font-family: 'Quicksand';">

We use gradient background style

   <nav style="background-color: #58B19F;">
      <div>
            <a href="#" class="navbar-brand mt-1" style="color:#dfe6e9;">
                 <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" fill="currentColor" color="#dfe6e9" class="bi bi-house-door-fill" viewBox="0 0 16 16">
                            <path d="M6.5 14.5v-3.505c0-.245.25-.495.5-.495h2c.25 0 .5.25.5.5v3.5a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.146-.354L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293L8.354 1.146a.5.5 0 0 0-.708 0l-6 6A.5.5 0 0 0 1.5 7.5v7a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5z"/>
                    </svg>
                GharDekho.com</a>
            <a style="float:right;color:#dfe6e9;" class="navbar-brand mt-1">Cause your DREAM HOME IS PRICELESS</a>
      </div>
   </nav>

We create a proper navigation bar.

    <div class="row">
        <div class="col-md-5" style="margin-top:50px;">
            <div align="center">
                <img align="center" src="https://lh3.googleusercontent.com/proxy/8O_zPiBzQxeimqebLiiNygWZgf_sTYnW5xWixBc52eko6Dugqpt-KAHmQcSfKFquyZbNU2L3EL7F_9lb0fKTKLgXy1Xgncc" style="height: 500px; width: 500px; border-radius: 250px; margin-left:30px;">

            </div>
        </div>
      <div class="col-md-6" style=" margin-left:50px; margin-top: 50px">

We next create a card which contains a form for users to enter the constraint.

         <div class="card" style="background-color:#F8EFBA">
            <div class="card-header" align="center" style="background-color:#58B19F;color:#dfe6e9;">
                     <svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" fill="currentColor" color="#dfe6e9" class="bi bi-house-door-fill" viewBox="0 0 16 16">
                            <path d="M6.5 14.5v-3.505c0-.245.25-.495.5-.495h2c.25 0 .5.25.5.5v3.5a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.146-.354L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293L8.354 1.146a.5.5 0 0 0-.708 0l-6 6A.5.5 0 0 0 1.5 7.5v7a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5z"/>
                    </svg>
                        <h2>Welcome To GharDekho</h2>
                </div>
            <div class="card-body" style="color:#3c6382;">
               <form action="/predict" method="post" style="color:#0a3d62" >
                    <div class="row">
                        <div class="form-group col-md-12" align="center">
                        <label><h5>Select Location</h5></label><br>
                        <select name="location" class="form-control">

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
Ready to use

1.	We check with valid inputs. (click on the gif below)
 














2.	We check for empty inputs and error message. (click on the gif below)
 


3.	We check for invalid inputs. (click on the gif below)
 

	
Conclusion

We have used Random Forest Regressor instead of Linear Regression to predict the price because of its accuracy and performance. However, the pickle file containing this model is considerably large, hence the project is difficult and slow to deploy. In spite of its shortcomings, this project is extremely useful. This project has used different machine learning technique and different models for data training attempting to predict the house price correctly. The project serves the purpose in its full potential.
The project has been uploaded successfully in my GitHub profile.
Link – https://github.com/Manjari-99/GharDekho.git




Acknowledgment

My sincere gratitude and thanks towards my project paper guide Sir Bandenawaz Bagwan.
It was only with his backing and support that I could complete the report. He provided me all sorts of help and corrected me if ever seemed to make mistakes. I acknowledge my dearest parents for being such a nice source of encouragement and moral support that helped me tremendously in this aspect. I also declare to the best of my knowledge and belief that the Project Work has not been submitted anywhere else.

References

•	Hands-On Machine Learning with scikit-learn, keras and tensorflow. – Aurelien Geron
•	Python Machine Learning - Sebastian Raschka
•	Study Materials provided by my instructor.
•	https://scikit-learn.org/stable/

