import pickle

import numpy as np
from flask import Flask, render_template, request
import pickle

# a flask object app
app = Flask(__name__)

# importing pickle files
label_encoder = pickle.load(open('models/loc_encoder.pkl', 'rb'))
random_forest_reg = pickle.load(open('models/rfr.pkl', 'rb'))

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
    if (location == "" or total_sqft == "" or bhk == "" or bath == ""):
        return render_template('form.html', response=-1, loc=loc)
    elif(int(bhk)<=0 or int(total_sqft)<=0 or int(bath)<=0):
       return render_template('form.html', response=-2,loc=loc)
    else:
        X = np.array([location, total_sqft, bath, bhk]).reshape(1, 4)
        y = 0
        y = random_forest_reg.predict(X)
        y = y * 100000
        answer = "Estimated Price: Rs " + str(round(float(y),2))
        return render_template('form.html', response=answer, loc=loc)


if __name__ == "__main__":
    app.run(debug=True)
