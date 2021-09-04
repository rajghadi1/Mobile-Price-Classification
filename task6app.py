from flask import Flask, request, render_template
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('mobile.pkl','rb'))

@app.route('/')
def home():
    return render_template('task6.html')

@app.route('/predict',methods=['POST'])  
def predict():
   
    input_features =[float(x) for x in request.form.values()]
    features_value =[np.array(input_features)]

    output =model.predict(features_value)
    if output ==0:
        return render_template('task6.html',predict_text='Very Low Price Category (0) ')
    elif output ==1:
        return render_template('task6.html', predict_text='Low Price Category (1) ')
    elif output ==2:
        return render_template('task6.html', predict_text='High Price Category (2) ')
    elif output ==3:
        return render_template('task6.html', predict_text='Very High Price Category (3) ')


if __name__ == '__main__':
    app.run(debug=True)