from flask import Flask, render_template, request
import joblib

app = Flask(__name__, static_folder='static')

model = joblib.load('model.pkl')

@app.route('/')
def front():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST']) 
def submit():
    
    qoe = float(request.form['qoe'])
    ae = float(request.form['ae'])
    qof = float(request.form['qof'])
    pu = float(request.form['pu'])
    ni = float(request.form['ni'])
    ci = float(request.form['ci'])
    pa = float(request.form['pa']) 
    
    input_data = [[qoe, ae, qof, pu, ni, ci, pa]]
    prediction = model.predict(input_data)

    return render_template('submit.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
