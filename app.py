from flask import Flask, render_template, request
from house_price_prediction_delhi import *  # Import your existing Python code

app = Flask(__name__)

Delhi_dataset.drop(['Price'], axis=1, inplace=True)
input_variables = list(Delhi_dataset.columns)

location_option_list = list(le.classes_)
# print(location_option_list)
# for op in location_option_list:
#     print(op)

# print(input_variables)

# @app.route('/', methods=['POST','GET'])
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', input_variables=input_variables, location_option_list = location_option_list)

input_form = []

@app.route('/result/', methods=['POST'])
def result():
    # Collect user input from the form
    user_input = pd.DataFrame([request.form[variable] for variable in input_variables],index=input_variables)
    input_form = user_input.copy()
    # user_input.columns = input_variables.columns
    for var in input_variables:
        if (var=='Location'):
            val1 = le.transform([user_input.loc[var,0]])
            user_input.loc[var,0] = val1
        elif ( (user_input.loc[var,0]=='Yes') | (user_input.loc[var,0]=='No')):
            if (user_input.loc[var,0]=='Yes'):
                user_input.loc[var,0]=1
            else :
                user_input.loc[var,0]=0
        user_input_final = user_input.reset_index()[0]
        # print(user_input.loc[var,0])
        # print(user_input.reset_index()[0])
    np_arr = np.array(user_input_final)
    # print(user_input.reset_index()[0])
    np_arr = np_arr.reshape(1,np_arr.shape[0])
    # print(np_arr.shape)
    # Call the prediction function
    predicted_price_lr = models[0].predict(np_arr)
    rounded_price_lr = np.round(predicted_price_lr, 2)
    predicted_price_knr = models[1].predict(np_arr)
    rounded_price_knr = np.round(predicted_price_knr, 2)
    predicted_price_dtr = models[2].predict(np_arr)
    rounded_price_dtr = np.round(predicted_price_dtr, 2)
    predicted_price_rfr = models[3].predict(np_arr)
    rounded_price_rfr = np.round(predicted_price_rfr, 2)
    predicted_price_xgbr = models[4].predict(np_arr)
    rounded_price_xgbr = np.round(predicted_price_xgbr, 2)


    return render_template('result.html',input_form = input_form,rounded_price_lr = rounded_price_lr,rounded_price_knr = rounded_price_knr, \
                           rounded_price_dtr = rounded_price_dtr, rounded_price_rfr = rounded_price_rfr, \
                           rounded_price_xgbr = rounded_price_xgbr)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
