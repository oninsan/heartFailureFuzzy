import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from category_encoders import BinaryEncoder
from tensorflow.keras.models import load_model # type: ignore
# from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS

model = load_model('heart_problem_refined.pb')

dcc = [
    'Smoking',
    'AlcoholDrinking',
    'Stroke',
    'DiffWalking',
    'Sex',
    'PhysicalActivity',
    'Asthma',
    'KidneyDisease',
    'SkinCancer'
]

# multi-categorical columns
mcc = [
    'AgeCategory',
    'Race',
    'Diabetic',
    'GenHealth'
]

# continuous columns
cc = [
    'BMI',
    'SleepTime',
    'PhysicalHealth',
    'MentalHealth'
]

# combine the columns for dataframe reference
c_combined = dcc+mcc+cc

scaler = MinMaxScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
sample_ref = pd.read_csv('Sample_for_testing.csv')
encoded = pd.DataFrame(encoder.fit_transform(sample_ref[mcc]))


# dou-categorical columns

api = Flask(__name__)
CORS(api)

@api.route('/api/heartfailure/prediction', methods=['POST'])
def predict_hf_proba():
    data = request.get_json()['inputs']
    inputs_ = pd.DataFrame()

    for cols in range(len(c_combined)):
        if c_combined[cols] == 'AgeCategory' or c_combined[cols] == 'Race' or c_combined[cols] == 'Diabetic' or c_combined[cols] == 'GenHealth':
            inputs_.loc[0, c_combined[cols]] = data[cols]
        else:
            inputs_.loc[0, c_combined[cols]] = data[cols]

    scaler.fit(inputs_[cc])
    sc_names = [col+'_scaled' for col in cc]
    n_data = scaler.transform(inputs_[cc])
    scaled_data = pd.DataFrame(n_data, columns=sc_names)
    inputs_ = inputs_.drop(columns=cc, axis=1)
    inputs_ = pd.concat([inputs_, scaled_data], axis=1)

    encoded.columns = encoder.get_feature_names_out(mcc)
    inputs_ = inputs_.drop(mcc, axis=1)
    inputs_ = inputs_.reset_index(drop=True)
    encoded.reset_index(drop=True)
    inputs_ = pd.concat([inputs_, encoded], axis=1)
    inputs_ = inputs_.head(1)

    prediction = model.predict(inputs_)
    print(inputs_)
    return jsonify(round(prediction[0][0]*100, 2))

if __name__ == '__main__':
    api.run(port=5000)