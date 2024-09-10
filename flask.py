from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('text_classifier_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    # Preprocess and encode the text
    encoded_text = encode(data)
    prediction = model.predict(encoded_text)
    return jsonify({'article_type': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)