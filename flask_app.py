from flask import Flask, render_template, request, jsonify
import joblib
from sms_prep import preprocess_text


app = Flask(__name__)

# Load the trained model
model = joblib.load('sentiment_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract the text from the request
    data = request.get_json(force=True)
    message = data['message']
    
    try:
        # Preprocess the message and predict its sentiment
        prediction = classify_new_chat(message, model)
        # Format the response as a JSON object
        response = {
            'message': message,
            'prediction': prediction
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def classify_new_chat(chat, model):
    processed_chat = preprocess_text(chat)
    sentiment = model.predict([processed_chat])[0]
    return sentiment

if __name__ == '__main__':
    app.run(debug=True)
