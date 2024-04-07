from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)


google_api_key = "AIzaSyBmkQYm47AktanebaWL8PI-cOosCPVeuyI"

genai.configure(api_key=google_api_key)

def analyze_instrument_request(sentence):
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if "gemini" in m.name.lower():
                gemini_model_name = m.name
                break
    else:
        raise Exception("Could not find a suitable Gemini model")

    response = genai.generate_text(
        prompt = f"return a list of only the desired instrument names and music genre from this sentence and last desired the music genre: '{sentence}' ",
    )

    item_list = response.result.split(',')
    cleaned_items = [''.join(filter(str.isalpha, item)) for item in item_list]

    return cleaned_items

@app.route('/api/ai-function', methods=['POST'])
def ai_function():
    data = request.json
    result = analyze_instrument_request(data)
    genre = result[len(result)-1]
    result.pop()
    return jsonify({"result": result}, {"genre": genre})

if __name__ == '__main__':
    app.run(debug=True)


