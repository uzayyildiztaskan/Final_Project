import google.generativeai as genai

google_api_key = "SECRET_KEY"

genai.configure(api_key=google_api_key)

def analyze_instrument_request(sentence):
    # Identify a suitable Gemini model using Google's API
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if "gemini" in m.name.lower():  # Look for names containing "gemini"
                gemini_model_name = m.name
                break  # Found our model, stop looking
    else:  # Handles the case where no suitable Gemini model is found
        raise Exception("Could not find a suitable Gemini model")

    # Generate content using the Gemini model
    response = genai.generate_text(
        prompt=f"Analyze this sentence to determine which musical instruments are mentioned and whether they are "
                  f"desired or not: '{sentence}'. List the instruments and their desired status.",
    )

    return response

# Example sentence
sentence = ("gitar ve bas olmasın, piyano olsun")

result = analyze_instrument_request(sentence)

print("Analiz Sonucu:", result)