import google.generativeai as genai

google_api_key = "SECRET_KEY"

genai.configure(api_key=google_api_key)


def analyze_instrument_request(sentence):
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if "gemini" in m.name.lower():  # Look for names containing "gemini"
                gemini_model_name = m.name
                break
    else:
        raise Exception("Could not find a suitable Gemini model")

    response = genai.generate_text(
       
        prompt = f"return a list of only the desired instrument names from this sentence: '{sentence}' ",
    ) 

    instruments_status = parse_instrument_response(response.result, ["gitar", "bas", "piyano"])
    return instruments_status


def parse_instrument_response(response, instruments):
    # Yanıt metnindeki enstrüman durumlarını ayrıştırma
    status_list = []
    print(response)
    # Tüm enstrümanlar için döngü
    for instrument in instruments:
        # Enstrümanın yanıtta geçip geçmediğini ve durumunu kontrol et
        if instrument in response.lower():
            if "not desired" in response.lower() or "not mentioned" in response.lower():
                status_list.append(0)  # Enstrüman istenmiyor
            else:
                status_list.append(1)  # Enstrüman isteniyor
        else:
            status_list.append(0)  # Enstrüman metinde geçmiyorsa varsayılan olarak istenmiyor kabul edilir
    
    return status_list


musical_instruments = ["gitar", "bas", "piyano"]

sentence = input("Enter a sentence to analyze: ")

result = analyze_instrument_request(sentence)

#print(result)