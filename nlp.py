import google.generativeai as genai
genai.configure(api_key="SECRET_KEY")

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

model = genai.GenerativeModel('gemini-pro')


response = model.generate_content(input("You: "))
print(response.text)









