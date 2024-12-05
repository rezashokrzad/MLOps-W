from pathlib import Path
import pickle
import re 

__version__ = "1.0.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent
print(BASE_DIR)


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

# Prediction function
def predict_pipeline(input_text):
    # Pass input directly to the pipeline
    if isinstance(input_text, str):
        input_text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', input_text)
        input_text = re.sub(r'[[]]', ' ', input_text)
        input_text = input_text.lower()
        input_text = [input_text]

    # Predict using the pipeline
    predictions = model.predict(input_text)
    
    # Map the numerical prediction to the corresponding class
    predicted_language = classes[predictions[0]]
    return predicted_language

if __name__ == "__main__":
    input_text = "This is an example sentence."
    predicted_language = predict_pipeline(input_text)
    print(f"Predicted language: {predicted_language}")

