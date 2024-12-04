from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import jinja2

# Initialize FastAPI
app = FastAPI()

# Load the model and features
try:
    features = joblib.load("Insurance_features.pkl")
    model = joblib.load("Insurance.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model or features: {str(e)}")

# Jinja2 setup
template_loader = jinja2.FileSystemLoader(searchpath="./")
template_env = jinja2.Environment(loader=template_loader)
template_file = "pop_template.html"

@app.get("/", response_class=HTMLResponse)
async def charges():
    return """
    <html>
        <head>
            <title>Insurance Charges Predictor</title>
        </head>
        <body>
            <h2>Insurance Charges Predictor</h2>
            <p>Welcome to the Insurance Charges Predictor! Use this application to estimate insurance charges based on features from the Insurance Dataset.</p>
            
            <h3>Required Query Parameters</h3>
            <p>Send a GET request to the <code>/predict</code> endpoint with the following parameters:</p>
            <pre>
age: Person's age (e.g., 20)
sex: Gender (0 for female, 1 for male)
bmi: Body Mass Index (e.g., 21.6)
children: Number of children (e.g., 0, 1, 2, etc.)
smoker: Smoker status (0 for no, 1 for yes)
region: Residential region (e.g., southeast, northwest, northeast, southwest)
            </pre>

            <h3>Example</h3>
            <p>Send a GET request to:</p>
            <pre>
http://127.0.0.1:8000/predict?smoker=0&age=20&sex=1&bmi=21.6&children=0& region=southeast
            </pre>
            <p>This example predicts insurance charges for a person with the specified attributes.</p>
        </body>
    </html>
    """


@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    try:
        # Parse and validate query parameters
        query_params = request.query_params
        input_data = []
        for feature in features:
            if feature not in query_params:
                raise HTTPException(status_code=400, detail=f"Missing required parameter: {feature}")
            input_data.append(float(query_params[feature]))
        
        # Make the prediction
        prediction = model.predict([input_data])[0]

        # Render the template with prediction
        template = template_env.get_template(template_file)
        return template.render(prediction=prediction)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)
