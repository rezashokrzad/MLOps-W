from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import jinja2

# inilize fastapi
app = FastAPI()

# load the model and features
model = joblib.load("california_model.pkl")
features = joblib.load("california_features.pkl")

# jinja2 setup
template_loader = jinja2.FileSystemLoader(searchpath="./")
template_env = jinja2.Environment(loader=template_loader)
template_file = "popup_template.html"

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>California Housing Price Predictor</title>
        </head>
        <body>
            <h2>California Housing Price Predictor</h2>
            <p>Welcome to the California Housing Price Predictor! Use this application to estimate housing prices based on features from the California Housing Dataset.</p>
            
            <h3>Required Query Parameters</h3>
            <p>Send a GET request to the <code>/predict</code> endpoint with the following parameters:</p>
            <pre>
MedInc: Median income in the block group
HouseAge: Median age of the houses in the block group
AveRooms: Average number of rooms per household
AveBedrms: Average number of bedrooms per household
Population: Block group population
AveOccup: Average household size
Latitude: Latitude of the block group
Longitude: Longitude of the block group
            </pre>

            <h3>Example</h3>
            <p>Send a GET request to:</p>
            <pre>
http://127.0.0.1:8000/predict?MedInc=4.0&HouseAge=25&AveRooms=6&AveBedrms=1.2&Population=800&AveOccup=3.5&Latitude=37.5&Longitude=-120.0
            </pre>
            <div class="example">
                <p>This example predicts the median house value for a block group with the given values.</p>
            </div>
        </body>
    </html>
    """.format(", ".join(features))

@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    try:
        query_params = {key: float(value) for key, value in request.query_params.items()}

        if len(query_params) != len(features):
            raise HTTPException(status_code=400,
                               detail="Missing or extra query parameters. ")
        # prepare input data for prediction
        input_date = [query_params.get(feature, 0) for feature in features]

        # make the prediction 
        prediction = model.predict([input_date])[0]

        #render the template
        template = template_env.get_template(template_file)

        return template.render(prediction=round(prediction, 2))
    except Exception as e:
        return f"Error: {str(e)}"



