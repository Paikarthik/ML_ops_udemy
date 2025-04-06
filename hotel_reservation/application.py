import joblib 
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load(MODEL_OUTPUT_PATH)

@app.route("/", methods= ['GET', 'POST'])
def index():

    if request.method == "POST":
        # obtain the data from the form
        lead_time = int(request.form["lead_time"])
        no_of_special_request = int(request.form["no_of_special_request"])
        avg_price_per_room = float(request.form["avg_price_per_room"])
        arrival_month = int(request.form["arrival_month"])
        arrival_date = int(request.form["arrival_date"])
        market_segment_type = int(request.form["market_segment_type"])
        no_of_week_nights = int(request.form["no_of_week_nights"])
        no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
        type_of_meal_plan = int(request.form["type_of_meal_plan"])
        room_type_reserved = int(request.form["room_type_reserved"])

        # convert the data into numpy array
        features = np.array([[lead_time,
                              no_of_special_request,
                              avg_price_per_room,
                              arrival_month, 
                              arrival_date,
                              market_segment_type,
                              no_of_week_nights,
                              no_of_weekend_nights,
                              type_of_meal_plan,
                              room_type_reserved]])
        
        # predict using the model 
        prediction = model.predict(features)

        return render_template('index.html', prediction=prediction[0])
    
    else:
        return render_template('index.html', prediction= None)
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080)



