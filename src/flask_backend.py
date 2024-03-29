from flask import Flask, jsonify, request

class Modeler:
    def predict(self, x):
        return x**2

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def post():
    data = request.get_json()
    x = data['x']
    m = Modeler()
    prediction = m.predict(x)
    return jsonify({
        "Input": x,
        "y": prediction
    })

if __name__ == "__main__":
    ## Uncomment for flask only (no docker container)
    #app.run(port=5000,debug=True)
    ## Comment out for flask only (no docker container)
     app.run(host="0.0.0.0", port=8080)