from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import time
import roger
#from gorilla import extract_loc, get_gorilla_response, function_documentation
#from database import get_coordinates

app = Flask(__name__)
#import roger
# Replace with a strong secret key
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/process_query', methods=['POST'])
def process_query():
    # Get the request 
    data = request.json

    # Extract location coordinates and query from the request data
    #location = data.get('location')
    query = data.get('query')

    # Process the request (e.g., perform a calculation, query a database, etc.)
    gorilla_response = get_gorilla_response(query, functions=function_documentation)
    
    place = extract_loc(gorilla_response)
    long_lat = get_coordinates(place)
        
    
    # Send the response
    return jsonify({'message': 'Success', 'longitude': long_lat[0], 'latitude': long_lat[1]})






@socketio.on('connect')
def handle_connect():
    print('Client connected!')
    emit('connected', {'message': 'Welcome!'})


@socketio.on('start')
def handle_message(data):
    roger.global_stop_event.clear()
    query = data["input"]
    print("app.py", query)
    if query == "":
        roger.modelRun(False)
    else:
        roger.modelRunWithMemory(True, query)
        #roger.modelRun(True, query)
    

@socketio.on('stopmodel')
def stop_model():
    print("stop-model-receidved")
    roger.global_stop_event.set()
 

if __name__ == '__main__':
    socketio.run(app, host="192.168.1.41", port=3000,  debug=True)