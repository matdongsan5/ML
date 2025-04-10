from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index_socket.html')

def background_thread():
    while True:
        socketio.emit('update', {'data': time.ctime()})
        time.sleep(1)

@socketio.on('connect')
def handle_connect():
    print("Client connected")

if __name__ == '__main__':
    threading.Thread(target=background_thread).start()
    socketio.run(app)
