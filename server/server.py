from bson.json_util import dumps, loads
import socket
import tornado.web
from io import BytesIO
import numpy as np

import pandas as pd

from scripts.model import load_model, compile_model, len_window
from scripts.data_preprocess import sliding_window
model = compile_model(load_model())


class CorsHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept, X-Requested-With")
        # self.set_header('Access-Control-Allow-Methods', 'OPTIONS, TRACE, GET, HEAD, POST, PUT')


class DiagnoseHandler(CorsHandler):
    def get(self):
        csv = BytesIO(self.request.body)
        file = pd.read_csv(csv, ';', names=['times', 'parts', 'x', 'y', 'z'])

        data = file[['x', 'y', 'z']].to_numpy()
        # temp_arr = np.linalg.norm(array, axis=-1, keepdims=True)
        data = data.reshape(-1, 20, 3)
        data = np.array([data[0:len_window]])

        result = model.predict(data)
        print(result)

        self.write(str(result.tolist()))
        self.finish()


class StudentSessionsHandler(CorsHandler):
    def get(self, student_id):
        self.finish()


application = tornado.web.Application([
    (r'/api/diagnose', DiagnoseHandler),
    (r'/api/students/(?P<student_id>[a-zA-Z0-9_\-]+)/sessions', StudentSessionsHandler),
], debug=True)

port = 8080
http_server = tornado.httpserver.HTTPServer(application)
http_server.listen(port)
host = socket.gethostbyname(socket.gethostname())

print('*** Web Server Started at %s at port %s ***' % (host, port))
tornado.ioloop.IOLoop.instance().start()

