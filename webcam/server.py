import argparse, random, os, time, json

from PIL import Image
from io import BytesIO
import base64

from flask import Flask, request
from flask.ext.cors import CORS
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
CORS(app)


input_dir = 'webcam/inputs'
output_dir = 'webcam/outputs'


class DenseCap(Resource):
  def get(self):
    return 'The DenseCap server seems to be running!'

  def post(self):
    img_id = random.randint(1, 1000000)
    img_name = os.path.join(input_dir, '%d.jpg' % img_id)

    # Get the base64 image data out of the request.
    # for some reason Flask doesn't parse this out at all for use, so we'll just
    # do it manually. There is a prefix telling us that this is an image and the
    # type of the image, then a comma, then the raw base64 data for the image.
    # We just grab the part after the comma and decode it.
    idx = request.data.find(',') + 1
    img_data = request.data[idx:]
    
    im = Image.open(BytesIO(base64.b64decode(img_data)))
    im.save(img_name)

    # request.files['image'].save(img_name)
    json_name = os.path.join(output_dir, '%d.json' % img_id)
    while not os.path.isfile(json_name):
      time.sleep(0.05)
    with open(json_name, 'r') as f:
      ann = json.load(f)
    os.remove(json_name)
    return ann

api.add_resource(DenseCap, '/')


if __name__ == '__main__':
  from tornado.wsgi import WSGIContainer
  from tornado.httpserver import HTTPServer
  from tornado.ioloop import IOLoop

  http_server = HTTPServer(WSGIContainer(app), ssl_options={
    'certfile': 'webcam/ssl/server.crt',
    'keyfile': 'webcam/ssl/server.key'
  })

  http_server.listen(5000)

  # We have to do a little weirdness to make the server actually die
  # when we hit CTRL+C
  try:
    IOLoop.instance().start()
  except KeyboardInterrupt:
    IOLoop.instance().stop()
