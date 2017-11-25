import os.path
import requests
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import logging
from PIL import Image
from io import BytesIO
from tornado import httpclient

logger = logging.getLogger(__file__)

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)
http_client = httpclient.HTTPClient()
s = requests.Session()


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        url = 'http://jcxmt2017.jcrb.com/voteAPI/CodeKaptcha'
        r1 = s.get(url)
        i = Image.open(BytesIO(r1.content))
        i.save('static/temp.jpg')
        self.render('index.html')


class PoemPageHandler(tornado.web.RequestHandler):
    def post(self):
        yanzhengma = self.get_argument('noun1')
        logger.info(yanzhengma)
        voteString = ''
        for _ in range(10):
            voteString = voteString + '3,'
        serverUrl = "http://jcxmt2017.jcrb.com/voteAPI/xmtwb/pcVoteAdd" + "?captcha=" + yanzhengma + "&voteString=" + voteString + "&jsoncallback=?"
        r = s.get(serverUrl)
        # r = http_client.fetch(serverUrl)
        logger.info(r.text)
        self.redirect('/')


if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[(r'/', IndexHandler), (r'/poem', PoemPageHandler)],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=True)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()