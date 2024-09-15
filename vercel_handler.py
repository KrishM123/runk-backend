from apis import app
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app)

def handler(request, response):
    return app(request, response)
