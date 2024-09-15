from apis import app

def handler(request, response):
    return app(request, response)
