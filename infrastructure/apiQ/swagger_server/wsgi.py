#!/usr/bin/env python3
import connexion
from swagger_server import encoder

def create_app():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'Swagger MLOPS OCRFLOW - OpenAPI 3.0'}, pythonic_params=True)
    return app

app = create_app()

if __name__ == '__main__':
    app.run(port=5000)
