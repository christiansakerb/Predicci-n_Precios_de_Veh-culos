#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m09_model_deployment import predic_price

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Prediccion del precio de un automovil',
    description='Prediccion del precio de un automovil')

ns = api.namespace('predict', 
     description='Precio Classifier')
   
parser = api.parser()


parser.add_argument('marca', type=str, required=True, help='Brand of the car', location='args')
parser.add_argument('modelo', type=str, required=True, help='Model of the car', location='args')
parser.add_argument('millas', type=float, required=True, help='Number of miles driven by the car', location='args')
parser.add_argument('estado_uso', type=str, required=True, help='State of usage of the car (e.g. new, used)', location='args')
parser.add_argument('ano', type=int, required=True, help='Year of manufacture of the car', location='args')


resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predic_price(args['marca'], args['modelo'], args['millas'], args['estado_uso'], args['ano'])
        }, 200
    
#http://3.136.27.89:5000/predict/?marca=Toyota&modelo=Camry&millas=50000&estado_uso=used&ano=2015
 
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
