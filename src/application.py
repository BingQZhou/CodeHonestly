from flask import Flask, request
from flask import render_template
from flask_cors import CORS
import preprocess as pp, json, similarity as sim, traceback, logging

application = Flask(__name__)
CORS(application)
logging.basicConfig(level=logging.DEBUG)

# @application.route('/')
def index():
    return render_template('index.html')

@application.route('/ast2json', methods=['POST'])
def process():
    return json.dumps(pp.process(request.form.get('pysrc'), request.form.get('ctx') == 'true', request.form.get('normalize') == 'true'))

@application.route('/simreport', methods=['POST'])
def similarity():
    try:
        trees = [(key, pp.process(request.form.get(key))) for key in request.form.keys() if key[:5] == 'pysrc' and key[5:].isdigit()]
        return json.dumps(sim.process(trees))
    except Exception as e:
        logging.error(traceback.format_exc())
        return None

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0')
