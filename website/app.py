import os
import sys

import boto3
import json, plotly
from jinja2 import Environment
from jinja2 import FileSystemLoader
from chalice import Chalice, Response, NotFoundError, ChaliceViewError, BadRequestError
from wrangling_scripts.wrangle_data import return_figures

from logging import basicConfig
log = basicConfig(level='DEBUG')

app = Chalice(app_name='sentiment')
app.debug = True



def _render_template(**kwargs):
    "render jinja template"
    env = Environment(loader=FileSystemLoader(os.path.abspath(os.path.dirname(__file__))))
    template = env.get_template('chalicelib/main.html')
    rendered_template = template.render(kwargs)
    return rendered_template

@app.route('/')
@app.route('/index')
def index():

    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)


    return Response(_render_template(ids=ids,
                                     figuresJSON=figuresJSON),
                    status_code=200,
                    headers={'Content-Type': 'text/html'})


@app.route('/inference', methods=['POST'], content_types=['application/x-www-form-urlencoded'])
def inference():

    if sys.version_info[0] == 3:
        from urllib.parse import parse_qs
    # Python 3 imports.

    else:
        from urlparse import parse_qs
    # Python 2 imports.

    parsed = parse_qs(app.current_request.raw_body.decode())
    content = parsed["content"][0]

    if "SAGEMAKER_ENDPOINT" not in os.environ or "REPLACE_WITH" in os.environ["SAGEMAKER_ENDPOINT"]:
        raise ChaliceViewError("No SAGEMAKER_ENDPOINT configured")

    client = boto3.client('sagemaker-runtime')
    response = client.invoke_endpoint(
        EndpointName=os.environ["SAGEMAKER_ENDPOINT"],
        Body=content,
        ContentType='text/plain'
    )

    inference = response['Body'].read().decode()
    return Response(_render_template(inference = inference),
                    status_code=200,
                    headers={'Content-Type': 'text/html'})