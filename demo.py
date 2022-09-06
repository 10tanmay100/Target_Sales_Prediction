from xml.etree.ElementTree import PI
from sales_prediction.config.configuration import Configuration
from sales_prediction.pipeline.pipeline import Pipeline
def run():
    c=Pipeline(Configuration)
    r=c.run_pipeline()
    print(r)



run()