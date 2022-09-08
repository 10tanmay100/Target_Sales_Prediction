from inspect import CO_GENERATOR
from sales_prediction.config.configuration import Configuration
from sales_prediction.pipeline.pipeline import Pipeline
def run():
    c=Pipeline(Configuration)
    r=c.run_pipeline()
    print(r)
    # c=Configuration()
    # print(c.get_data_transformation_config())



run()