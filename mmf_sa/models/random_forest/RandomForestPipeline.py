from pyspark.ml.regression import RandomForestRegressor 
from mmf_sa.models.abstract_model import ForecastingRegressor

class RandomForestPipeline(ForecastingRegressor): 
    def __init__(self, params): 
        super().__init__(params)
        self.params = params 
        self.model = RandomForestRegressor()

