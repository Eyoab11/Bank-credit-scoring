from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """
    Defines the input data schema for a prediction request.
    The field names must match the feature names of the model.
    """
    Recency: float
    Frequency: float
    Monetary: float
    AvgTransactionValue: float
    StdTransactionValue: float
    NumUniqueProducts: float
    MostFrequentChannel_ChannelId_1: float
    MostFrequentChannel_ChannelId_2: float
    MostFrequentChannel_ChannelId_3: float
    MostFrequentChannel_ChannelId_5: float
    
    class Config:
        # Pydantic configuration to allow creating from a dict
        from_attributes = True
        # Example for API documentation
        json_schema_extra = {
            "example": {
                "Recency": -0.8,
                "Frequency": 1.5,
                "Monetary": 0.5,
                "AvgTransactionValue": 0.2,
                "StdTransactionValue": -0.1,
                "NumUniqueProducts": -0.4,
                "MostFrequentChannel_ChannelId_1": 0.0,
                "MostFrequentChannel_ChannelId_2": 1.0,
                "MostFrequentChannel_ChannelId_3": 0.0,
                "MostFrequentChannel_ChannelId_5": 0.0
            }
        }

class PredictionResponse(BaseModel):
    """
    Defines the output data schema for a prediction response.
    """
    risk_probability_class_1: float