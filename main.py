import joblib as j  
from fastapi import FastAPI  
from pydantic import BaseModel  


app=FastAPI(title="IRIS API Prediction ") 
model=j.load('svc.pkl')   



# Schema     
class iris(BaseModel): 
    sepal_length:float
    sepal_width:float 
    petal_length:float 
    petal_width:float   


@app.get("/")  
def home(): 
    return{'message':"iris FastApi is running"}



@app.post("/predict") 


def predict(data:iris):  
    try: 
        features=[[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        predction=model.predict(features)

        return{
            "prediction": int(predction[0]), 
            "species": ["setosa",'versicolor','verginica'][predction[0]]
          }  


    except  Exception as e:  
        return {'error': str(e)}
        