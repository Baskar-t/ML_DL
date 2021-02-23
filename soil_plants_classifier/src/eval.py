#eval.py

import model
import train
import soildata
import numpy as np

model.Model.evaluate(soildata.test_data)

y_pred =  model.Model.predict(soildata.test_data)
y_pred =  np.argmax(y_pred,axis=1)
print(len(soildata.test_data))
print(soildata.test_data.classes)
print(y_pred)