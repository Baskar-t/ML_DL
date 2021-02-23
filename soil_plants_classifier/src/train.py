#train.py

import soildata
import model

history = model.Model.fit(soildata.train_data,validation_data= soildata.val_data, batch_size=32, epochs = 100, callbacks=[model.early])

model.Model.save('models/soil_model_19_Feb.h5')