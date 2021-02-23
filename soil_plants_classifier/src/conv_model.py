#conv_model.py

import tensorflow as tf
import config


model_path = 'config.saved_model_dir'

# convert the model
model = tf.keras.models.load_model(model_path)
print('model loaded!')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print('model converted!')

print('saving model...')
# save the lite model
with open('/content/drive/MyDrive/project1/models/soil_model.tflite','wb') as f:
    f.write(tflite_model)

# done
print('convertion finished!')
