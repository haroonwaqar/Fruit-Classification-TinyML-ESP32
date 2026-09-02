import tensorflow as tf
import edgeimpulse as ei
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
EDGE_API = os.getenv('EDGE_IMPULSE_API')

# Edge Impulse Authentication
ei.API_KEY = EDGE_API

# Load the Original Model
print("Loading trained model:")
original_model = tf.keras.models.load_model('fruit_classifier.keras')

# Strip Augmentation
inference_model = tf.keras.Sequential()
inference_model.add(tf.keras.layers.Input(shape=(96, 96, 1)))

# The data_augmentation layer is at index 0. We skip it and add everything else.
for layer in original_model.layers[1:]:
    inference_model.add(layer)

# Transfer the trained weights perfectly to the new clean model
inference_model.set_weights(original_model.get_weights())
print("Clean inference model ready.")

# Prepare Representative Data
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/Users/haroon/Developer/Fruit_Fresh_ESP/dataset_Fruits/train",
    color_mode='grayscale',
    image_size=(96, 96),
    batch_size=100, 
    shuffle=True
)

for images, labels in train_ds.take(1):
    sample_data = images.numpy()

# Deploy to Edge Impulse Arduino ZIP Target
print("\nUploading clean model to Edge Impulse for INT8 Quantization")
try:
    deploy_bytes = ei.model.deploy(
        model=inference_model,
        model_output_type=ei.model.output_type.Classification(
            labels=['apple', 'banana', 'orange']
        ),
        deploy_target='arduino',
        representative_data_for_quantization=sample_data
    )
    
    with open('ei_fruit_classifier_arduino.zip', 'wb') as f:
        if isinstance(deploy_bytes, bytes):
            f.write(deploy_bytes)
        else:
            f.write(deploy_bytes.getvalue())
            
    print("\nSuccess! Quantized Arduino library saved as 'ei_fruit_classifier_arduino.zip'")
    
except Exception as e:
    print(f"Edge Impulse error: {e}")