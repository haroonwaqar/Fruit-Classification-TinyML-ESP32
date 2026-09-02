import tensorflow as tf
import numpy as np

print("Loading trained model and stripping augmentation...")
original_model = tf.keras.models.load_model('fruit_classifier.keras')

inference_model = tf.keras.Sequential()
inference_model.add(tf.keras.layers.Input(shape=(96, 96, 1)))

for layer in original_model.layers[1:]:
    inference_model.add(layer)
inference_model.set_weights(original_model.get_weights())

# Bypass Keras 3 Bug via Concrete Function because running directly gives errors on M-Series chips
run_model = tf.function(lambda x: inference_model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 96, 96, 1], tf.float32)
)

# Prepare Representative Dataset
print("Preparing calibration data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/Users/haroon/Developer/Fruit_Fresh_ESP/dataset_Fruits/train",
    color_mode='grayscale',
    image_size=(96, 96),
    batch_size=1, 
    shuffle=True
)

def representative_data_gen():
    for input_value, _ in train_ds.take(100):
        yield [input_value]

# Quantize to INT8 
print("\nQuantizing model to INT8")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8 
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open('fruit_classifier_local_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)
print("Saved as 'fruit_classifier_local_quantized.tflite'")

# Verify Against Isolated Verification Dataset
print("\nRunning Verification on Unseen Data ")
verify_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/Users/haroon/Developer/Fruit_Fresh_ESP/dataset_Fruits/verification",
    color_mode='grayscale',
    image_size=(96, 96),
    batch_size=1,
    shuffle=False
)

interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details["quantization"]

correct_predictions = 0
total_images = 0

for img, label in verify_ds:
    # Convert float image to INT8 using the exact math as ESP32
    img_int8 = np.clip(np.round((img / input_scale) + input_zero_point), -128, 127).astype(np.int8)
    
    interpreter.set_tensor(input_details['index'], img_int8)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details['index'])
    prediction = np.argmax(output[0])
    true_label = label.numpy()[0]
    
    if prediction == true_label:
        correct_predictions += 1
    total_images += 1

print(f"\nFinal INT8 Verification Accuracy: {(correct_predictions/total_images)*100:.2f}% ({correct_predictions}/{total_images})")