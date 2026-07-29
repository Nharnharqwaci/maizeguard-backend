import tensorflow as tf
import tf2onnx
import os

# Load the trained Keras model
model = tf.keras.models.load_model(
    "app/models/best.keras",
    compile=False
)

# Define the model input
spec = (
    tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),
)

# Convert to ONNX
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,
)

# Create output folder
os.makedirs("../frontend/public", exist_ok=True)

# Save model
with open("../frontend/public/model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("✅ model.onnx exported successfully!")
