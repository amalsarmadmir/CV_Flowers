import tensorflow as tf
import scipy.io
import os

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load image labels
labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

# Load dataset splits
splits = scipy.io.loadmat('setid.mat')
train_ids = splits['trnid'][0]
val_ids = splits['valid'][0]
test_ids = splits['tstid'][0]

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

def load_image_paths(image_ids):
    image_paths = [f'jpg/image_{i:05d}.jpg' for i in image_ids]
    labels_for_ids = [labels[i-1] - 1 for i in image_ids]  # Adjust labels to start from 0
    return tf.data.Dataset.from_tensor_slices((image_paths, labels_for_ids))

def load_dataset(image_ids):
    dataset = load_image_paths(image_ids)
    dataset = dataset.map(lambda path, label: (preprocess_image(path), label))
    return dataset

# Create datasets with reduced batch size
train_dataset = load_dataset(train_ids).shuffle(1000).batch(16)  # Reduced batch size
val_dataset = load_dataset(val_ids).batch(16)
test_dataset = load_dataset(test_ids).batch(16)

# Load and prepare the ResNet-50 model
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(102, activation='softmax')  # 102 classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)]
)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

