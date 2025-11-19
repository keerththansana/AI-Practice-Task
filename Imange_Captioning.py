import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet')
base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Function to preprocess and extract features from an image
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array)
    return features

# Example usage
image_path = 'path/to/your/image.jpg'
image_features = extract_features(image_path)

# Dummy caption data (replace this with your dataset)
captions = ["startseq dog playing in the grass endseq", "startseq brown dog running on the beach endseq", ...]

# Tokenize captions
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1

# Create sequences
sequences = tokenizer.texts_to_sequences(captions)

# Create input-output pairs for training
X, y, image_data = [], [], []
for sequence in sequences:
    for i in range(1, len(sequence)):
        in_seq, out_seq = sequence[:i], sequence[i]
        in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X.append(image_features)
        y.append(out_seq)
        image_data.append(in_seq)

X = np.array(X)
y = np.array(y)
image_data = np.array(image_data)

# Image captioning model
image_input = Input(shape=(4096,))
image_dense = Dense(256, activation='relu')(image_input)

caption_input = Input(shape=(max_len,))
caption_embedding = Embedding(vocab_size, 256, input_length=max_len)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)

decoder_input = Concatenate()([image_dense, caption_lstm])
output = Dense(vocab_size, activation='softmax')(decoder_input)

model = Model(inputs=[image_input, caption_input], outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model (replace X and y with your actual data)
model.fit([X, image_data], y, epochs=10, batch_size=32)

# Use the trained model for generating captions for new images
def generate_caption(image_path):
    image_features = extract_features(image_path)
    in_text = 'startseq'
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Function to map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
