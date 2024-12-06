import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
import json
import os

class OneHotLayer(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHotLayer, self).__init__(**kwargs)
        self.depth = depth

    def call(self, inputs):
        inputs = tf.cast(inputs, 'int32')
        return tf.one_hot(inputs, depth=self.depth)

class AttentionLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        query, key = inputs
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, key)
        return attention_output

class GenerativeAIChatbot:
    def __init__(self, max_words=10000, max_len=50):
        """
        Khởi tạo chatbot với các tham số cấu hình
        
        Tham số:
            max_words (int): Số lượng từ tối đa trong từ điển
            max_len (int): Độ dài tối đa của chuỗi đầu vào/đầu ra
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None
        
    def prepare_data(self, conversations):
        """
        Chuẩn bị dữ liệu huấn luyện từ tập dữ liệu hội thoại
        
        Tham số:
            conversations (list): Danh sách các cặp hội thoại
        """
        # Trích xuất văn bản đầu vào và đầu ra
        input_texts = [conv['input'] for conv in conversations]
        output_texts = [conv['output'] for conv in conversations]
        
        # Huấn luyện tokenizer trên tất cả các văn bản
        self.tokenizer.fit_on_texts(input_texts + output_texts)
        
        # Chuyển đổi văn bản thành chuỗi số
        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        output_sequences = self.tokenizer.texts_to_sequences(output_texts)
        
        # Đệm các chuỗi
        self.input_data = pad_sequences(input_sequences, maxlen=self.max_len, padding='post')
        self.output_data = pad_sequences(output_sequences, maxlen=self.max_len, padding='post')
        
        # Tạo từ điển
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1
        
    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.max_len,))
        encoder_embedding = Embedding(self.vocab_size, 256, mask_zero=True)(encoder_inputs)
        encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(self.max_len,))
        decoder_embedding = Embedding(self.vocab_size, 256, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        # Attention mechanism wrapped in custom layer
        attention_result = AttentionLayer()([decoder_outputs, encoder_outputs])

        # Concatenate attention output with decoder outputs
        decoder_concat_input = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, attention_result])

        # Dense layer to predict the vocabulary
        output = Dense(self.vocab_size, activation='softmax')(decoder_concat_input)

        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    def train(self, epochs=50, batch_size=32):
        """
        Huấn luyện mô hình mạng nơ-ron
        
        Tham số:
            epochs (int): Số lượng epoch huấn luyện
            batch_size (int): Kích thước batch cho huấn luyện
        """
        if self.model is None:
            self.build_model()
        
        # Chu���n bị dữ liệu đích cho huấn luyện
        target_data = np.expand_dims(self.output_data, axis=-1)
        
        self.model.fit(
            [self.input_data, self.output_data], 
            target_data, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=0.2
        )
        
    def generate_response(self, input_text, max_response_length=50):  # Changed to match max_len
        input_seq = self.tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=self.max_len, padding='post')
        
        # Initialize response sequence with zeros
        generated_sequence = np.zeros((1, self.max_len))  # Changed to use self.max_len
        generated_sequence[0, 0] = self.tokenizer.word_index.get('<start>', 1)
        
        for i in range(1, self.max_len):  # Changed to use self.max_len
            decoder_pred = self.model.predict([input_seq, generated_sequence])
            predicted_word_index = np.argmax(decoder_pred[0, i-1, :])
            
            generated_sequence[0, i] = predicted_word_index
            
            if predicted_word_index == self.tokenizer.word_index.get('<end>', 2):
                break
        
        # Chuyển đổi chuỗi số trở lại thành văn bản
        response_words = []
        for idx in generated_sequence[0]:
            if idx > 0:
                word = self.tokenizer.index_word.get(idx, '')
                response_words.append(word)
        
        return ' '.join(response_words)
    
    def save_model(self, filepath='chatbot_model'):
        """Lưu mô hình đã huấn luyện và tokenizer"""
        self.model.save(filepath + '.keras')  # Add .keras extension
        with open(f'{filepath}_tokenizer.json', 'w') as f:
            json.dump({
                'word_index': self.word_index,
                'word_index_inverse': {v: k for k, v in self.word_index.items()}
            }, f)
    
    def load_model(self, filepath='chatbot_model'):
        """Tải mô hình đã huấn luyện trước và tokenizer"""
        self.model = tf.keras.models.load_model(filepath + '.keras')  # Add .keras extension
        with open(f'{filepath}_tokenizer.json', 'r') as f:
            tokenizer_data = json.load(f)
            self.word_index = tokenizer_data['word_index']
            self.word_index_inverse = tokenizer_data['word_index_inverse']


# Ví dụ sử dụng và dữ liệu huấn luyện
example_conversations = [
    {"input": "hello", "output": "hi there"},
    {"input": "how are you", "output": "i'm doing great, thanks"},
    {"input": "what's your name", "output": "i'm an AI chatbot"},
    {"input": "tell me a joke", "output": "why did the AI cross the road? To optimize the path!"}
]

# Khởi t���o và huấn luyện chatbot
chatbot = GenerativeAIChatbot()
chatbot.prepare_data(example_conversations)
chatbot.train(epochs=100)

# Kiểm tra chatbot
test_input = "hello"
response = chatbot.generate_response(test_input)
print(f"Input: {test_input}")
print(f"Response: {response}")

# Tùy chọn lưu mô hình
chatbot.save_model()