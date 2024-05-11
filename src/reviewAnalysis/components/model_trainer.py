import pandas as pd
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.optimizers import RMSprop
from reviewAnalysis.entity.config_entity import ModelTrainerConfig 


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        # Load your dataset
        df = pd.read_csv(os.path.join(self.config.data_path, "main_df.csv"))
        df['Review'] = df['Review'].astype(str)

        # Prepare data
        x = df['Review']
        y = df['Rating']

        # Convert ratings to categorical labels
        y = to_categorical(y - 1)  

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

        # Tokenize text data
        max_words = self.config.max_words
        max_len = self.config.max_len

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(x_train)

        sequences_train = tokenizer.texts_to_sequences(x_train)
        sequences_test = tokenizer.texts_to_sequences(x_test)

        # Pad sequences
        sequences_matrix_train = pad_sequences(sequences_train, maxlen=max_len)
        sequences_matrix_test = pad_sequences(sequences_test, maxlen=max_len)

        # Saving tokenizer
        with open(os.path.join(self.config.root_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.config.root2_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Define model architecture
        model = Sequential()
        model.add(Embedding(max_words, 100, input_length=max_len))  
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(5, activation='softmax')) \

        model.summary()

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

        # Train model
        model.fit(sequences_matrix_train, y_train, batch_size=self.config.batch_size,
                  epochs=self.config.epochs, validation_split=self.config.validation_split)

        # Evaluate model
        accr = model.evaluate(sequences_matrix_test, y_test)

        # Save evaluation metrics
        metrics = {"eval": accr}
        with open(os.path.join(self.config.root_dir, 'metrics.json'), "w") as file:
            json.dump(metrics, file)

        # Save model
        model.save(os.path.join(self.config.root_dir, 'model.h5'))
        model.save(os.path.join(self.config.root2_dir, 'model.h5'))

