from keras import models, layers, saving
from _system.log_config import setup_logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pathlib as pl

class ModelManager():
    
    def __init__(self):
        self.logger = setup_logger(name=__name__)
        self.model = None
        self.dataset = None
        self.logger.info(msg=f'ModelManager <{id(self)}> created')
        
    def set_dataset(self, dataset:pd.DataFrame) -> None:
        self.dataset = dataset
    
    def init_model(self):
        self.model = models.Sequential(
            layers=[
                layers.Input(shape=(128,128,3)),
                layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Flatten(),
                layers.Dense(units=128, activation='relu'),
                layers.Dropout(rate=.5),
                layers.Dense(units=64, activation='relu'),
                layers.Dense(units=2, activation='sigmoid')
            ]
        ) 
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        ) 
        self.model.summary()
    
    def train_model(self, train_split:float=.8) -> None:
        train, test = train_test_split(self.dataset, train_size=train_split)
        x_train = np.stack(train['image'].values)
        y_train = train['label'].values
        x_test = np.stack(test['image'].values)
        y_test = test['label'].values
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y=y_train)
        y_test_encoded = encoder.fit_transform(y=y_test)
        
        print(x_train.shape, y_train_encoded.shape)
        
        self.model.fit(
            x=x_train,
            y=y_train_encoded,
            batch_size=10,
            epochs=5,
            validation_split=.2
        )
        
        loss, accuracy = self.model.evaluate(
            x=x_test, 
            y=y_test_encoded, 
            batch_size=10
        )
        
        self.logger.info(msg=f'MODEL STATS [LOSS: {loss}, ACC: {accuracy}]')
    
    def save_model(self, out_path:pl=pl.Path('_model', 'face_recognition_model.keras')):
        self.model.save(filepath=out_path)
        
    def load_model(in_path:pl=pl.Path('_model', 'face_recognition_model.keras')):
        return saving.load_model(filepath=in_path)
