from transformers import BertTokenizer
import tensorflow as tf
import pandas as pd 
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
from tensorflow.keras import layers, Model
from Callbacks import CustomCallback,lr_scheduler


csv_file_path = r"C:\Users\Debajyoti\OneDrive\Desktop\project task-1\data\updated_publishable_data.csv"
df = pd.read_csv(csv_file_path)
texts = df['PDF'].tolist() 
labels = df['Label'].tolist()

labels_tensor = tf.convert_to_tensor(labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, max_length=512):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

csv_file_path = r"C:\Users\Debajyoti\OneDrive\Desktop\project task-1\data\updated_publishable_data.csv"
df = pd.read_csv(csv_file_path)
texts = df['PDF'].tolist() 
labels = df['Label'].tolist()

labels_tensor = tf.convert_to_tensor(labels)
tokenized_data = tokenize_texts(texts)
input_ids, attention_masks = tokenized_data['input_ids'], tokenized_data['attention_mask']
X_train = (input_ids, attention_masks)
Y_train = labels_tensor

class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model_name='bert-base-uncased', **kwargs):
        super().__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(bert_model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output  # Use pooled output for classification tasks

input_ids_layer = tf.keras.Input(shape=(512,), dtype=tf.int32, name='input_ids')
attention_mask_layer = tf.keras.Input(shape=(512,), dtype=tf.int32, name='attention_mask')

bert_output = BertLayer()(inputs=[input_ids_layer, attention_mask_layer])

output = layers.Dense(1, activation='sigmoid')(bert_output)

model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])


log = {} 
custom_callback = CustomCallback(log)
history = model.fit(
    x={'input_ids': X_train[0], 'attention_mask': X_train[1]},
    y=Y_train,
    epochs=20,
    batch_size=4,
    callbacks=[custom_callback, lr_scheduler]
)
model.save_weights(r"C:\Users\Debajyoti\OneDrive\Desktop\project task-1\model\bert_model_weights.weights.h5")

print("models weight saved successfully!")

prediction = model.predict({'input_ids': X_train[0], 'attention_mask': X_train[1]})

