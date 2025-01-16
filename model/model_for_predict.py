from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
from tensorflow.keras import layers, Model

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

classification_model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
weights_file = r"C:\Users\Debajyoti\OneDrive\Desktop\project task-1\model\bert_model_weights.weights.h5"

classification_model.load_weights(weights_file)
print("Weight loaded successfully!")
