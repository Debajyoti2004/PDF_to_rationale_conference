from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
import pandas as pd


csv_file = r"C:\Users\Debajyoti\OneDrive\Desktop\project task-1\data\main_csv_data.csv"
df = pd.read_csv(csv_file)

model_name='google/flan-t5-base'

llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

