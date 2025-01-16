from model import llm_model, tokenizer,df
from create_prompt import create_prompt_text

example_indices_full = [0, 1, 2]
rationaleies = ["The MetaFood Workshop Challenge aims to improve food portion estimation by integrating computer vision with culinary practices. Traditional methods like 24-Hour Recall and Food Frequency Questionnaires are prone to inaccuracies due to self-reporting biases. By developing solutions that generate accurate 3D food models from multiple 2D images, the challenge seeks to enhance the precision of food portion measurements. This can revolutionize nutritional tracking, improve dietary assessments, and support public health initiatives. Experts from computer vision, machine learning, and nutrition science are collaborating to advance 3D food reconstruction techniques, offering potential breakthroughs in both health and technology.",
             "Contemporary recommender systems often face popularity bias, where popular items are overrepresented and unpopular items underrepresented, leading to imbalanced recommendations. Collaborative filtering methods typically struggle with this, resulting in skewed recommendation quality. To address this, a re-weighting approach is introduced, adjusting the importance of items based on popularity during training. Using contrastive learning, the model optimizes item representations to ensure balanced recommendations. Hyperparameters are used to control the weight of different item groups, while dynamic classification based on popularity further helps maintain balance. This method improves generalization, ensuring that both popular and unpopular items are fairly represented.",
             "This experimental research aims to push the boundaries of traditional scientific inquiry by employing whimsical and unconventional methods to explore photosynthesis. By integrating creative techniques like interpretive dance, puzzles, and cryptic messages, the study uncovers previously unobserved connections between photosynthesis and diverse phenomena, such as frisbee trajectories and harmonica-playing plants. The findings suggest that photosynthesis may have broader implications, influencing global ecosystems, climate change, and even extraterrestrial life. The rationale is to challenge established scientific frameworks, promoting a broader, more imaginative approach to research and encouraging exploration of photosynthesis' potential links to both life on Earth and the universe beyond."]

def create_summary_all_df(df, example_indices_full, rationaleies):
    pred_rationaleies = []
    
    for i in range(3, len(df)):
        current_prompt = create_prompt_text(example_indices_full, i, rationaleies, df)
        inputs = tokenizer.encode(current_prompt, return_tensors='pt') 
        outputs = llm_model.generate(
            inputs,
            max_new_tokens=150
        )
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_rationaleies.append(output)
        
    return pred_rationaleies


pred_rationaleies = create_summary_all_df(df,example_indices_full,rationaleies)
print("Congratulation this successfully completed!")
print("Some of summarization", pred_rationaleies[:3])

