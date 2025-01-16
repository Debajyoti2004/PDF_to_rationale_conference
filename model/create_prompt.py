def cut_text(text):
    return text[400:1200]

def create_prompt_text(example_indices_full, example_index_to_prompt, rationaleies, df):
    prompt = ''
    for index in example_indices_full:
        long_text = df['PDF'].iloc[index]
        text = cut_text(long_text)
        rationale = rationaleies[index]
        prompt += f"""
             Text:
             {text}
             
             Rationale:
             {rationale}
        """
        
    long_text = text = df['PDF'].iloc[example_index_to_prompt]
    text = cut_text(long_text)
    prompt += f"""
            Text:
            {text}
            
            Rationale:
    """
    
    return prompt

def create_prompt_text_test(train_df,example_indices_full, example_index_to_prompt, rationaleies, test_df):
    prompt = ''
    for index in example_indices_full:
        long_text = train_df['PDF'].iloc[index]
        text = cut_text(long_text)
        rationale = rationaleies[index]
        prompt += f"""
             Text:
             {text}
             
             Rationale:
             {rationale}
        """
        
    long_text = text = test_df['Text'].iloc[example_index_to_prompt]
    text = cut_text(long_text)
    prompt += f"""
            Text:
            {text}
            
            Rationale:
    """
    
    return prompt



