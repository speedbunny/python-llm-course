from transformers import pipeline, TFAutoModelForSeq2SeqLM, AutoTokenizer

# English text
text = "It was a well-known fact in the village, where everyone knew everyone else and secrets were hard to keep, that the old man who lived in the last but one house by \
        the creek had once found a treasure chest when he was young."

# Transformer-based model
translator_transformer = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')

# RNN-based model (T5-small)
model_t5 = TFAutoModelForSeq2SeqLM.from_pretrained('t5-small')
tokenizer_t5 = AutoTokenizer.from_pretrained('t5-small')
translator_rnn = pipeline('translation_en_to_fr', model=model_t5, tokenizer=tokenizer_t5)

# Translate the text
translation_transformer = translator_transformer(text)[0]['translation_text']
translation_rnn = translator_rnn(text)[0]['translation_text']

print(f'Original English Text: {text}')
print(f'Translation by Transformer: {translation_transformer}')
print(f'Translation by RNN (T5-small): {translation_rnn}')
