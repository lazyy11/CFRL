# Using T5-base Model

To use the T5-base model, you can clone it directly from Hugging Face by running the following command:

```bash
git clone https://huggingface.co/google/t5-base
```

Alternatively, you can install it directly in your Python environment using the Hugging Face `transformers` library:

```bash
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/t5-base")
```

For more details on the model, please visit the [T5-base model page on Hugging Face](https://huggingface.co/google-t5/t5-base/tree/main).
