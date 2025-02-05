import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='bitsandbytes')

# Set logging level
import transformers
transformers.logging.set_verbosity_error()

model_name = 'deepseek-ai/DeepSeek-V2-Lite-Chat'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, low_cpu_mem_usage=True,
    torch_dtype=torch.float16, quantization_config=bnb_config
) # .to('cuda')
print(f'Model loaded on {model.device}')

chat_history = []

def chatbot():
    print(f'Chat ready! You are speaking with {model_name}. Type "exit" to quit.')
    while True:
        # User input
        user_input = input('\nYou: ')
        if user_input.lower() == 'exit':
            print('Goodbye!')
            break

        # Format conversation history
        chat_history.append(f'User: {user_input}')
        formatted_input = '\n'.join(chat_history) + '\nDeepSeek:'

        # Tokenize input
        inputs = tokenizer(formatted_input, return_tensors='pt').to(model.device)

        # Generate response
        start = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000, # Limit response length
                temperature=0.7, # Control randomness
                top_p=0.9, # Control diversity
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decide and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (remove user text)
        if 'DeepSeek:' in response:
            response = response.split('DeepSeek:')[-1].strip()

        # Store chat history
        chat_history.append(f'DeepSeek: {response}')

        # Print response
        print(
            f'\nDeepSeek: {response}'
            f'\nTime taken: {time.time() - start:.2f} seconds'
        )

if __name__ == '__main__':
    chatbot()