from openai import OpenAI
import time
import os
import logging
import pickle

logger = logging.getLogger('root')

class OpenAIAPI:
    def __init__(self, args):
        self.settings = vars(args)
        self.client = OpenAI(
            api_key=args.api_key
        )
        self.max_tokens = args.max_tokens

        if args.gpt_model == 'gpt4':
            self.settings['model'] = 'gpt-4-0125-preview'
        elif args.gpt_model == 'chatgpt':
            self.settings['model'] = 'gpt-3.5-turbo-0125'

    def chatgpt_response(self, message):
        try:
            response = self.client.chat.completions.create(
                model=self.settings['model'],
                messages=[
                    {"role": "user", "content": message}
                ],
                max_tokens=self.settings['max_tokens'],
                seed=self.settings['seed'],
                temperature=self.settings['temperature']
            )
            chatgpt_answer = response.choices[0].message.content.strip("\n")
            return chatgpt_answer
        
        # except OpenAI.RateLimitError as e:
        except Exception as e:
            # Handle rate limit error
            print(f"Error message: {e}")
            # Wait for the recommended duration before retrying
            print("Waiting for 60 seconds to start again...")
            time.sleep(60)
            # Retry the API call
            return self.chatgpt_response(message)
    
    def generate_response(self, prompts, output_dir, ckpt_dir=None):
        
        result = []
        ckpt_idx = 0
        
        if ckpt_dir is not None:
            with open(ckpt_dir, 'rb') as f:
                generated_responses = pickle.load(f)
                logger.info(f"Length of Loaded Outputs >>> {len(generated_responses)}")
            ckpt_idx = len(generated_responses)
            result = generated_responses
            prompts = prompts[ckpt_idx:]

        # TODO: batching inputs?
        for prompt in prompts:
            answer = self.chatgpt_response(prompt)
            result.append(answer)
            ckpt_idx += 1
            
            if (ckpt_idx % 50 == 0) or (ckpt_idx == len(prompts)):
                logger.info(f'Saving ckpt at {ckpt_idx}th responses')
                # Save partial response
                CKPT = os.path.join(output_dir, 'ckpt')
                if not os.path.exists(CKPT):
                    print(f"{CKPT} does not exist. Creating...")
                    os.makedirs(CKPT)
                with open(os.path.join(CKPT, f"result_{ckpt_idx}.pkl"), "wb") as f:
                    pickle.dump(result, f)

        logger.info("API CALL FINISHED")
        return result