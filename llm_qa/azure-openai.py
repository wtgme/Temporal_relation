import os
from openai import AzureOpenAI
import llm_qa


def temporal_inference_all_bulk(text, events, prompt_key="all_notime"):
    """Evaluate the model's ability to infer time for all events at once"""
    prompts = llm_qa.load_prompts()
    print(prompts)
    results = []
    
    if prompt_key not in prompts:
        raise ValueError(f"Prompt key '{prompt_key}' not found in prompts file")
    
    event_prompt = prompts[prompt_key]

    print(event_prompt + "\n\nCLINICAL TEXT: " + text)
    
    # No preprocessing needed for "all" mode - use text as is
    # For "all" mode, use guided_json since we expect a list of objects
    entry = {
        "text": text,
        'messages': [
            {"role": "system", "content": "You are a medical assistant with expertise in understanding clinical timelines and temporal relationships between medical events."},
            {"role": "user", "content": event_prompt + "\n\nCLINICAL TEXT: " + text}]
    }
    results.append(entry)
    return results





def single_turn_qa(question: str) -> str:
    """
    Function to perform a single-turn question answering using Azure OpenAI.
    """
    endpoint = "https://timelin-azure-openai.openai.azure.com/"
    model_name = "gpt-4o"
    deployment = "gpt-4o-2"

    subscription_key = "YOUR_AZURE_OPENAI_SUBSCRIPTION_KEY"
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )

    return response.choices[0].message.content





if __name__ == "__main__":
    # question = "What is the capital of France?"
    # answer = single_turn_qa(question)
    # print(f"Question: {question}")
    # print(f"Answer: {answer}")

    prompts = llm_qa.load_prompts()
    for key, value in prompts.items():
        print(f"{key}: {value}")
        print("\n"*5)

