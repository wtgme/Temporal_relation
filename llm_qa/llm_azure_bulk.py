import os, json, datetime
from openai import AzureOpenAI
import time
import datetime 
    
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    # azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_endpoint="https://timelin-azure-openai.openai.azure.com/",
    )

def upload_file(file_path):
    """
    Upload a file to Azure OpenAI with a specified purpose and expiration.
    
    :param file_path: Path to the file to upload.
    :param purpose: Purpose of the file (default is "batch").
    :param expires_after: Optional expiration time in seconds.
    :return: The uploaded file object.
    """

    # Upload a file with a purpose of "batch"
    file = client.files.create(
    file=open(file_path, "rb"), 
    purpose="batch",
    extra_body={"expires_after":{"seconds": 1209600, "anchor": "created_at"}} # Optional you can set to a number between 1209600-2592000. This is equivalent to 14-30 days
    )


    print(file.model_dump_json(indent=2))

    print(f"File expiration: {datetime.fromtimestamp(file.expires_at) if file.expires_at is not None else 'Not set'}")

    file_id = file.id

    # Submit a batch job with the file
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h",
        extra_body={"output_expires_after":{"seconds": 1209600, "anchor": "created_at"}} # Optional you can set to a number between 1209600-2592000. This is equivalent to 14-30 days
    )

    # Save batch ID for later use
    batch_id = batch_response.id

    print(batch_response.model_dump_json(indent=2))
    return batch_id


def track_batch_status(batch_id):
    status = "validating"
    while status not in ("completed", "failed", "canceled"):
        time.sleep(60)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

    if batch_response.status == "failed":
        for error in batch_response.errors.data:  
            print(f"Error code {error.code} Message {error.message}")
    return batch_response

def get_batch_output(batch_response, output_file_path):
    """
    Retrieve the output of a batch job by its ID.
    
    :param batch_response: The batch response object.
    :param output_file_path: Path to save the output file.
    :return: The output of the batch job.
    """
    output_file_id = batch_response.output_file_id

    if not output_file_id:
        output_file_id = batch_response.error_file_id

    if output_file_id:
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')  

        # Save to output file
        with open(output_file_path, 'w') as f:
            for raw_response in raw_responses:  
                f.write(raw_response + '\n')
        
        print(f"Output saved to: {output_file_path}")
        print(f"Total responses: {len(raw_responses)}")
        
        # Print first few responses as preview
        for i, raw_response in enumerate(raw_responses[:3]):  
            json_response = json.loads(raw_response)  
            formatted_json = json.dumps(json_response, indent=2)  
            print(f"\n--- Response {i+1} (Preview) ---")
            print(formatted_json)
    else:
        print("No output file available")

if __name__ == "__main__":
    # Test
    file_path = "/home/ubuntu/work/Temporal_relation/llm_qa/qa_data/timeline_azure_bulk_gpt-4o-mini_notime_all_sections.jsonl"
    output_path = "/home/ubuntu/work/Temporal_relation/llm_qa/qa_results/timeline_azure_bulk_gpt-4o-mini_notime_all_sections_results.jsonl"
    # batch_id = upload_file(file_path)
    batch_id = 'batch_2b6c2f71-612b-4b22-a221-9e76919d9010'
    batch_response = track_batch_status(batch_id)
    get_batch_output(batch_response, output_path)
