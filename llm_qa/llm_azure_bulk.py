import os, json, datetime
from openai import AzureOpenAI
import time

def load_azure_config(config_path="azure_config.json"):
    """Load Azure OpenAI configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Please create it with your Azure OpenAI settings.")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing {config_path}. Please check the JSON format.")
        return None

# Load configuration
config = load_azure_config()
if not config:
    raise Exception("Failed to load Azure configuration")

client = AzureOpenAI(
    api_key=config["azure_openai"]["api_key"],
    api_version=config["azure_openai"]["api_version"],
    azure_endpoint=config["azure_openai"]["endpoint"],
)

def estimate_tokens(file_path):
    """Rough estimate of tokens in a file (1 token ≈ 4 characters)."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return len(content) // 4
    except:
        return 0

def get_queue_usage():
    """Get current queue usage by checking in-progress batches."""
    try:
        batches = client.batches.list(limit=100)
        total_tokens = 0
        
        for batch in batches.data:
            if batch.status in ("validating", "in_progress", "finalizing"):
                # Rough estimate - you might want to track this more precisely
                total_tokens += 1000000  # Assume 1M tokens per active batch
        
        return total_tokens
    except Exception as e:
        print(f"Warning: Could not check queue usage: {e}")
        return 0

def upload_file_with_retry(file_path, max_retries=5):
    """Upload a file to Azure OpenAI with retry logic for token limit errors."""
    for attempt in range(max_retries):
        try:
            file = client.files.create(
                file=open(file_path, "rb"), 
                purpose="batch",
                extra_body={"expires_after":{"seconds": 1209600, "anchor": "created_at"}}
            )

            batch_response = client.batches.create(
                input_file_id=file.id,
                endpoint="/chat/completions",
                completion_window="24h",
                extra_body={"output_expires_after":{"seconds": 1209600, "anchor": "created_at"}}
            )

            print(f"✅ Submitted: {os.path.basename(file_path)} -> Batch ID: {batch_response.id}")
            return batch_response.id
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a token limit error
            if "50000K" in error_msg or "enqueued tokens has surpassed" in error_msg:
                print(f"🚨 Token limit reached on attempt {attempt + 1}/{max_retries}")
                print(f"   Error: {error_msg}")
                
                if attempt < max_retries - 1:
                    wait_time = 300 + (attempt * 120)  # Increasing wait time: 5min, 7min, 9min, etc.
                    print(f"⏳ Waiting {wait_time} seconds for queue to clear...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Max retries reached for {os.path.basename(file_path)}")
                    raise e
            else:
                # For other errors, don't retry
                print(f"❌ Upload error for {os.path.basename(file_path)}: {error_msg}")
                raise e
    
    raise Exception(f"Failed to upload {file_path} after {max_retries} attempts")

def upload_file(file_path):
    """Upload a file to Azure OpenAI and submit batch job."""
    return upload_file_with_retry(file_path)

def track_batch_status(batch_id):
    """Check batch status."""
    return client.batches.retrieve(batch_id)

def get_batch_output(batch_response, output_file_path):
    """Download batch output."""
    output_file_id = batch_response.output_file_id or batch_response.error_file_id
    
    if output_file_id:
        try:
            file_response = client.files.content(output_file_id)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            with open(output_file_path, 'w') as f:
                f.write(file_response.text)
            
            print(f"✅ Downloaded: {output_file_path}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    return False

def find_all_jsonl_files(base_dir):
    """Find all .jsonl files in directory and subdirectories."""
    jsonl_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def get_output_path(input_file_path, input_base_dir, output_base_dir):
    """Generate output file path maintaining directory structure."""
    rel_path = os.path.relpath(input_file_path, input_base_dir)
    rel_dir = os.path.dirname(rel_path)
    filename = os.path.basename(rel_path)
    
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_results{ext}"
    
    if rel_dir:
        output_dir = os.path.join(output_base_dir, rel_dir)
        output_file_path = os.path.join(output_dir, output_filename)
    else:
        output_file_path = os.path.join(output_base_dir, output_filename)
    
    return output_file_path

def save_batch_tracker(batch_tracker, tracker_file="batch_tracker.json"):
    """Save batch tracking information."""
    with open(tracker_file, 'w') as f:
        json.dump(batch_tracker, f, indent=2, default=str)

def load_batch_tracker(tracker_file="batch_tracker.json"):
    """Load batch tracking information."""
    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as f:
            return json.load(f)
    return {}

def wait_for_queue_space(target_tokens=40000000):  # Wait until queue is under 40M tokens
    """Wait for queue space to become available."""
    print(f"⏳ Waiting for queue space (target: {target_tokens//1000000}M tokens)...")
    
    while True:
        try:
            batches = client.batches.list(limit=100)
            active_batches = [b for b in batches.data if b.status in ("validating", "in_progress", "finalizing")]
            
            if len(active_batches) == 0:
                print("✅ Queue is empty, proceeding...")
                return
            
            # Rough estimate of queue usage
            estimated_tokens = len(active_batches) * 1000000  # 1M tokens per batch estimate
            
            print(f"📊 Active batches: {len(active_batches)}, Estimated tokens: {estimated_tokens//1000000}M")
            
            if estimated_tokens < target_tokens:
                print("✅ Sufficient queue space available, proceeding...")
                return
            
            print(f"⏳ Queue still busy, waiting 60 seconds...")
            time.sleep(60)
            
        except Exception as e:
            print(f"Warning: Could not check queue status: {e}, proceeding anyway...")
            return

def submit_all_files():
    """Submit files in batches with queue management."""
    input_dir = "/home/ubuntu/work/Temporal_relation/llm_qa/qa_data"
    output_dir = "/home/ubuntu/work/Temporal_relation/llm_qa/qa_results"
    
    input_files = find_all_jsonl_files(input_dir)
    if not input_files:
        print(f"No .jsonl files found in {input_dir}")
        return

    batch_tracker = load_batch_tracker()
    
    # Filter files that need submission
    files_to_submit = []
    for input_file_path in input_files:
        rel_path = os.path.relpath(input_file_path, input_dir)
        output_file_path = get_output_path(input_file_path, input_dir, output_dir)
        
        if rel_path not in batch_tracker and not os.path.exists(output_file_path):
            files_to_submit.append((input_file_path, rel_path, output_file_path))

    if not files_to_submit:
        print("✅ All files already submitted or completed.")
        return

    print(f"📋 Found {len(files_to_submit)} files to submit")
    
    # Submit files one by one with error handling
    total_submitted = 0
    
    for i, (input_file_path, rel_path, output_file_path) in enumerate(files_to_submit):
        print(f"\n🚀 Submitting file {i+1}/{len(files_to_submit)}: {rel_path}")
        
        try:
            batch_id = upload_file(input_file_path)
            
            batch_tracker[rel_path] = {
                "batch_id": batch_id,
                "input_file": input_file_path,
                "output_file": output_file_path,
                "submitted_at": datetime.datetime.now().isoformat(),
                "status": "submitted"
            }
            
            total_submitted += 1
            save_batch_tracker(batch_tracker)
            
            # Short delay between successful submissions
            if i < len(files_to_submit) - 1:
                time.sleep(5)
                
        except Exception as e:
            print(f"❌ Failed to submit {rel_path}: {e}")
            # Continue with next file instead of stopping
            continue

    print(f"\n🎉 Submission complete! Total submitted: {total_submitted}/{len(files_to_submit)}")

def check_and_download_results():
    """Check status and download completed results."""
    batch_tracker = load_batch_tracker()
    
    if not batch_tracker:
        print("No batch tracker found.")
        return

    completed = 0
    failed = 0
    in_progress = 0
    downloaded = 0
    
    for rel_path, batch_info in batch_tracker.items():
        batch_id = batch_info["batch_id"]
        output_file = batch_info["output_file"]
        
        try:
            batch_response = track_batch_status(batch_id)
            status = batch_response.status
            
            print(f"📁 {rel_path}: {status}")
            
            if status == "completed":
                completed += 1
                if not os.path.exists(output_file):
                    if get_batch_output(batch_response, output_file):
                        downloaded += 1
            elif status == "failed":
                failed += 1
                print(f"   ❌ Failed - check with 'errors' command")
            elif status in ("validating", "in_progress", "finalizing"):
                in_progress += 1
                
            batch_info["status"] = status
            batch_info["checked_at"] = datetime.datetime.now().isoformat()
            
        except Exception as e:
            print(f"❌ Error checking {rel_path}: {e}")
    
    save_batch_tracker(batch_tracker)
    
    print(f"\n📊 Summary: {completed} completed, {failed} failed, {in_progress} in progress")
    print(f"📥 Downloaded this run: {downloaded}")

def check_errors():
    """Show errors for failed batches."""
    batch_tracker = load_batch_tracker()
    
    if not batch_tracker:
        print("No batch tracker found.")
        return
    
    failed_count = 0
    
    for rel_path, batch_info in batch_tracker.items():
        try:
            batch_response = track_batch_status(batch_info["batch_id"])
            
            if batch_response.status == "failed":
                failed_count += 1
                print(f"\n❌ {rel_path}")
                print(f"   Batch ID: {batch_info['batch_id']}")
                
                if hasattr(batch_response, 'errors') and batch_response.errors and batch_response.errors.data:
                    for error in batch_response.errors.data:
                        print(f"   Error: {error.message}")
                        if "50000K" in error.message or "enqueued tokens" in error.message:
                            print(f"   🚨 TOKEN LIMIT ERROR - Use 'resubmit' command")
                
        except Exception as e:
            print(f"❌ Error checking {rel_path}: {e}")
    
    if failed_count == 0:
        print("✅ No failed batches found.")

def resubmit_failed_batches():
    """Resubmit failed batches that hit token limits."""
    batch_tracker = load_batch_tracker()
    
    if not batch_tracker:
        print("No batch tracker found.")
        return
    
    # Find token limit failures
    token_limit_failures = []
    
    for rel_path, batch_info in batch_tracker.items():
        try:
            batch_response = track_batch_status(batch_info["batch_id"])
            
            if batch_response.status == "failed":
                if hasattr(batch_response, 'errors') and batch_response.errors and batch_response.errors.data:
                    for error in batch_response.errors.data:
                        if "50000K" in error.message or "enqueued tokens has surpassed" in error.message:
                            token_limit_failures.append((rel_path, batch_info))
                            break
                            
        except Exception as e:
            continue
    
    if not token_limit_failures:
        print("✅ No token limit failures found.")
        return
    
    print(f"🔄 Found {len(token_limit_failures)} token limit failures to resubmit")
    
    # Wait for queue space
    wait_for_queue_space()
    
    resubmitted = 0
    for rel_path, batch_info in token_limit_failures:
        try:
            print(f"🔄 Resubmitting: {rel_path}")
            new_batch_id = upload_file(batch_info["input_file"])
            
            batch_tracker[rel_path].update({
                "batch_id": new_batch_id,
                "status": "submitted",
                "resubmitted_at": datetime.datetime.now().isoformat()
            })
            
            resubmitted += 1
            save_batch_tracker(batch_tracker)
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ Failed to resubmit {rel_path}: {e}")
    
    print(f"✅ Resubmitted {resubmitted} files")

def cancel_batches():
    """Cancel batches that are still in progress."""
    batch_tracker = load_batch_tracker()
    
    if not batch_tracker:
        print("No batch tracker found.")
        return
    
    # Find batches that can be cancelled
    cancellable_batches = []
    
    for rel_path, batch_info in batch_tracker.items():
        try:
            batch_response = track_batch_status(batch_info["batch_id"])
            
            if batch_response.status in ("validating", "in_progress", "finalizing"):
                cancellable_batches.append((rel_path, batch_info, batch_response))
                            
        except Exception as e:
            print(f"❌ Error checking {rel_path}: {e}")
            continue
    
    if not cancellable_batches:
        print("✅ No cancellable batches found.")
        return
    
    print(f"🔍 Found {len(cancellable_batches)} batches that can be cancelled:")
    for i, (rel_path, batch_info, batch_response) in enumerate(cancellable_batches, 1):
        print(f"  {i}. {rel_path} (Status: {batch_response.status})")
    
    # Ask for confirmation
    response = input(f"\n❓ Cancel all {len(cancellable_batches)} batches? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cancellation aborted.")
        return
    
    cancelled = 0
    failed_to_cancel = 0
    
    for rel_path, batch_info, batch_response in cancellable_batches:
        try:
            print(f"🚫 Cancelling: {rel_path}")
            client.batches.cancel(batch_info["batch_id"])
            
            # Update batch tracker
            batch_tracker[rel_path].update({
                "status": "cancelled",
                "cancelled_at": datetime.datetime.now().isoformat()
            })
            
            cancelled += 1
            
        except Exception as e:
            print(f"❌ Failed to cancel {rel_path}: {e}")
            failed_to_cancel += 1
    
    save_batch_tracker(batch_tracker)
    
    print(f"\n📊 Cancellation Summary:")
    print(f"   ✅ Successfully cancelled: {cancelled}")
    print(f"   ❌ Failed to cancel: {failed_to_cancel}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "submit":
            submit_all_files()
        elif command == "check":
            check_and_download_results()
        elif command == "errors":
            check_errors()
        elif command == "resubmit":
            resubmit_failed_batches()
        elif command == "cancel":
            cancel_batches()
        else:
            print("Usage: python llm_azure_bulk.py [submit|check|errors|resubmit|cancel]")
    else:
        print("Usage: python llm_azure_bulk.py [submit|check|errors|resubmit|cancel]")
        print("Commands:")
        print("  submit   - Submit files in batches with queue management")
        print("  check    - Check status and download results")
        print("  errors   - Show error details for failed batches")
        print("  resubmit - Resubmit failed token limit batches")
        print("  cancel   - Cancel batches that are still in progress")
        # Submit will submit all files in the input directory, but not all will complete immediately
        # Check will check the status of all submitted batches and download results if available
        # Errors will show details of any failed batches, especially those that hit token limits
        # Resubmit will retry any batches that failed due to token limits, after waiting for queue
        # Cancel will cancel any batches that are still validating, in progress, or finalizing