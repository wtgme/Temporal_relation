# LLM Temporal Relation Data Builder

This directory contains scripts for building training and evaluation datasets for Large Language Models (LLMs) to learn temporal relation inference from clinical text. The output is formatted as Azure OpenAI Bulk JSONL for batch processing.

## Workflow Overview

The complete pipeline consists of four main steps:

1. **Data Construction**: Use `/home/ubuntu/work/Temporal_relation/llm_qa/llm_qa_data_builder.py` to construct datasets in Azure bulk format
2. **Azure OpenAI Processing**: Submit the bulk files to Azure OpenAI service using `/home/ubuntu/work/Temporal_relation/llm_qa/llm_azure_bulk.py`
3. **Local Inference**: Alternatively, use local vLLM for inference with `/home/ubuntu/work/Temporal_relation/llm_qa/llm_local_inference.py`
4. **Evaluation**: Evaluate the results using `/home/ubuntu/work/Temporal_relation/llm_qa/llm_evaluation.py`

## Main Script: `llm_qa_data_builder.py`

The main script `llm_qa_data_builder.py` automatically builds datasets in Azure OpenAI Bulk JSONL format for all possible configurations:

- Processing events individually or all at once
- Including or excluding time tags in the text
- Including or excluding section context information in the prompts

### Usage

```bash
python llm_qa_data_builder.py
```

The script will automatically process all 8 possible configurations:

1. Individual events without time tags
2. Individual events with time tags
3. Individual events without time tags but with section context
4. Individual events with time tags and section context
5. All events at once without time tags
6. All events at once with time tags
7. All events at once without time tags but with section context
8. All events at once with time tags and section context

### Configuration Details

The script processes the following combinations automatically:

- **Individual mode without time tags**: `--mode individual`
- **Individual mode with time tags**: `--mode individual --time_tags`
- **Individual mode with section context**: `--mode individual --section_context`
- **Individual mode with time tags and section context**: `--mode individual --time_tags --section_context`
- **All mode without time tags**: `--mode all`
- **All mode with time tags**: `--mode all --time_tags`
- **All mode with section context**: `--mode all --section_context`
- **All mode with time tags and section context**: `--mode all --time_tags --section_context`

### Default Settings

- **Data directory**: `/home/ubuntu/work/Temporal_relation/data/timeline_training/`
- **Output directory**: `/home/ubuntu/work/Temporal_relation/llm_qa/qa_data/`
- **File limit**: 50,000,000 (essentially unlimited)
- **Model**: `openai`

To modify these settings, edit the `common_args` dictionary in the `main()` function.

## Input Data

The input data directory must contain the following files for each document:
- `<file_id>.xml.label.txt`: The clinical text with event tags (used when time_tags=False)
- `<file_id>.xml.notime.label.txt`: The clinical text without time tags (used when time_tags=True)
- `<file_id>.xml.starttime.json`: A JSON file containing the start time information for the document
- `<file_id>.xml.interval_paths.json`: A JSON file containing interval path information (optional)

## Prompts File: `prompts.json`

The `prompts.json` file contains all the prompts used for different evaluation modes. The file must be located in the same directory as the script. The script automatically selects the appropriate prompt based on the configuration:

- `individual_notime`: For processing individual events without time tags
- `individual_time`: For processing individual events with time tags
- `individual_notime_section`: For processing individual events without time tags but with section context
- `individual_time_section`: For processing individual events with time tags and section context
- `all_notime`: For processing all events at once without time tags
- `all_time`: For processing all events at once with time tags
- `all_notime_section`: For processing all events at once without time tags but with section context
- `all_time_section`: For processing all events at once with time tags and section context

## Output

The script generates 8 Azure OpenAI Bulk JSONL files in the output directory, one for each configuration. Each line in the JSONL file contains a training/evaluation example with:
- `custom_id`: Unique identifier for the example
- `method`: "POST" 
- `url`: "/chat/completions"
- `body`: Contains the messages array with system prompt and user prompt

The filenames include information about the dataset configuration:
- Model name
- Whether time tags were used (`time` or `notime`)
- Whether events were processed individually or all at once (`individual` or `all`)
- Whether section context was used (`_sections` suffix)

Example output files:
- `timeline_azure_bulk_openai_notime_individual.jsonl`
- `timeline_azure_bulk_openai_time_individual.jsonl`
- `timeline_azure_bulk_openai_notime_individual_sections.jsonl`
- `timeline_azure_bulk_openai_time_individual_sections.jsonl`
- `timeline_azure_bulk_openai_notime_all.jsonl`
- `timeline_azure_bulk_openai_time_all.jsonl`
- `timeline_azure_bulk_openai_notime_all_sections.jsonl`
- `timeline_azure_bulk_openai_time_all_sections.jsonl`

## Notes

- The script automatically processes all 8 configurations in sequence
- Each configuration is processed independently with its own progress tracking
- If one configuration fails, the script continues with the remaining configurations
- When using the `all` mode, the text is not preprocessed (all event tags remain intact)
- When using the `individual` mode, the text is preprocessed to keep only the current event's tags
- Each JSONL entry represents a complete training example with input prompt for temporal relation inference
