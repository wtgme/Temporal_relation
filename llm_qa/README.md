# LLM Temporal Relation Evaluation

This directory contains scripts for evaluating the performance of Large Language Models (LLMs) in inferring temporal relations from clinical text.

## Main Script: `llm_qa.py`

The main script `llm_qa.py` is a consolidated script that supports different evaluation modes:

- Processing events individually or all at once
- Including or excluding time tags in the text
- Including or excluding section context information in the prompts

### Usage

```bash
python llm_qa.py [options]
```

#### Options

- `--mode`: Choose between `individual` (process each event separately) or `all` (process all events at once). Default: `individual`
- `--time_tags`: Include this flag to use input files with time tags. Default: False (uses files without time tags)
- `--section_context`: Include this flag to add section context information to the prompt. Default: False
- `--data_dir`: Directory containing the data files. Default: `/home/ubuntu/work/Temporal_relation/data/timeline_training/`
- `--output_dir`: Directory to save the results. Default: `/home/ubuntu/work/Temporal_relation/llm_qa/qa_results/`
- `--intermediate_dir`: Directory to save intermediate results. Default: `/home/ubuntu/work/Temporal_relation/llm_qa/intermediate_results/`
- `--limit`: Limit the number of files to process. Default: 50
- `--api_base`: Base URL for the API. Default: `http://host.docker.internal:8000/v1`
- `--model`: Model name to use. Default: `Qwen/QwQ-32B-AWQ`

#### Examples

Process individual events without time tags:
```bash
python llm_qa.py --mode individual
```

Process individual events with time tags:
```bash
python llm_qa.py --mode individual --time_tags
```

Process individual events with section context:
```bash
python llm_qa.py --mode individual --section_context
```

Process individual events with time tags and section context:
```bash
python llm_qa.py --mode individual --time_tags --section_context
```

Process all events at once:
```bash
python llm_qa.py --mode all
```

Process all events with time tags:
```bash
python llm_qa.py --mode all --time_tags
```

Process all events with section context:
```bash
python llm_qa.py --mode all --section_context
```

Process all events with time tags and section context:
```bash
python llm_qa.py --mode all --time_tags --section_context
```

## Input Data

The input data directory (`data_dir`) must contain the following files for each document:
- `<file_id>.xml.label.txt`: The clinical text with event tags.
- `<file_id>.xml.starttime.json`: A JSON file containing the start time information for the document.
- `<file_id>.xml.interval_paths.json`: A JSON file containing interval path information (optional).

## Prompts File: `prompts.json`

The `prompts.json` file contains all the prompts used for different evaluation modes. The file must be located in the same directory as the script. The script automatically selects the appropriate prompt based on the provided command line arguments:

- `individual_notime`: For processing individual events without time tags
- `individual_time`: For processing individual events with time tags
- `individual_notime_section`: For processing individual events without time tags but with section context
- `individual_time_section`: For processing individual events with time tags and section context
- `all_notime`: For processing all events at once without time tags
- `all_time`: For processing all events at once with time tags
- `all_notime_section`: For processing all events at once without time tags but with section context
- `all_time_section`: For processing all events at once with time tags and section context

## Output

The script saves results as JSON files in the specified output directory. The filename includes information about the evaluation parameters:
- Model name
- Whether time tags were used
- Whether events were processed individually or all at once
- Whether section context was used

## Notes

- When using the `all` mode, the text is not preprocessed (all event tags remain intact).
- When using the `individual` mode, the text is preprocessed to keep only the current event's tags.
- Intermediate results are automatically saved during processing to prevent data loss.
