# Temporal_relation

This repository is to extract temporal relations from clinical text notes, using the n2c2 2012 dataset. 
The first task is predict the value of "sec_time_rel" attribute for an EVENT.

## Background on "sec_time_rel" relations

As detailed in the [annotation guidelines](https://portal.dbmi.hms.harvard.edu/projects/download_dataset/?file_uuid=9af140cb-3452-41f0-bf79-9460c0ec94f2): 
"The sec_time_rel attribute stands for “section time relation”. Section time refers to the section creation time, that is, the time at which that section of the record was written. More specifically, the discharge summary records in this annotation project contain two sections that require annotation: they are the clinical history section and the hospital course section. We define the section time of the clinical history section to be the date of admission because usually this section is written on the admission date. The section time of the hospital course section is defined to be the discharge date for the same reason.

For each EVENT, the sec_time_rel attribute records whether the EVENT happens before, after or whether it overlaps with the section creation time. That is, for every EVENT in the history section, we will mark whether it happened before, after or whether it overlapped with the admission date, and for every EVENT in the hospital course section, we will mark whether it happened before, after or whether it overlapped with the discharge date.

The sec_time_rel attribute is, in fact, a temporal relation. It links every EVENT to either the admission date, or the discharge date, depending on which section this EVENT appears in the text. This attribute is very important, and it does not have a default value. The annotator must select a sec_time_rel for each EVENT. It must not be left blank.

The four attribute values that the sec_time_rel attribute can take, are BEFORE, AFTER, OVERLAP and BEFORE_OVERLAP. The definitions of the values are shown in Figure 7. 
"

# Files
*data_loader.ipynb*: This script loads the original XML data into CSV formats to make the data easier to process. This results in two files for training (from the "2012-07-15.original-annotation.release" file) and test data (from the "2012-08-23.test-data.groundtruth/ground_truth/merged_xml/" files) respectively:

 1. "train/test_text.csv" contains the text content for training or test data. Each row contains a document ID (i.e., file name like "1.xml" in the original dataset) and text content (i.e., the "TEXT" section) of a discharge document.
 2.  "train/test_sec_time_rel_sentence.csv" contains all "sec_time_rel" temporal links within training or test data. Each row is a "sec_time_rel" relation between an event and Admission/Discharge date, i.e., those "TLINK" tags with id="SectimeXX" in the original XML files. The original "TLINK" tags only contain the IDs of an EVENT and a temporal expression (tagged with "TIMEX3"). This new CSV file retrieved all information about an EVENT and temporal expression by joining the relevant tags together. The suffixes "_TLINK", "_EVENT" and "_TIMEX3" of column names label the original source of a column, from a "TLINK", "EVENT" or "TIMEX3" tag. Several additional fields are added to provide more context, "doc_id" labels the name of the original XML file, "adm_dis_TIMEX3" labels whether a temporal expression is about DISCHARGE or ADMISSION, and "sentence" provides the content of a sentence where the relation is extracted. 
 
 # Task
 To train a classifier to predict the "type_TLINK" between a given event "fromID_TLINK" and "adm_dis_TIMEX3". 
 Apart from the sentence where a relation was mentioned, it can be useful to consider the context of the sentence (e.g., document structure or adjacent sentences), which can be extracted from the "train/test_text.csv" file. To identify the context of a EVENT in the original document, "start_EVENT" and "end_EVENT", which indicate the starting and end character indexes of an EVENT mentioned in a document, can be useful.
