## Experiment Results
### Stage 1
#### 1a_tougher
| S.No | Matching Input       | Model Used           | Accuracy | F1 Score | Remarks |
|------|----------------------|----------------------|----------|----------|---------|
| 1    | Terms                | all-mpnet-base-v2    | 0.29     | 0.23     |         |
| 2    | Desc                 | all-mpnet-base-v2    | 0.35     | 0.252    |         |
| 3    | Terms + Desc         | all-mpnet-base-v2    | 0.36     | 0.29     |         |
| 4    | Terms + Desc         | bge-base-en-v1.5     | 0.31     | 0.23     |         |
| 5    | Terms + Desc         | bge-large-en-v1.5    | 0.36     | 0.29     |         |
| 6    | Terms + Desc         | mxbai-embed-large-v1 | 0.38     | 0.32     |         |
| 7    | Terms + Desc         | bge-large-en-v1.5    | 0.39     | 0.325    | added 'defined as' as stitching term |
| 8    | Terms + Desc         | all-mpnet-base-v2    | 0.39     | 0.318    | added 'is' as stitching term |
| 9    | Terms + Desc         | bge-large-en-v1.5    | 0.41     | 0.341    | added 'can be defined as' as stitching term |
| 10   | Terms + Desc         | bge-large-en-v1.5    | 0.41     | 0.341    | added re-ranker: didn't help |
| 11   | Terms + Desc         | bge-large-en-v1.5    | 0.46     | 0.355    | miniLM re-ranker finetuned on hard-neg, noisy queries |
| 12   | Terms + Desc         | bge-large-en-v1.5    | **0.49**     | 0.40     | sno 11 + reranker trained with glossary desc |
| 13   | Terms + Desc         | bge-large-en-v1.5    | 0.46     | 0.34     | sno 12 + reranker trained with simpler data |


#### 1a_simpler
| S.No | Matching Input       | Model Used           | Accuracy | F1 Score | Remarks |
|------|----------------------|----------------------|----------|----------|---------|
| 1    | Terms + Desc         | bge-large-en-v1.5    | 0.863    | 0.845    |         |
| 2    | Terms + Desc         | bge-large-en-v1.5    | 0.954    | 0.941    | With re-ranker, top5 |
| 3    | Terms + Desc         | bge-large-en-v1.5    | 0.962    | 0.95     | With re-ranker, top10 |

#### 1a_simpler_noisy  
- This contains natural spelling, grammatical mistakes

| S.No | Matching Input       | Model Used           | Accuracy | F1 Score | Remarks |
|------|----------------------|----------------------|----------|----------|---------|
| 1    | Terms + Desc         | bge-large-en-v1.5    | 0.9688   | 0.9583   | With re-ranker, top10 |
| 2    | Terms + Desc         | bge-large-en-v1.5    | 0.89     | 0.86   | miniLM re-ranker finetuned on hard-neg, noisy queries |
| 3    | Terms + Desc         | bge-large-en-v1.5    | **0.87**     | 0.85   | sno 2 + glossary data 4b with desc |
| 4    | Terms + Desc         | bge-large-en-v1.5    | 0.82     | 0.78   | sno 3 + finetuned on simpler data too |

#### 1a_balanced  
- This contains tougher interpreted queries, direct queries, and natural spelling/ grammatical mistakes: ~1k queries

| S.No | Matching Input       | Model Used           | Accuracy | F1 Score | Remarks |
|------|----------------------|----------------------|----------|----------|---------|
| 1    | Best setting #12 in 1a_noisy       | bge-large-en-v1.5    | 0.8746   | 0.8626   | miniLM retrained, top 5 used|
| 2    | Best setting #12 in 1a_noisy       | bge-large-en-v1.5    | 0.8973   | 0.8849   | miniLM retrained, top 10 used|
| 2b    | model in no. 2       | bge-large-en-v1.5    | 0.69   | 0.62   | Only Eval on only tough set |
| 3    |    Same as 2    | bge-large-en-v1.5    | **0.926**   | **0.911**   | reranker changed to BGE |


### Stage 2
#### 1b_clean_data
| S.No | Glossary Input       | Model Used           | Accuracy | F1 Score | Remarks |
|------|----------------------|----------------------|----------|----------|---------|
| 1    | Term                 | all-mpnet-base-v2    | 0.41     | 0.26     |         |
| 2    | Term                 | bge-large-en-v1.5    | 0.48     | 0.32     |         |
| 3    | Term + Desc          | bge-large-en-v1.5    | 0.44     | 0.29     |         |
| 4    | Term                 | bge-large-en-v1.5    | 0.38     | 0.23     | With reranker |
| 5    | Term                 | bge-large-en-v1.5    | 0.51     | 0.35     | With reranker fine-tuned** |
| 6    | Term                 | bge-large-en-v1.5    | 0.55     | 0.38     | reranker miniLM fine-tuned + weights b/w ranker & sim |
| 7    | Term                 | bge-large-en-v1.5    | 0.62     | 0.45     | reranker miniLM fine-tuned on more hard-negative + weights b/w ranker & sim  |

#### higher quality data

| S.No | Description       | Model Used           | Accuracy | F1 Score | 
|------|----------------------|----------------------|----------|----------|
| 1    | Using top-5 failures for re-ranker | bge-large-en-v1.5    | 0.68     | 0.53     |
| 2    | Using top-10 failures for re-ranker | bge-large-en-v1.5    | 0.56     | 0.40    |
| 3    | Only using embedding model | bge-large-en-v1.5    | **0.76**     | 0.64     |

- Top-K misclassifications in re-ranker's negative set
- Ratios removed

### Period
| S.No | Method Used          | Model Used           | Accuracy           | F1 Score         | Remarks |
|------|----------------------|----------------------|--------------------|------------------|---------|
| 1    | Regex                | Regex Matching       | 0.00               | 0.00             |         |
| 2    | Semantic             | bge-large-en-v1.5    | Overall- 0.48, View- 0.67     | Overall-0.34     | Used semantic matching with basic prototypes of PRD and FTP, like for the period and to date        |
| 3    | Semantic + Rule          | bge-large-en-v1.5    | Overall- 0.48, View- 0.67     | Overall-0.34     | used rules-based system along with semantic to improve the sequence ,but no improvement        |
| 4    | Extended Prototype + Semantic fall back                 | bge-large-en-v1.5    | Overall- 0.55, View- 0.77    | Overall 0.39     | Extended prototype list with more keywords, first the matching of the word from the list, then semantic fallback |
| 5    | Above + Verbose                 | bge-large-en-v1.5    | Overall- 0.69, View- 1.00     | Overall 0.54     | added verbosity like FTP and PRd "can be defined as" in the prototypes  |
| 6    | Above + fuzzy matching                 | bge-large-en-v1.5    | Overall- 1.00, View- 1.00, Sequence- 1.00    | Overall 1     | improved the regex in which `calendar.month_name[0]` is the empty string, and added fuzzy matching for the sequence""  |

## Summary of Data (14/06/25)
1. **Stage 1:** NL Query to Glossary
    1. NL Query: We generated (~1k queries)
    2. Glossary (with description): 64 terms
2. **Stage 2:** Glossary to Grouping Label
    1. Labeled and given: 30
        1. Removed formulae - Multiple queries
    2. Final list: 25

## Ongoing Work-List
- Stage 1:
  - Scenario Detection: Actual/Budget/Cashflow
- Stage 2: 
  - Add in type, etc. in the matching queries.
- Multi-SQL query generation
  - Ratio Computation (basis glossary)
  - Comparison queries

## Files
nl2glossary.py: 
- natural language to glossary
- i/p, 1: queries_and_glossary.csv (generated by us: query and GT glossary)
- i/p, 2: glossary_v1.csv (given by finalyser: glossary and description)
- o/p: nl2glossary.csv (storing csv for gt, pred)

glossary2grouping.py
- i/p, 1: nl2glossary.csv (glossary, GT grouping lables & id)
- You can use **glossary2label.csv** as this file contains the glossary, GT grouping_label, GT grouping_id for better testing.
  - As the Glossary in this one has the correct grouping_label.
  - Check the columns' names, as they might be different.
  - For example: Glossary -> glossary, GT grouping_label -> ground_label. 
- i/p, 2: fbi_groping_master.csv (grouping labels, id)
- o/p: glossary2label_results

metric.py
- I/p: Results file from respective step
- O/P: Generate Metrics for given step
<br/>

glossary2label.csv
- Contains the Glossary, ground truth grouping_label, GT grouping_id.
- Columns: glossary, ground_label, grouping_id
<br/>

nl2glossary.csv
- Contains Natural Query, GT Glossary, the predicted Glossary, and the similarity score.
- Columns: NL_Query,	GT_Glossary,	Predicted_Glossary,	Similarity_Score
<br/>

glossary2label_results.csv
- Contains the Glossary, it GT grouping_label, GT grouping_id, Predicted grouping_label, Predicted grouping_id.
- Columns: glossary,	ground_label,	grouping_id,	Predicted_Label,	Predicted_Grouping_ID


<br/>



## Quantitative results
- nl2glossary.csv -> Contains Natural Query, Ground Truth Glossary, and the Predicted Glossary.
- glossary2label_results.csv -> Contains Glossary, Ground Truth grouping_label, Ground Truth grouping_id, Predicted grouping_label, Predicted grouping_id
<br/

## Run locally
- Use the app_v2.py file for running the model locally.
- Use the PEM file you have to connect it to the database.

