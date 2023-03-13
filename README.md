# Neuromatch for CCN

Implement matching for CCN. This was 3 rounds of 4-people matches. Uses sentence transformers for embedding and cosine distance for comparing matches. [See this post for background](https://twitter.com/patrickmineault/status/1561454618781138945).

Run these in sequence:

* `python clean_users.py`
* `python resolve_openalex.py`
* `python resolve_coauthors.py`
* `python create_match_matrix.py`
* `python do_matching.py`
* `python generate_word_clouds.py`
* `python generate_printout.py`

You will need a data file, which contains PII, hence is not included in this repo; email me for a scrubbed version. Requires my fork of [`paper-reviewer-matcher`](https://github.com/patrickmineault/paper-reviewer-matcher). For GPT-3 based keyword inference, write the key (starting with sk) into `.openai-key`. For wordclouds, I recommend using the [Fira Sans Condensed font](https://fonts.google.com/specimen/Fira+Sans+Condensed), which you can download for free in TTF format.