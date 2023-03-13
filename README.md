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

TODO:

* Suggestion from Mohammed: use a UMAP, t-SNE or similar so that close-by tables discuss similar topics. This will improve table fusions
    * Allow people to visualize interests by position so that late-comers can find kindred spirits easily
* ~~Switch from davinci-003 to gpt-3.5-turbo-0301 (10X cheaper)~~
* Add caching to OpenAlex lookups (faster)

