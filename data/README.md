Place downloaded Microsoft MIND files here. Do not commit large raw datasets to GitHub.

Expected local structure:

```text
data/
  MINDsmall_train/
    behaviors.tsv
    news.tsv
  MINDsmall_dev/
    behaviors.tsv
    news.tsv
```

The train split is used for preprocessing, automatic cluster selection, model fitting, and
persona naming. The dev split is held out and used only for validation. Small MIND-format records
under `tests/` are software fixtures, not analytical data.
