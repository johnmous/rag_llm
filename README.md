# Building a RAG-based LMM application for querying documentation

This repository is inspired by [this](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) blog post and this [github repo](https://github.com/ray-project/llm-applications).

## App description
Input of the app is the downloaded material from our target website containing the [Django documentation](https://docs.djangoproject.com/en/5.0/). The following snippet was used:
```commandline
export EFS_DIR=/home/imoustakas/llm_playground/RAG/django_docs/
wget -e robots=off --recursive --no-clobber --page-requisites \
  --html-extension --convert-links --restrict-file-names=windows \
  --domains docs.djangoproject.com --no-parent --accept=html \
  -P $EFS_DIR https://docs.djangoproject.com/en/5.0/
```

