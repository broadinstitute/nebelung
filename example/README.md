Hello World example
---

This module contains example commands to create a sample data table and trivial workflow, then run that workflow on a new sample set. Your environment must have these environment variables defined:

```
WORKSPACE_NAMESPACE="terra_workspace_namespace"
WORKSPACE_NAME="terra_workspace_name"
FIRECLOUD_OWNERS=["your_terra_user@example.com"]
GITHUB_PAT="github_pat_..."
WOMTOOL_JAR="/path/to/womtool.jar"
```

To test Nebelung's functionality in a new Terra workspace:

```shell
# 1. create a sample data table with a single sample
poetry run python -m example upload_entities
# confirm the workspace has a `sample` data table with a single row

# 2. create/update the hello_world workflow (method and method config) 
poetry run python -m example update_workflow
# confirm the workspace has a `hello_world` workflow with an imported GitHub gist file

# 3. run the workflow on a new sample set comprising the one sample
poetry run python -m example run_workflow
# confirm the job succeeded and a `result` column is added to the `sample` data table
```
