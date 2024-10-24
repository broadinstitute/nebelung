Nebelung: Python wrapper for the Firecloud API
---

![](https://github.com/broadinstitute/nebelung/blob/main/nebelung.jpg?raw=true)

This package provides a wrapper around the [Firecloud](https://pypi.org/project/firecloud/) package and performs a similar, though cat-themed, function as [dalmation](https://github.com/getzlab/dalmatian).

# Installation

Nebelung requires Python 3.11 or later.

```shell
poetry add nebelung # or pip install nebelung
```

# Usage

The package has two classes, `TerraWorkspace` and `TerraWorkflow`, and a variety of utility functions that wrap a subset of Firecloud API functionality.

## Workspaces

```python
from nebelung.terra_workspace import TerraWorkspace

terra_workspace = TerraWorkspace(
    workspace_namespace="terra_workspace_namespace",
    workspace_name="terra_workspace_name",
    owners=["user1@example.com", "group@firecloud.org"],
)
```

### Entities

```python
# get a workspace data table as a Pandas data frame
df = terra_workspace.get_entities("sample")

# get a workspace data table as a Pandas data frame typed with Pandera
# (`YourPanderaSchema` should subclass `nebelung.types.CoercedDataFrame`)
df = terra_workspace.get_entities("sample", YourPanderaSchema)   

# upsert a data frame to a workspace data table
terra_workspace.upload_entities(df)  # first column of `df` should be, e.g., `entity:sample_id` 

# create a sample set named, e.g., `sample_2024-08-21T17-24-19_call_cnvs"
sample_set_id = terra_workspace.create_entity_set(
    entity_type="sample",
    entity_ids=["sample_id1", "sample_id2"], 
    suffix="call_cnvs",
)
```

### Workflow outputs

```python
# collect workflow outputs from successful jobs as a list of `nebelung.types.TaskResult` objects 
outputs = terra_workspace.collect_workflow_outputs() 

# collect workflow outputs from successful jobs submitted in the last week
import datetime
a_week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
outputs = terra_workspace.collect_workflow_outputs(since=a_week_ago)
```

## Workflow

Here, a "workflow" (standard data pipeline terminology) comprises a "method" and "method config" (Terra terminology).

The standard method for making a WDL-based workflow available in a Terra workspace is to configure the git repo to push to [Dockstore](https://dockstore.org/). Although this would be the recommended technique to make a workflow available publicly, there are several drawbacks:

- The git repo must be public (for GCP-backed Terra workspaces at least).
- Every change to the method (WDL) or method config (JSON) requires creating and pushing a git commit.
- The workflow isn't updated on Dockstore immediately, since it depends on continuous deployment (CD).
- The Dockstore UI doesn't provide great visibility into CD build failures and their causes.

An alternative to Dockstore is to push the WDL directly to Firecloud. However, [that API endpoint](https://api.firecloud.org/#/Method%20Repository/post_api_methods) doesn't support uploading a WDL script that imports other local WDL scripts, nor a zip file of cross-referenced WDL scripts (like Cromwell does). The endpoint will accept WDL that imports other scripts via URLs, but currently only from the `githubusercontent.com` domain.

### Method persistence with GitHub gists

Thus, Nebelung (ab)uses [GitHub gists](https://gist.github.com/) to persist all the WDL scripts for a workflow as multiple files belonging to a single gist, then uploads the top-level WDL script's code to Firecloud. Any `import "./path/to/included/script.wdl" as other_script` statement is rewritten so that the imported script is persisted in the gist and thus imported from a `https://gist.githubusercontent.com` URL. This happens recursively, so local imports can have their own local imports.

### Method config

To aid in automation and make it easier to submit jobs manually without filling out many fields in the job submission UI, a JSON-formatted method config is also required, e.g.:

```json
{
  "deleted": false,
  "inputs": {
    "call_cnvs.sample_id": "this.sample_id"
  },
  "methodConfigVersion": 1,
  "methodRepoMethod": {
    "methodNamespace": "omics_pipelines",
    "methodName": "call_cnvs",
    "methodVersion": 1
  },
  "namespace": "omics_pipelines",
  "name": "call_cnvs",
  "outputs": {
    "call_cnvs.segs": "this.segments"
  },
  "rootEntityType": "sample"
}
```

- Both methods and method configs have their own namespaces. To simplify things, the above example uses the same sets of values for both. This approach might not be ideal if your methods and their configs are not one-to-one.
- The `TerraWorkspace.update_workflow` method will replace the `methodVersion` with an auto-incrementing version number based on the latest method's "snapshot ID" each time the method is updated. The `methodConfigVersion` should be incremented manually if desired.

### Versioning

Some information about a submitted job's method isn't easily recovered via the Firecloud API later on. Both `update_workflow` and `collect_workflow_outputs` are written to make it easier to connect workflow outputs to method versions for use in object (workflow output files and values) versioning. Include these workflow inputs in the WDL to enable this feature:

```wdl
version 1.0

workflow call_cnvs {
    input {
        String workflow_version = "1.0" # internal version number for your use
        String workflow_source_url # populated automatically with URL of this script
    }
}
```

The `update_workflow` method will automatically include these workflow inputs in the new method config's inputs, with `workflow_source_url` being set dynamically to the URL of the GitHub gist of that WDL script and `workflow_version` available for explicitly versioning the WDL.

Because GitHub gist has its own built-in versioning, a `workflow_source_url` stored in a job submission's inputs will always resolve to the exact WDL script that was used in the job, even if that method is updated later. 

### Validation

To avoid persisting potentially invalid WDL, `update_workflow` also validates all the WDL scripts with [WOMtool](https://cromwell.readthedocs.io/en/stable/WOMtool) first.

### Example

See also the [example module](https://github.com/broadinstitute/nebelung/tree/main/example) module in this repo.

```python
import os
from pathlib import Path
from nebelung.terra_workflow import TerraWorkflow

# download the latest WOMtool from https://github.com/broadinstitute/cromwell/releases
os.environ["WOMTOOL_JAR"] = "/path/to/womtool.jar"

# generate a Github personal access token (fine-grained) at 
# https://github.com/settings/tokens?type=beta
# with the "Read and Write access to gists" permission 
os.environ["GITHUB_PAT"] = "github_pat_..."

terra_workflow = TerraWorkflow(
    method_namespace="omics_pipelines", # should match `methodRepoMethod.methodNamespace` from method config
    method_name="call_cnvs", # should match `methodRepoMethod.name` from method config
    method_config_namespace="omics_pipelines", # should match `namespace` from method config
    method_config_name="call_cnvs", # should match `name` from method config
    method_synopsis="This method calls CNVs.",
    workflow_wdl_path=Path("/path/to/call_cnvs.wdl").resolve(),
    method_config_json_path=Path("/path/to/call_cnvs.json").resolve(),
    github_pat="github_pat_...", # (if not using the GITHUB_PAT ENV variable) 
    womtool_jar="/path/to/womtool.jar", # (if not using the WOMTOOL_JAR ENV variable) 
)

# create or update a workflow (i.e. method and method config) directly in Firecloud
terra_workspace.update_workflow(terra_workflow, n_snapshots_to_keep=20)

# submit a job
terra_workspace.submit_workflow_run(
    terra_workflow,
    # any arguments below are passed to `firecloud_api.create_submission`
    entity="sample_2024-08-21T17-24-19_call_cnvs", # from `create_entity_set`
    etype="sample_set", # data type of the `entity` arg
    expression="this.samples", # the root entity (i.e. the WDL expects a single sample)
    use_callcache=True,
    use_reference_disks=False,
    memory_retry_multiplier=1.2,
)
```

## Call Firecloud API directly

All calls to the Firecloud API made internally by Nebelung are retried automatically (with a backoff function) in the case of a networking-related error. This function also detects other errors returned by the API and parses the JSON response if the call was successful.

To use this functionality in the cases where Nebelung doesn't provide an endpoint wrapper, import the Firecloud API and the `call_firecloud_api` function:

```python
from firecloud import api as firecloud_api
from nebelung.utils import call_firecloud_api

# get a job submission
result = call_firecloud_api(
    firecloud_api.get_submission,
    namespace="terra_workspace_namespace",
    workspace="terra_workspace_name",
    max_retries=1,
    # kwargs for `get_submission`
    submission_id="<uuid>",
)
```

# Development

Run `pre-commit run --all-files` to automatically format your code with [Ruff](https://docs.astral.sh/ruff/) and check static types with [Pyright](https://microsoft.github.io/pyright).

To update the [package on pipy.org](https://pypi.org/project/nebelung), update the `version` in `pyproject.toml` and run `poetry publish --build`.
