import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from nebelung.terra_workflow import TerraWorkflow
from nebelung.terra_workspace import TerraWorkspace

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("step")
    args = parser.parse_args()

    terra_workspace = TerraWorkspace(
        workspace_namespace=os.environ["WORKSPACE_NAMESPACE"],
        workspace_name=os.environ["WORKSPACE_NAME"],
        owners=json.loads(os.environ["FIRECLOUD_OWNERS"]),
    )

    terra_workflow = TerraWorkflow(
        method_namespace="my_group",
        method_name="hello_world",
        method_config_namespace="my_group",
        method_config_name="hello_world",
        method_synopsis="Trivial workflow",
        workflow_wdl_path=Path(
            os.path.join(os.path.dirname(__file__), "workflows", "hello_world.wdl")
        ).resolve(),
        method_config_json_path=Path(
            os.path.join(os.path.dirname(__file__), "workflows", "hello_world.json")
        ).resolve(),
        workflow_inputs_json_path=Path(
            os.path.join(
                os.path.dirname(__file__), "workflows", "hello_world_inputs.json"
            )
        ).resolve(),
    )

    if args.step == "upload_entities":
        samples = pd.DataFrame({"sample_id": ["a"]})
        terra_workspace.upload_entities(samples)

    elif args.step == "update_workflow":
        terra_workspace.update_workflow(terra_workflow, n_snapshots_to_keep=5)

    elif args.step == "run_workflow":
        sample_set_id = terra_workspace.create_entity_set(
            entity_type="sample", entity_ids=["a"], suffix="hello_world"
        )

        terra_workspace.submit_workflow_run(
            terra_workflow,
            entity=sample_set_id,
            etype="sample_set",
            expression="this.samples",
            use_callcache=False,
            use_reference_disks=False,
            memory_retry_multiplier=1.2,
        )

    else:
        raise NotImplementedError(f"Invalid step: {args.step}")

    logging.info("Done")
