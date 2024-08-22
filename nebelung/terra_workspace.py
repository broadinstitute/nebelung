import datetime
import logging
import pathlib
import tempfile
from typing import Iterable, Type, Unpack

import pandas as pd
from firecloud import api as firecloud_api

from nebelung.terra_workflow import TerraWorkflow
from nebelung.types import (
    PanderaBaseSchema,
    TaskResult,
    TerraJobSubmissionKwargs,
    TypedDataFrame,
)
from nebelung.utils import (
    batch_evenly,
    call_firecloud_api,
    expand_dict_columns,
    type_data_frame,
)


class TerraWorkspace:
    def __init__(
        self,
        workspace_namespace: str,
        workspace_name: str,
        firecloud_owners: list[str] | None = None,
    ) -> None:
        self.workspace_namespace = workspace_namespace
        self.workspace_name = workspace_name

        if firecloud_owners is None:
            self.firecloud_owners = []
        else:
            self.firecloud_owners = firecloud_owners

    def get_entities(
        self, entity_type: str, pandera_schema: Type[PanderaBaseSchema]
    ) -> TypedDataFrame[PanderaBaseSchema]:
        """
        Get a data frame of entities from a Terra data table.

        :param entity_type: the kind of entity (e.g. "sample")
        :param pandera_schema: a Pandera schema for the output data frame
        :return: a data frame of entities
        """

        logging.info(f"Getting {entity_type} entities")
        j = call_firecloud_api(
            firecloud_api.get_entities,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            etype=entity_type,
        )

        records = [{f"{entity_type}_id": x["name"], **x["attributes"]} for x in j]

        return type_data_frame(pd.DataFrame(records), pandera_schema)

    def upload_entities(self, df: pd.DataFrame) -> None:
        """
        Upload a data frame of entities to a Terra data table.

        :param df: a data frame of entities
        """

        logging.info(f"{len(df)} entities to upload to Terra")

        for batch in batch_evenly(df, max_batch_size=500):
            with tempfile.NamedTemporaryFile(suffix="tsv") as f:
                batch.to_csv(f, sep="\t", index=False)  # pyright: ignore
                f.flush()

                logging.info(f"Upserting {len(batch)} entities to Terra")
                call_firecloud_api(
                    firecloud_api.upload_entities_tsv,
                    namespace=self.workspace_namespace,
                    workspace=self.workspace_name,
                    entities_tsv=f.name,
                    model="flexible",
                )

    def create_workspace_config(self, config_body: dict) -> None:
        """
        Create a Terra workspace config for the pipeline in Terra.

        :param config_body: a dictionary containing the method config
        """

        logging.info("Creating workspace config")
        call_firecloud_api(
            firecloud_api.create_workspace_config,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            body=config_body,
        )

        logging.info("Setting workspace config ACL")
        call_firecloud_api(
            firecloud_api.update_workspace_acl,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            acl_updates=[
                {"email": x, "accessLevel": "OWNER"} for x in self.firecloud_owners
            ],
        )

    def update_workspace_config(
        self, terra_workflow: TerraWorkflow, config_body: dict
    ) -> None:
        """
        Update the Terra workspace config for a given method.

        :param terra_workflow: a `TerraWorkflow` instance
        :param config_body: a dictionary containing the method config
        """

        logging.info("Update workspace config")
        call_firecloud_api(
            firecloud_api.update_workspace_config,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            cnamespace=terra_workflow.repo_namespace,
            configname=terra_workflow.repo_method_name,
            body=config_body,
        )

        logging.info("Setting workspace config ACL")
        call_firecloud_api(
            firecloud_api.update_workspace_acl,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            acl_updates=[
                {"email": x, "accessLevel": "OWNER"} for x in self.firecloud_owners
            ],
        )

    def create_method(self, terra_workflow: TerraWorkflow) -> dict:
        """
        Create the initial method using the WDL file in this repo.

        :param terra_workflow: a `TerraWorkflow` instance
        :return: the latest method's snapshot
        """

        terra_workflow.persist_method_on_github()
        assert terra_workflow.persisted_wdl_script is not None

        with tempfile.NamedTemporaryFile("w") as f:
            f.write(terra_workflow.persisted_wdl_script["wdl"])
            f.flush()

            logging.info("Creating method")
            snapshot = call_firecloud_api(
                firecloud_api.update_repository_method,
                namespace=terra_workflow.repo_namespace,
                method=terra_workflow.repo_method_name,
                synopsis=terra_workflow.method_synopsis,
                wdl=f.name,
            )

        logging.info("Setting method ACL")
        call_firecloud_api(
            firecloud_api.update_repository_method_acl,
            namespace=terra_workflow.repo_namespace,
            method=terra_workflow.repo_method_name,
            snapshot_id=snapshot["snapshotId"],
            acl_updates=[{"user": x, "role": "OWNER"} for x in self.firecloud_owners],
        )

        return snapshot

    def update_method(self, terra_workflow: TerraWorkflow) -> dict:
        """
        Update the Terra method using a WDL file in this repo.

        :param terra_workflow: a `TerraWorkflow` instance
        :return: the latest method's snapshot
        """

        # get contents of WDL uploaded to GCS
        terra_workflow.persist_method_on_github()
        assert terra_workflow.persisted_wdl_script is not None

        with tempfile.NamedTemporaryFile("w") as f:
            f.write(terra_workflow.persisted_wdl_script["wdl"])
            f.flush()

            logging.info("Updating method")
            snapshot = call_firecloud_api(
                firecloud_api.update_repository_method,
                namespace=terra_workflow.repo_namespace,
                method=terra_workflow.repo_method_name,
                synopsis=terra_workflow.method_synopsis,
                wdl=f.name,
            )

        logging.info("Setting method ACL")
        call_firecloud_api(
            firecloud_api.update_repository_method_acl,
            namespace=terra_workflow.repo_namespace,
            method=terra_workflow.repo_method_name,
            snapshot_id=snapshot["snapshotId"],
            acl_updates=[{"user": x, "role": "OWNER"} for x in self.firecloud_owners],
        )

        return snapshot

    def update_workflow(self, terra_workflow: TerraWorkflow) -> None:
        """
        Update the Terra workflow (method and method config).

        :param terra_workflow: a `TerraWorkflow` instance
        """

        snapshots = terra_workflow.get_method_snapshots()

        # update or create the method for the current WDL file
        if len(snapshots) == 0:
            snapshot = self.create_method(terra_workflow)
        else:
            snapshot = self.update_method(terra_workflow)

        # assocate the method config with the latest method version
        terra_workflow.method_config["methodRepoMethod"]["methodVersion"] = snapshot[
            "snapshotId"
        ]

        # inject the workflow version and URL into inputs so it gets stored in job
        # submissions
        assert terra_workflow.persisted_wdl_script is not None

        if "version" in terra_workflow.persisted_wdl_script:
            terra_workflow.method_config["inputs"][
                f"{terra_workflow.repo_method_name}.workflow_version"
            ] = f'"{terra_workflow.persisted_wdl_script["version"]}"'

        terra_workflow.method_config["inputs"][
            f"{terra_workflow.repo_method_name}.workflow_url"
        ] = f'"{terra_workflow.persisted_wdl_script["public_url"]}"'

        logging.info("Checking for existing workspace config")
        res = firecloud_api.get_workspace_config(
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            cnamespace=terra_workflow.repo_namespace,
            config=terra_workflow.method_config_name,
        )

        # update or create the method config
        if res.status_code == 404:
            self.create_workspace_config(terra_workflow.method_config)
        else:
            self.update_workspace_config(terra_workflow, terra_workflow.method_config)

        # don't let old method configs accumulate
        terra_workflow.delete_old_method_snapshots(nkeep=20)

    def submit_workflow_run(
        self, terra_workflow: TerraWorkflow, **kwargs: Unpack[TerraJobSubmissionKwargs]
    ) -> None:
        """
        Submit a run of a workflow.

        :param terra_workflow: a `TerraWorkflow` instance
        """

        logging.info(f"Submitting {terra_workflow.repo_method_name} job")
        call_firecloud_api(
            firecloud_api.create_submission,
            wnamespace=self.workspace_namespace,
            workspace=self.workspace_name,
            cnamespace=terra_workflow.repo_namespace,
            config=terra_workflow.repo_method_name,
            **kwargs,
        )

    def create_sample_set(self, sample_ids: Iterable[str], suffix: str) -> str:
        """
        Create a new sample set for a list of sample IDs and upload it to Terra.

        :param sample_ids: a list of sample IDs
        :param suffix: a suffix to add to the sample set ID (e.g.
        "preprocess_wgs_sample")
        :return: the ID of the new sample set
        """

        # make an ID for the sample set of new samples
        sample_set_id = "_".join(
            [
                "samples",
                datetime.datetime.now(datetime.UTC)
                .isoformat(timespec="seconds")
                .rstrip("+00:00")
                .replace(":", "-"),
                suffix,
            ]
        )

        # construct a data frame of sample IDs for this sample set
        sample_sets = pd.DataFrame({"entity:sample_id": sample_ids}, dtype="string")
        sample_sets["entity:sample_set_id"] = sample_set_id

        logging.info("Creating new sample set in Terra")
        self.upload_entities(
            sample_sets.loc[:, ["entity:sample_set_id"]].drop_duplicates()
        )

        # construct the join table between the sample set and its samples
        sample_sets = sample_sets.rename(
            columns={
                "entity:sample_set_id": "membership:sample_set_id",
                "entity:sample_id": "sample",
            }
        )

        sample_sets = sample_sets.loc[:, ["membership:sample_set_id", "sample"]]

        logging.info(f"Adding {len(sample_sets)} samples to sample set {sample_set_id}")
        self.upload_entities(sample_sets)

        return sample_set_id

    def collect_workflow_outputs(
        self, since: datetime.datetime | None = None
    ) -> list[TaskResult]:
        """
        Collect workflow outputs and metadata about the jobs that ran them.

        :param since: don't collect outputs for job submissions before this `datetime`
        :return: a list of Gumbo `task_result` objects to insert
        """

        logging.info("Getting previous job submissions")
        submissions = pd.DataFrame(
            call_firecloud_api(
                firecloud_api.list_submissions,
                namespace=self.workspace_namespace,
                workspace=self.workspace_name,
            )
        ).convert_dtypes()

        submissions = expand_dict_columns(submissions).convert_dtypes()

        if since is not None:
            # the list of submissions can grow very long and the endpoint doesn't
            # support filtering
            submissions = submissions.loc[submissions["submissionDate"].ge(since)]

        # only collect outputs from submissions with successful workflow runs
        submissions = submissions.loc[submissions["workflowStatuses__Succeeded"].gt(0)]

        outputs = []

        for _, s in submissions.iterrows():
            sid = s["submissionId"]

            # start constructing a common output object to be used for `task_result`
            # inserts
            base_o = TaskResult(
                terra_method_config_name=str(s["methodConfigurationName"]),
                terra_method_config_namespace=str(s["methodConfigurationNamespace"]),
                terra_submission_id=str(sid),
                terra_workspace_name=self.workspace_name,
                terra_workspace_namespace=self.workspace_namespace,
            )

            # get the workflows for this job submission (often there is only 1)
            logging.info(f"Getting workflows for job submission {sid}")
            submission = call_firecloud_api(
                firecloud_api.get_submission,
                namespace=self.workspace_namespace,
                workspace=self.workspace_name,
                submission_id=sid,
            )

            for w in submission["workflows"]:
                if "workflowId" not in w:
                    # this workflow didn't manage to start
                    continue

                if "workflowEntity" not in w:
                    # this workflow didn't manage to start
                    continue

                wid = w["workflowId"]
                base_o.terra_workflow_id = wid

                if "workflowEntity" in w:
                    base_o.terra_entity_name = w["workflowEntity"]["entityName"]
                    base_o.terra_entity_type = w["workflowEntity"]["entityType"]

                base_o.created_at = pd.Timestamp(
                    ts_input=w["statusLastChangedDate"]
                ).isoformat()  # pyright: ignore

                logging.info(f"Getting workflow {wid} metadata")
                wmd = call_firecloud_api(
                    firecloud_api.get_workflow_metadata,
                    namespace=self.workspace_namespace,
                    workspace=self.workspace_name,
                    submission_id=sid,
                    workflow_id=wid,
                    include_key=[
                        "inputs",
                        "labels",
                        "outputs",
                        "status",
                        "workflowName",
                        "workflowRoot",
                    ],
                )

                if wmd["status"] != "Succeeded":
                    continue

                base_o.workflow_name = wmd["workflowName"]
                base_o.terra_workflow_inputs = wmd["inputs"]
                base_o.terra_workflow_root_dir = wmd["workflowRoot"]
                base_o.terra_workspace_id = wmd["labels"]["workspace-id"]

                # workflow URL and version should be injected into the inputs when the
                # method config is deployed
                if "workflow_source_url" in wmd["inputs"]:
                    base_o.workflow_source_url = wmd["inputs"]["workflow_source_url"]

                if "workflow_version" in wmd["inputs"]:
                    base_o.workflow_version = wmd["inputs"]["workflow_version"]

                # iterate through the outputs and make a `task_result` object for each
                for label, output in wmd["outputs"].items():
                    # all attributes up to now have been invariant between outputs
                    o = base_o.copy()

                    # the workflow outputs are named like `workflow_name.output_name`
                    o.label = label.rsplit(".", maxsplit=1)[-1]

                    if isinstance(output, list):
                        # we shouldn't be writing any workflows that output lists, so
                        # these would just be intermediate or legacy outputs
                        continue
                    elif isinstance(output, str):
                        if output.startswith("gs://"):
                            o.url = output
                            o.format = pathlib.Path(o.url).suffix[1:].upper()
                        else:
                            # it's a plain string
                            o.value = {"value": output}
                    else:
                        # it's a number, bool, or something else
                        o.value = {"value": output}

                    outputs.append(o)

        return outputs
