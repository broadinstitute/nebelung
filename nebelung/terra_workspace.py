import datetime
import logging
import pathlib
from io import StringIO
from typing import Any, Iterable, Type, Unpack

import pandas as pd
from firecloud import api as firecloud_api
from pd_flatten import pd_flatten

from nebelung.terra_workflow import TerraWorkflow
from nebelung.types import (
    PanderaBaseSchema,
    Submissions,
    SubmittableEntities,
    SubmittedEntities,
    TaskResult,
    TerraJobSubmissionKwargs,
    TypedDataFrame,
)
from nebelung.utils import batch_evenly, call_firecloud_api, type_data_frame


class TerraWorkspace:
    def __init__(
        self,
        workspace_namespace: str,
        workspace_name: str,
        owners: list[str] | None = None,
    ) -> None:
        self.workspace_namespace = workspace_namespace
        self.workspace_name = workspace_name
        self.owners = [] if owners is None else owners

    def get_entities(
        self,
        entity_type: str,
        pandera_schema: Type[PanderaBaseSchema] | None = None,
    ) -> pd.DataFrame | TypedDataFrame[PanderaBaseSchema]:
        """
        Get a data frame of entities from a Terra data table.

        :param entity_type: the kind of entity (e.g. "sample")
        :param pandera_schema: an optional Pandera schema for the output data frame
        :return: a data frame of entities
        """

        logging.info(f"Getting {entity_type} entities from {self.workspace_name}")
        j = call_firecloud_api(
            firecloud_api.get_entities,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            etype=entity_type,
        )

        records = [{f"{entity_type}_id": x["name"], **x["attributes"]} for x in j]
        df = pd.DataFrame(records)

        if pandera_schema is None:
            return df

        return type_data_frame(df, pandera_schema)

    def upload_entities(
        self, df: pd.DataFrame, model: str = "flexible", delete_empty: bool = True
    ) -> None:
        """
        Upload a data frame of entities to a Terra data table.

        :param df: a data frame of entities
        :param model: the entity upload model to use ("flexible" is recommended over
        "firecloud")
        :param delete_empty: whether an empty cell in the data frame should clear the
        corresponding value in the Terra data table
        """

        logging.info(f"{len(df)} entities to upload to {self.workspace_name}")
        buffer = StringIO()  # store batches of TSV rows in this buffer

        for batch in batch_evenly(df, max_batch_size=500):
            logging.info(f"Upserting {len(batch)} entities to {self.workspace_name}")

            # write the latest batch of TSV rows to the emptied buffer
            buffer.seek(0)
            buffer.truncate(0)
            batch.to_csv(buffer, sep="\t", index=False)  # pyright: ignore

            call_firecloud_api(
                firecloud_api.upload_entities,
                namespace=self.workspace_namespace,
                workspace=self.workspace_name,
                entity_data=buffer.getvalue(),
                model=model,
                delete_empty=delete_empty,
            )

    def delete_entities(self, entity_type: str, entity_ids: set[str]) -> None:
        """
        Delete entities from a Terra workspace, including their related entities.

        :param entity_type: the type of entity to delete
        :param entity_ids: a set of entity IDs to delete
        """

        if len(entity_ids) == 0:
            logging.info(f"No entities in {self.workspace_name} to delete")
            return

        # get all entities
        all_entities = call_firecloud_api(
            firecloud_api.get_entities_with_type,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
        )

        for x in all_entities:
            if x["entityType"] == entity_type:
                # we're looking for entities that might *reference* the entity type
                # we're deleting, not this entity type itself, which we'll delete
                # directly later
                continue

            if "attributes" in x:
                x2 = x.copy()
                x_updated = False

                for k, v in x["attributes"].items():
                    if not (
                        isinstance(v, dict)
                        and "itemsType" in v
                        and v["itemsType"] == "EntityReference"
                    ):
                        # this is some other irrelevant entity type
                        continue

                    items = v["items"]

                    if len(items) == 0:
                        continue

                    # remove entity IDs from the join table
                    updated_items = [
                        y
                        for y in items
                        if y["entityType"] == entity_type
                        and y["entityName"] not in entity_ids
                    ]

                    if len(items) == len(updated_items):
                        continue

                    # we changed the list of referenced items
                    x2["attributes"][k]["items"] = updated_items
                    x_updated = True

                if x_updated:
                    logging.info(
                        f"Removing entities from {x2['name']} in {self.workspace_name}"
                    )

                    attribute_name = list(x2["attributes"].keys())[0]

                    # update the entity for the join table
                    _ = call_firecloud_api(
                        firecloud_api.update_entity,
                        namespace=self.workspace_namespace,
                        workspace=self.workspace_name,
                        etype=x2["entityType"],
                        ename=x2["name"],
                        updates=[
                            {
                                "op": "AddUpdateAttribute",
                                "attributeName": attribute_name,
                                "addUpdateAttribute": x2["attributes"][attribute_name],
                            }
                        ],
                    )

        # now that we've deleted related entities, delete the entities themselves
        call_firecloud_api(
            firecloud_api.delete_entities,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            json_body=[
                {"entityType": entity_type, "entityName": x} for x in entity_ids
            ],
        )

    def create_entity_set(
        self,
        entity_type: str,
        entity_ids: Iterable[str],
        entity_set_id: str | None = None,
        suffix: str | None = None,
    ) -> str:
        """
        Create a new entity set for a list of entity IDs and upload it to Terra.

        :param entity_type: the kind of entity (e.g. "sample")
        :param entity_ids: a list of entity IDs
        :param entity_set_id: an optional ID for the new entity set (if `suffix` isn't
        provided)
        :param suffix: a suffix to add to an auto-generated, timestamped, entity set ID
        (if `entity_set_id` isn't provided)
        :return: the ID of the new entity set
        """

        if entity_set_id is None:
            assert suffix is not None, (
                "suffix is required if you don't specify a entity set ID"
            )

            # generate an ID for the entity set of new entities
            entity_set_id = "_".join(
                [
                    entity_type,
                    datetime.datetime.now(datetime.UTC)
                    .isoformat(timespec="seconds")
                    .rstrip("+00:00")
                    .replace(":", "-"),
                    suffix,
                ]
            )

        elif suffix is not None:
            logging.warning(
                "An entity_set_id was provided, so the suffix option is ignored"
            )

        # construct a data frame of entity IDs for this entity set
        entity_set = pd.DataFrame(
            {f"entity:{entity_ids}_id": entity_ids}, dtype="string"
        )
        entity_set[f"entity:{entity_type}_set_id"] = entity_set_id

        logging.info(f"Creating new {entity_type} set in {self.workspace_name}")
        self.upload_entities(
            entity_set.loc[:, [f"entity:{entity_type}_set_id"]].drop_duplicates()
        )

        # construct the join/membership table between the entity set and its entities
        entity_set = entity_set.rename(
            columns={
                f"entity:{entity_type}_set_id": f"membership:{entity_type}_set_id",
                f"entity:{entity_ids}_id": entity_type,
            }
        )

        entity_set = entity_set.loc[
            :, [f"membership:{entity_type}_set_id", entity_type]
        ]

        logging.info(
            f"Adding {len(entity_set)} {entity_type} entities "
            f"to {entity_type} set {entity_set_id} in {self.workspace_name}"
        )
        self.upload_entities(entity_set)

        return entity_set_id

    def create_method_config(self, config_body: dict) -> None:
        """
        Create a Terra method config.

        :param config_body: a dictionary containing the method config
        """

        logging.info("Creating method config")
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
            acl_updates=[{"email": x, "accessLevel": "OWNER"} for x in self.owners],
        )

    def update_method_config(
        self, terra_workflow: TerraWorkflow, config_body: dict
    ) -> None:
        """
        Update a Terra method config.

        :param terra_workflow: a `TerraWorkflow` instance
        :param config_body: a dictionary containing the method config
        """

        logging.info("Update method config")
        call_firecloud_api(
            firecloud_api.update_workspace_config,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            cnamespace=terra_workflow.method_config_namespace,
            configname=terra_workflow.method_config_name,
            body=config_body,
        )

        logging.info("Setting workspace config ACL")
        call_firecloud_api(
            firecloud_api.update_workspace_acl,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            acl_updates=[{"email": x, "accessLevel": "OWNER"} for x in self.owners],
        )

    def update_workflow(
        self, terra_workflow: TerraWorkflow, n_snapshots_to_keep: int = 20
    ) -> None:
        """
        Update the Terra workflow (method and method config).

        :param terra_workflow: a `TerraWorkflow` instance
        :param n_snapshots_to_keep: the number of method snapshots to keep
        """

        # update or create the method
        snapshot = terra_workflow.update_method(self.owners)

        # assocate the method config with the latest method version
        terra_workflow.method_config["methodRepoMethod"]["methodVersion"] = snapshot[
            "snapshotId"
        ]

        # inject the workflow version and URL into inputs so they get stored in job
        # submissions
        assert terra_workflow.persisted_wdl_script is not None

        if "version" in terra_workflow.persisted_wdl_script:
            terra_workflow.method_config["inputs"][
                f"{terra_workflow.method_name}.workflow_version"
            ] = f'"{terra_workflow.persisted_wdl_script["version"]}"'

        terra_workflow.method_config["inputs"][
            f"{terra_workflow.method_name}.workflow_source_url"
        ] = f'"{terra_workflow.persisted_wdl_script["public_url"]}"'

        logging.info("Checking for existing method config")
        # all of the `firecloud_api.*_workspace_config` methods are really operations on
        # a method config, just inside a workspace
        res = firecloud_api.get_workspace_config(
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            cnamespace=terra_workflow.method_config_namespace,
            config=terra_workflow.method_config_name,
        )

        # update or create the method config
        if res.status_code == 404:
            self.create_method_config(terra_workflow.method_config)
        else:
            self.update_method_config(terra_workflow, terra_workflow.method_config)

        # don't let old method versions accumulate
        terra_workflow.delete_old_method_snapshots(
            n_snapshots_to_keep=n_snapshots_to_keep
        )

    def check_submittable_entities(
        self,
        entity_type: str,
        entity_ids: Iterable[str],
        terra_workflow: TerraWorkflow,
        since: datetime.datetime | None = None,
        resubmit_n_times: int = 0,
        force_retry: bool = False,
    ) -> SubmittableEntities:
        """
        Check a list of entity IDs that might have been submitted to a particular
        workflow and determine whether they eligible to be (re)submitted.

        :param entity_type: the kind of entity (e.g. "sample")
        :param entity_ids: a list of entity IDs
        :param terra_workflow: a workflow to potentially create a job for
        :param since: don't collect job submissions before this `datetime`
        :param resubmit_n_times: the number of times to resubmit a failed entity to a
        job
        :param force_retry: allow resubmission even if an entity has failed more than
        `resubmit_n_times` for that workflow
        :return: a dictionary of statuses and entity IDs belong to them
        """

        # get all submissions in the workspace
        submissions = type_data_frame(
            pd.DataFrame(
                call_firecloud_api(
                    firecloud_api.list_submissions,
                    namespace=self.workspace_namespace,
                    workspace=self.workspace_name,
                )
            ),
            Submissions,
        )

        # filter submissions to those for the TerraWorkflow of interest
        subs_for_workflow = submissions.loc[
            submissions["methodConfigurationNamespace"].eq(
                terra_workflow.method_config_namespace
            )
            & submissions["methodConfigurationName"].str.startswith(
                terra_workflow.method_config_name
            ),  # Terra appends a random string to the method config used in a job
            ["submissionDate", "submissionId"],
        ]

        subs_for_workflow["submissionDate"] = pd.to_datetime(
            subs_for_workflow["submissionDate"]
        )

        if since is not None:
            # the list of submissions can grow very long and the endpoint doesn't
            # support filtering
            subs_for_workflow = subs_for_workflow.loc[
                subs_for_workflow["submissionDate"].ge(pd.Timestamp(since, tz="UTC"))
            ]

        # collect the entities submitted as part of this job and their status
        submitted_entities = []

        for sid in subs_for_workflow["submissionId"]:
            submission = call_firecloud_api(
                firecloud_api.get_submission,
                namespace=self.workspace_namespace,
                workspace=self.workspace_name,
                submission_id=sid,
            )

            for w in submission["workflows"]:
                if "workflowEntity" not in w:
                    # this workflow didn't manage to start
                    continue

                submitted_entities.append(
                    {
                        "entity_type": w["workflowEntity"]["entityType"],
                        "entity_id": w["workflowEntity"]["entityName"],
                        "status": w["status"],
                    }
                )

        # make data frame of previously submitted entities
        sub_ent_df = type_data_frame(
            pd.DataFrame(submitted_entities), SubmittedEntities
        )

        sub_ent_df = sub_ent_df.loc[
            sub_ent_df["entity_type"].eq(entity_type)
            & sub_ent_df["entity_id"].isin(list(entity_ids))
        ]

        # count number of failures per entity
        failure_counts = (
            sub_ent_df.loc[sub_ent_df["status"].eq("Failed"), "entity_id"]
            .value_counts()
            .reset_index()
        )

        if force_retry:
            # we aren't returning counts, so just zero them out to simplify logic
            failure_counts["count"] = 0

        return {
            # if an input entity ID isn't in the history, it hasn't been submitted yet
            "unsubmitted": set(
                [x for x in entity_ids if x not in sub_ent_df["entity_id"]]
            ),
            # a running entity might have state 'Queued', 'Submitted', 'Launching',
            # 'Running', or anything that's not one of the below
            "running": set(
                sub_ent_df.loc[
                    ~sub_ent_df["status"].isin(
                        ["Aborted", "Aborting", "Succeeded", "Failed"]
                    ),
                    "entity_id",
                ]
            ),
            # failed tasks are potentially retryable based on above logic
            "retryable": set(
                failure_counts.loc[
                    failure_counts["count"].le(resubmit_n_times), "entity_id"
                ]
            ),
            "failed": set(
                failure_counts.loc[
                    failure_counts["count"].gt(resubmit_n_times), "entity_id"
                ]
            ),
        }

    def get_workflow_config(self, terra_workflow: TerraWorkflow) -> dict[str, Any]:
        """
        Get the method configuration for a workflow in this workspace.

        :param terra_workflow: a `TerraWorkflow` instance
        """

        return call_firecloud_api(
            firecloud_api.get_workspace_config,
            namespace=self.workspace_namespace,
            workspace=self.workspace_name,
            cnamespace=terra_workflow.method_config_namespace,
            config=terra_workflow.method_config_name,
        )

    def submit_workflow_run(
        self, terra_workflow: TerraWorkflow, **kwargs: Unpack[TerraJobSubmissionKwargs]
    ) -> None:
        """
        Submit a run of a workflow.

        :param terra_workflow: a `TerraWorkflow` instance
        """

        logging.info(
            f"Submitting {terra_workflow.method_name} job in {self.workspace_name}"
        )
        call_firecloud_api(
            firecloud_api.create_submission,
            wnamespace=self.workspace_namespace,
            workspace=self.workspace_name,
            cnamespace=terra_workflow.method_config_namespace,
            config=terra_workflow.method_config_name,
            **kwargs,
        )

    def collect_workflow_outputs(
        self, since: datetime.datetime | None = None
    ) -> list[TaskResult]:
        """
        Collect workflow outputs and metadata about the jobs that ran them.

        :param since: don't collect outputs for job submissions before this `datetime`
        :return: a list of Gumbo `task_result` objects to insert
        """

        logging.info(f"Getting previous job submissions in {self.workspace_name}")
        submissions = type_data_frame(
            pd.DataFrame(
                call_firecloud_api(
                    firecloud_api.list_submissions,
                    namespace=self.workspace_namespace,
                    workspace=self.workspace_name,
                )
            ),
            Submissions,
        )

        submissions = pd_flatten(submissions).convert_dtypes()

        submissions["submissionDate"] = pd.to_datetime(submissions["submissionDate"])

        if since is not None:
            # the list of submissions can grow very long and the endpoint doesn't
            # support filtering
            submissions = submissions.loc[
                submissions["submissionDate"].ge(pd.Timestamp(since, tz="UTC"))
            ]

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
                base_o.terra_entity_name = w["workflowEntity"]["entityName"]
                base_o.terra_entity_type = w["workflowEntity"]["entityType"]

                base_o.completed_at = pd.Timestamp(
                    ts_input=w["statusLastChangedDate"], tz="UTC"
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
                    o = base_o.model_copy()

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
