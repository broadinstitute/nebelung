import datetime
import logging
import pathlib
from io import StringIO
from typing import Any, Iterable, Type, Unpack

import numpy as np
import pandas as pd
from firecloud_api_cds import api as firecloud_api
from pd_flatten import pd_flatten

from nebelung.terra_workflow import TerraWorkflow
from nebelung.types import (
    EntityStateCounts,
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

    def count_entity_workflow_states(
        self,
        entity_type: str,
        entity_ids: Iterable[str],
        terra_workflow: TerraWorkflow,
        since: datetime.datetime | None = None,
    ) -> TypedDataFrame[EntityStateCounts]:
        """
        Get counts of workflow states for a workflow and a list of entity IDs.

        :param entity_type: the kind of entity (e.g. "sample")
        :param entity_ids: a list of entity IDs
        :param terra_workflow: a workflow to potentially create a job for
        :param since: don't collect job submissions before this `datetime`
        :return: a data frame of entity IDs and counts of workflow states:
            - queued
            - submitted
            - launching
            - running
            - aborted
            - aborting
            - succeeded
            - failed
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

        # subset to entities in the list
        sub_ent_df = sub_ent_df.loc[
            sub_ent_df["entity_type"].eq(entity_type)
            & sub_ent_df["entity_id"].isin(list(entity_ids))
        ]

        sub_ent_df["status"] = sub_ent_df["status"].str.lower()

        # make wide data frame of workflow state counts per entity
        state_counts_obs = (
            sub_ent_df.value_counts(["entity_id", "status"])
            .reset_index()
            .pivot(index="entity_id", columns="status", values="count")
        ).reset_index()

        # ensure there are rows for entities that have never been submitted
        state_counts = type_data_frame(
            pd.DataFrame({"entity_id": entity_ids}).merge(
                state_counts_obs, on="entity_id", how="left"
            ),
            EntityStateCounts,
        )

        return state_counts

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

    def submit_delta_job(
        self,
        terra_workflow: TerraWorkflow,
        entity_type: str,
        entity_set_type: str,
        entity_id_col: str,
        expression: str,
        input_cols: set[str] | None = None,
        output_cols: set[str] | None = None,
        resubmit_n_times: int = 0,
        force_retry: bool = False,
        use_callcache: bool = True,
        use_reference_disks: bool = False,
        memory_retry_multiplier: float = 1.0,
        max_n_entities: int | None = None,
        dry_run: bool = False,
    ):
        """
        Identify entities in a Terra data table that need to have a workflow run on them
        by:

            1. checking for the presence of workflow inputs and outputs in data table
               columns
            2. confirming the entity is eligible to be submitted in a job by checking
               for previous submissions of that same entity to the workflow

        :param terra_workflow: a TerraWorkflow instance for the method
        :param entity_type: the name of the Terra entity type
        :param entity_set_type: the name of the Terra entity set type for `entity_type`
        :param entity_id_col: the name of the ID column for the entity type
        :param expression: the entity type expression (e.g. "this.samples")
        :param input_cols: the set of column names that must all be present in the
        entity type in order for an entity to be submittable
        :param output_cols: the set of column names that must all be missing in the
        entity type in order for an entity to be submittable
        :param resubmit_n_times: the number of times to resubmit an entity in the event
        it has failed in the past
        :param force_retry: whether to retry even if `resubmit_n_times` has been reached
        :param use_callcache: whether to use call caching
        :param use_reference_disks: whether to use reference disks
        :param memory_retry_multiplier: a multiplier for retrying with more memory
        :param max_n_entities: submit at most this many entities (random sample)
        :param dry_run: whether to skip updates to external data stores
        """

        # get the method config for this workflow in this workspace
        workflow_config = self.get_workflow_config(terra_workflow)

        assert not workflow_config["deleted"]
        assert workflow_config["rootEntityType"] == entity_type

        # identify columns in data table used for input/output if not provided
        if input_cols is None:
            input_cols = {
                v[5:]
                for k, v in workflow_config["inputs"].items()
                if v.startswith("this.")
            }

        if output_cols is None:
            output_cols = {
                v[5:]
                for k, v in workflow_config["outputs"].items()
                if v.startswith("this.")
            }

        # get the entities for this workflow entity type
        entities = self.get_entities(entity_type)

        # ensure columns exist to check for populated values
        for c in input_cols.union(output_cols):
            if c not in entities.columns:
                entities[c] = pd.NA
            else:
                entities[c] = entities[c].replace({"": pd.NA})

        # identify entities that have all required inputs but no outputs
        entities_todo = entities.loc[
            entities[list(input_cols)].notna().all(axis=1)
            & entities[list(output_cols)].isna().all(axis=1)
        ]

        if len(entities_todo) == 0:
            logging.info(f"No {entity_type}s to run {terra_workflow.method_name} for")
            return

        # get statuses of submitted workflows for the entitites
        state_counts = self.count_entity_workflow_states(
            entity_type,
            entity_ids=entities_todo[entity_id_col],
            terra_workflow=terra_workflow,
        )

        logging.info(f"Entity-workflow state counts: \n{state_counts}")

        # remove successful entities
        state_counts = state_counts.loc[state_counts["succeeded"].eq(0)]

        # remove runnning entities
        state_counts = state_counts.loc[
            (
                state_counts[["queued", "submitted", "launching", "running"]]
                .sum(axis=1)
                .eq(0)
            )
        ]

        # remove unretryable failures
        if not force_retry:
            if bool(state_counts["failed"].gt(resubmit_n_times).any()):
                logging.warning(
                    f"Some entities have failed more than {resubmit_n_times} times"
                )

            state_counts = state_counts.loc[state_counts["failed"].le(resubmit_n_times)]

        if len(state_counts) == 0:
            logging.info("No entities to submit")
            return
        elif max_n_entities is not None and len(state_counts) > max_n_entities:
            logging.info(f"Sampling {max_n_entities} of {len(state_counts)} entities")

            # prioritize entities with fewest previous failures, then randomly
            state_counts["rnd"] = np.random.rand(len(state_counts))
            state_counts = state_counts.sort_values(["failed", "rnd"])
            state_counts = state_counts.iloc[:max_n_entities]

        if dry_run:
            logging.info(f"(skipping) Submitting {terra_workflow.method_name} job")
            return

        entity_set_id = self.create_entity_set(
            entity_type,
            entity_ids=state_counts[entity_id_col],
            suffix=terra_workflow.method_name,
        )

        self.submit_workflow_run(
            terra_workflow=terra_workflow,
            entity=entity_set_id,
            etype=entity_set_type,
            expression=expression,
            use_callcache=use_callcache,
            use_reference_disks=use_reference_disks,
            memory_retry_multiplier=memory_retry_multiplier,
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
