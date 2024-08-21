from __future__ import annotations

import json
import logging
from pathlib import Path

from firecloud import api as firecloud_api

from nebelung.types import PersistedWdl
from nebelung.utils import call_firecloud_api
from nebelung.wdl import GistedWdl


class TerraWorkflow:
    def __init__(
        self,
        github_pat: str,
        pipelines_bucket_name: str,
        repo_namespace: str,
        repo_method_name: str,
        method_config_name: str,
        method_synopsis: str,
        workflow_wdl_path: Path,
        method_config_json_path: Path,
    ) -> None:
        self.github_pat = github_pat
        self.pipelines_bucket_name = pipelines_bucket_name
        self.repo_namespace = repo_namespace
        self.repo_method_name = repo_method_name
        self.method_config_name = method_config_name
        self.method_synopsis = method_synopsis
        self.workflow_wdl_path = workflow_wdl_path
        self.method_config_json_path = method_config_json_path
        self.method_config = json.load(open(self.method_config_json_path, "r"))
        self.persisted_wdl_script: PersistedWdl | None = None

    def persist_method_on_github(self) -> None:
        """
        Upload the method's WDL script to GitHub, rewriting import statements for
        dependent WDL scripts as needed.
        """

        if self.persisted_wdl_script is None:
            logging.info(f"Persisting {self.workflow_wdl_path} on GitHub")
            gisted_wdl = GistedWdl(
                method_name=self.repo_method_name, github_pat=self.github_pat
            )
            self.persisted_wdl_script = gisted_wdl.persist_wdl_script(
                wdl_path=self.workflow_wdl_path, subpath="wdl"
            )

    def get_method_snapshots(self) -> list[dict]:
        """
        Get all of the snapshots of the method.

        :return: list of snapshot information, most recent first
        """

        logging.info(f"Getting {self.repo_method_name} method snapshots")
        snapshots = call_firecloud_api(
            firecloud_api.list_repository_methods,
            namespace=self.repo_namespace,
            name=self.repo_method_name,
        )

        snapshots.sort(key=lambda x: x["snapshotId"], reverse=True)
        return snapshots

    def delete_old_method_snapshots(self, nkeep: int) -> None:
        """
        Delete all but `n` of the most recent snapshots of the method. This might fail
        if the service account doesn't have OWNER permission on the method configuration
        namespace. This can't be set using the firecloud package, so POST to the
        `setConfigNamespaceACL` endpoint on https://api.firecloud.org/ once to resolve
        this problem.

        :param nkeep: the number of snapshots to keep
        """

        snapshots = self.get_method_snapshots()

        to_delete = snapshots[nkeep:]
        logging.info(f"Deleting {len(to_delete)} old snapshot(s)")

        for x in to_delete:
            call_firecloud_api(
                firecloud_api.delete_repository_method,
                namespace=self.repo_namespace,
                name=self.repo_method_name,
                snapshot_id=x["snapshotId"],
            )
