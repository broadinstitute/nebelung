import json
import logging
import os
import tempfile
from pathlib import Path

from firecloud import api as firecloud_api
from firecloud.api import __post as firecloud_post

from nebelung.types import PersistedWdl
from nebelung.utils import call_firecloud_api
from nebelung.wdl import GistedWdl


class TerraWorkflow:
    def __init__(
        self,
        method_namespace: str,
        method_name: str,
        method_config_namespace: str,
        method_config_name: str,
        method_synopsis: str,
        workflow_wdl_path: Path,
        method_config_json_path: Path,
        github_pat: str | None = None,
        womtool_jar: str | None = None,
    ) -> None:
        self.method_namespace = method_namespace
        self.method_name = method_name
        self.method_config_namespace = method_config_namespace
        self.method_config_name = method_config_name
        self.method_synopsis = method_synopsis
        self.workflow_wdl_path = workflow_wdl_path
        self.method_config_json_path = method_config_json_path
        self.github_pat = os.getenv("GITHUB_PAT", github_pat)
        self.womtool_jar = os.getenv("WOMTOOL_JAR", womtool_jar)

        self.method_config = json.load(open(self.method_config_json_path, "r"))
        self.persisted_wdl_script: PersistedWdl | None = None

    def persist_method_on_github(self) -> None:
        """
        Upload the method's WDL script to GitHub, rewriting import statements for
        dependent WDL scripts as needed.
        """

        assert self.github_pat is not None, (
            "A GitHub personal access token must be defined to persist this method on "
            "GitHub. Set the GITHUB_PAT environment variable or the `github_pat` "
            "argument when instantiating this `TerraWorkflow` instance."
        )

        assert self.womtool_jar is not None, (
            "A path to a WOMTool .jar file is required to validate the WDL script. Set "
            "the WOMTOOL_JAR environment variable or the `womtool_jar` argument when "
            "instantiating this `TerraWorkflow` instance."
        )

        if self.persisted_wdl_script is None:
            logging.info(f"Persisting {self.workflow_wdl_path} on GitHub")
            gisted_wdl = GistedWdl(
                method_name=self.method_name,
                github_pat=self.github_pat,
                womtool_jar=self.womtool_jar,
            )
            self.persisted_wdl_script = gisted_wdl.persist_wdl_script(
                wdl_path=self.workflow_wdl_path
            )

    def update_method(self, owners: list[str]) -> dict:
        """
        Update a Firecloud method.

        :param owners: a list of Firecloud users/groups to set as owners
        :return: the latest method's snapshot
        """

        # get contents of WDL uploaded to GCS
        self.persist_method_on_github()
        assert self.persisted_wdl_script is not None

        with tempfile.NamedTemporaryFile("w") as f:
            f.write(self.persisted_wdl_script["wdl"])
            f.flush()

            logging.info("Updating method")
            snapshot = call_firecloud_api(
                firecloud_api.update_repository_method,
                namespace=self.method_namespace,
                method=self.method_name,
                synopsis=self.method_synopsis,
                wdl=f.name,
            )

        logging.info("Setting method ACL")
        call_firecloud_api(
            firecloud_api.update_repository_method_acl,
            namespace=self.method_namespace,
            method=self.method_name,
            snapshot_id=snapshot["snapshotId"],
            acl_updates=[{"user": x, "role": "OWNER"} for x in owners],
        )

        logging.info("Setting method repository config ACL")
        # the firecloud package doesn't have a wrapper for this endpoint
        call_firecloud_api(
            firecloud_post,
            methcall=f"configurations/{self.method_namespace}/permissions",
            json=[{"user": x, "role": "OWNER"} for x in owners],
        )

        return snapshot

    def get_method_snapshots(self) -> list[dict]:
        """
        Get all of the snapshots of the method.

        :return: list of snapshot information, most recent first
        """

        logging.info(f"Getting {self.method_name} method snapshots")
        snapshots = call_firecloud_api(
            firecloud_api.list_repository_methods,
            namespace=self.method_namespace,
            name=self.method_name,
        )

        snapshots.sort(key=lambda x: x["snapshotId"], reverse=True)
        return snapshots

    def delete_old_method_snapshots(self, n_snapshots_to_keep: int) -> None:
        """
        Delete all but `n_snapshots_to_keep` of the most recent snapshots of the method.

        :param n_snapshots_to_keep: the number of snapshots to keep
        """

        snapshots = self.get_method_snapshots()

        to_delete = snapshots[n_snapshots_to_keep:]
        logging.info(f"Deleting {len(to_delete)} old snapshot(s)")

        for x in to_delete:
            call_firecloud_api(
                firecloud_api.delete_repository_method,
                namespace=self.method_namespace,
                name=self.method_name,
                snapshot_id=x["snapshotId"],
            )
