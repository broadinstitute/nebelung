from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import github

from nebelung.types import PersistedWdl

IMPORT_PATTERN = re.compile(r"^import\s+\"(?!http)([^\"]+)\"\s+as\s+(\S+)")
WORKFLOW_VERSION_PATTERN = re.compile(
    r"^\s*String\s+workflow_version\s*=\s*\"([a-zA-Z0-9.]+)\""
)


class GistedWdl:
    def __init__(self, method_name: str, github_pat: str):
        self.method_name = method_name
        self.github_pat = github_pat
        self.gist = None

    def find_or_create_gist(self):
        """
        Find or create a GitHub gist for a Terra method.
        """

        github_auth = github.Auth.Token(self.github_pat)
        g = github.Github(auth=github_auth)

        # check if there's already a gist for this method by comparing the descriptions
        gists = g.get_user().get_gists()
        matching_gists = [x for x in gists if x.description == self.method_name]

        if len(matching_gists) == 0:
            self.gist = g.get_user().create_gist(  # pyright: ignore
                description=self.method_name,
                files={
                    "README.md": github.InputFileContent(
                        content=f"Persisted WDL for the `{self.method_name}` method"
                    )
                },
                public=True,
            )
        else:
            self.gist = matching_gists[0]

    def persist_wdl_script(self, wdl_path: Path, subpath: str = "") -> PersistedWdl:
        """
        Save a copy of a local WDL script and its dependencies to GitHub, rewriting the
        import statements to refer to the public gist.github.com URLs.

        :param wdl_path: the absolute path to a WDL script
        :param subpath: a prefix to prepend to the destiation URL
        :return: a dictionary of the uploaded WDL, its public URL, and its
        `pipeline_version` (if found as a variable in the `workflow`)
        """

        with open(wdl_path, "r") as f:
            wdl_lines = f.readlines()

        wdl_basename = os.path.basename(wdl_path)
        version = None
        versioned_subpath = subpath
        buffer = []

        # include the version number on the uploaded object path if one is found
        for line in wdl_lines:
            if version_match := re.match(WORKFLOW_VERSION_PATTERN, line):
                version = version_match[1]
                versioned_subpath = "/".join(
                    [versioned_subpath, Path(wdl_basename).stem, version]
                )

        for line in wdl_lines:
            if import_match := re.match(IMPORT_PATTERN, line):
                # need to upload the dependent WDL file to GCS and rewrite the `import`
                # statement in this WDL file to the dependent file's public GCS URL

                rel_path = import_match[1]  # relative path to the dependent file
                import_alias = import_match[2]  # `as <x>` component of the statement

                # absolute path to the dependent file
                abs_wdl_path = Path.joinpath(wdl_path.parent.absolute(), rel_path)

                # append the local relative path to the end of parent's prefix
                imported_versioned_subpath = "/".join(
                    [versioned_subpath, Path(rel_path).parent.name]
                )

                # recurse: upload the dependent WDL and get its URL
                imported_public_url = self.persist_wdl_script(
                    abs_wdl_path, imported_versioned_subpath
                )["public_url"]

                # replace the local relative import with the absolute one
                converted_line = f'import "{imported_public_url}" as {import_alias}'
                buffer.append(converted_line)

            else:
                buffer.append(line.rstrip())

        # construct the final version of this WDL script
        converted_wdl = "\n".join(buffer)

        # validate the WDL before uploading
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(converted_wdl)
            f.flush()

            res = subprocess.run(
                ["java", "-jar", os.environ["WOMTOOL_JAR"], "validate", f.name],
                capture_output=True,
            )

            if res.returncode != 0 or "Success" not in res.stdout.decode():
                raise ChildProcessError(
                    f"Error validating WDL script: {res.stderr.decode()}"
                )

        # update the gist with the new version of the WDL script
        if self.gist is None:
            self.find_or_create_gist()

        assert self.gist is not None

        self.gist.edit(
            files={wdl_basename: github.InputFileContent(content=converted_wdl)}
        )

        # return the public URL of the new version of the WDL script so that it can be
        # imported by its parent (if there is one)
        raw_url = self.gist.files[wdl_basename].raw_data["raw_url"]

        return {"wdl": converted_wdl, "public_url": raw_url, "version": version}
