from typing import Any, NotRequired, Optional, TypedDict, TypeVar

import pandera as pa
import pandera.typing
from pandera.api.pandas.model_config import BaseConfig as PaBaseConfig
from pydantic import BaseModel


class PersistedWdl(TypedDict):
    wdl: str
    public_url: str
    version: NotRequired[str | None]


class TerraJobSubmissionKwargs(TypedDict):
    entity: NotRequired[str | None]
    etype: NotRequired[str | None]
    expression: NotRequired[str | None]
    use_callcache: NotRequired[bool | None]
    delete_intermediate_output_files: NotRequired[bool | None]
    use_reference_disks: NotRequired[bool | None]
    memory_retry_multiplier: NotRequired[float | None]
    workflow_failure_mode: NotRequired[str | None]
    user_comment: NotRequired[str | None]


class TaskResult(BaseModel):
    completed_at: Optional[Any] = None
    crc32c_hash: Optional[str] = None
    format: Optional[str] = None
    id: Optional[str] = None
    label: Optional[str] = None
    size: Optional[int] = None
    task_entity_id: Optional[int] = None
    terra_entity_name: Optional[str] = None
    terra_entity_type: Optional[str] = None
    terra_method_config_name: Optional[str] = None
    terra_method_config_namespace: Optional[str] = None
    terra_submission_id: Optional[str] = None
    terra_workflow_id: Optional[str] = None
    terra_workflow_inputs: Optional[dict] = None
    terra_workflow_root_dir: Optional[str] = None
    terra_workspace_id: Optional[str] = None
    terra_workspace_name: Optional[str] = None
    terra_workspace_namespace: Optional[str] = None
    url: Optional[str] = None
    value: Optional[dict] = None
    workflow_name: Optional[str] = None
    workflow_source_url: Optional[str] = None
    workflow_version: Optional[str] = None


class CoercedDataFrame(pa.DataFrameModel):
    class Config(PaBaseConfig):
        coerce = True  # convert to indicated dtype upon TypedDataFrame init


PanderaBaseSchema = TypeVar("PanderaBaseSchema", bound=CoercedDataFrame)
TypedDataFrame = pandera.typing.DataFrame
T = TypeVar("T")
