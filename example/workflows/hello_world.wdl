version 1.0

import "./other_tasks.wdl" as other_tasks

workflow hello_world {
    input {
        String workflow_version = "1.0" # internal semver not tied to WARP releases
        String workflow_source_url # populated automatically with URL of this script
        String sample_id
    }

    call other_tasks.print_message {
        input:
            sample_id = sample_id,
            message = "Hello"
    }

    output {
        String result = print_message.result
    }
}
