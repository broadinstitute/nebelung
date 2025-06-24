version 1.0

import "./other_tasks.wdl" as other_tasks

workflow hello_world {
    input {
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
