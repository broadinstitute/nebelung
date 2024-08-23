version 1.0

task print_message {
    input {
        String sample_id
        String message
    }

    command <<<
        echo "~{message}, ~{sample_id}" > out.txt
    >>>

    output {
        String result = read_string("out.txt")
    }

    runtime {
        docker: "debian:stable-slim"
        memory: "1 GiB"
        disks: "local-disk 10 SSD"
        cpu: 1
    }
}
