set timeout -1
set PW $env(RSYNC_PASSWORD)
set LOAD_DIR "data/example/raw_data/h5/type=raw"
set SAVE_DIR "user@backup-host:/path/to/backup"

set run_numbers {1 2 3}

foreach run_number $run_numbers {
    set formatted_run_number [format %03d $run_number]
    set full_load_dir "$LOAD_DIR/run=$formatted_run_number"

    spawn rsync -auvz "$full_load_dir" "$SAVE_DIR"

    expect "password:"
    send "$PW\r"

    expect eof

    puts "rsync of run $formatted_run_number completed."
}

puts "All rsync operations completed."
