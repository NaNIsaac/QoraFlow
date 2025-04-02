set timeout -1
set PW "***REMOVED***"
set LOAD_DIR "/xfel/ffs/dat/ue_250324_FXS/raw_data/h5/type=raw"
set SAVE_DIR "portable2@10.4.133.1:/volume1/NetBackup"

set run_numbers {2 4 5 6 7 8 9 10}

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