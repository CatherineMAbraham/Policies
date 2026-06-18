source activate softsurg
# Read the correct line from params_curr_compare.csv
for TASK_ID in {1..11}; do
    PARAM_LINE=$(sed -n "${TASK_ID}p" tests.csv)
    IFS=',' read -r FILE YOUNGS_MODULUS EXPERT <<< "$PARAM_LINE"
    echo "Running test with: Young's Modulus=$YOUNGS_MODULUS, VTK File=$FILE, Expert Trajectory=$EXPERT"
    # Run the script 
    python force_from_expert.py --threshold_pos 0.0001 --threshold_ori 0.5 --maxforce 4 --softtissue soft --youngs_modulus $YOUNGS_MODULUS --vtk_file $FILE --expert $EXPERT --log 0
done
