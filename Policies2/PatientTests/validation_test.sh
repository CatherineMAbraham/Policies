source activate softsurg
# Read the correct line from params_curr_compare.csv

python validation_tests_random.py \
                --model_path '/home/catherineabraham/Policies2/Policies2/TD3_Alg/contact/1/model-spring_randomYM_09031956_1' \
                --log 1\
                --safemode 0\
                --seed 11