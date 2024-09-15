yes | python supcont_training.py

exit_code=$?

while [ $exit_code = 1 ]
do
    echo "=================== The script resulted in an error ==================="
    yes | python supcont_training.py
    exit_code=$?
done
echo "+++++++ Finished the job without errors +++++++++++"