dir=$(find . -mindepth 1 -maxdepth 1 -type d)

for d in $dir; do
 python3 ../view_loss.py --path $d/1_model_log.csv
done
