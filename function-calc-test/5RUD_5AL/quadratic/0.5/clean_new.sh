sed -i 's/move2narval\/5_AL_lin16qubit/best_learning_curves\/quadratic\/0.5/g' run.sh
mv 0.5_quadratic_train.bin quadratic_train.bin
sed -i 's/5_AL_lin16qubit/quadratic_0.5/g' run.sh
sed -i 's/linear/quadratic/g' run.sh
