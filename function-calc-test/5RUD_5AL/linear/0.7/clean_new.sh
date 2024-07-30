sed -i 's/move2narval\/5_AL_lin16qubit/best_learning_curves\/linear\/0.7/g' run.sh
mv 0.7_linear_train.bin linear_train.bin
sed -i 's/5_AL_lin16qubit/linear_0.7/g' run.sh
