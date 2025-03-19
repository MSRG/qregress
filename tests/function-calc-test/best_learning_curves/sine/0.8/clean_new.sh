sed -i 's/move2narval\/5_AL_lin16qubit/best_learning_curves\/sine\/0.7/g' run.sh
mv 0.7_sine_train.bin sine_train.bin
sed -i 's/5_AL_lin16qubit/sine_0.7/g' run.sh
sed -i 's/linear/sine/g' run.sh
