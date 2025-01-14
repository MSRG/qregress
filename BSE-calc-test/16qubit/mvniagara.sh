# Check if "Full-CRX" is not in the directory name
dirs=$(find . -type d -mindepth 1 -maxdepth 1)
for d in $dirs; do
 if [[ "$d" != *"Full-CRX"* ]]; then
     echo "The string 'Full-CRX' is not in the directory name: $d"
     cp -r $d mvniagara/
 else
     echo "The string 'Full-CRX' is in the directory name: $d"
 fi
done
