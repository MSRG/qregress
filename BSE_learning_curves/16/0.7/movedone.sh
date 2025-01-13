finished=$(find . -name "*results.json")

for i in $finished; do
  fin=$(dirname $i)
  echo $fin
  mv -f $fin finished/
done
