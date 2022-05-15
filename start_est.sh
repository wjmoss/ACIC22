#! /bin/bash

svals=(0 500 1000 1500 2000 2500)

if [ ! -d "./log" ]; then
  mkdir log
fi

if [ ! -d "./res" ]; then
  mkdir res
fi

for ((i=0; i<${#svals[@]} ;i++))
do

  st=${svals[$i]}
  let end=st+500
  let st++
  
  fnm="./log/$st-$end.out"
  OMP_NUM_THREADS=1 nohup nice python3 est.py $st $end 2>&1 | tee $fnm &
  echo "OMP_NUM_THREADS=1 nohup nice python3 est.py $st $end 2>&1 | tee $fnm &"

done

st=3001
end=3400
fnm="./log/$st-$end.out"
OMP_NUM_THREADS=1 nohup nice python3 est.py $st $end 2>&1 | tee $fnm &
echo "OMP_NUM_THREADS=1 nohup nice python3 est.py $start $end 2>&1 | tee $fnm &"