log=bch.63.51.log
os=(-0.4 0 0.4 0.8 1 1.2 1.6 2 2.4)
rm -f $log
for ((i=0;i<${#os[@]};++i)); do
    python bp.py --offset ${os[i]} &>> $log
done
